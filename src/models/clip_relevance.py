from pathlib import Path
from typing import Sequence

import decord
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import os, hashlib, pickle

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int_]

CACHE_DIR = Path("./cache/clip_embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _frame_cache_key(video_path: str, frame_indices: tuple) -> str:
    """Cache key based on video and specific frame indices, not boundaries."""
    h = hashlib.md5(f"{video_path}_{frame_indices}".encode()).hexdigest()
    return h

class CLIPRelevanceScorer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        frames_per_event: int = 4,
    ):
        self.model_name = model_name
        self.frames_per_event = frames_per_event

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, dim=-1)

        return text_features

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if len(images) == 0:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)

        return image_features

    def load_event_frames(
        self,
        video_path: str,
        start: int,
        end: int,
        num_frames: int,
    ) -> list[Image.Image]:
        if end <= start or num_frames <= 0:
            return []

        vr = decord.VideoReader(str(Path(video_path)))

        n = min(num_frames, end - start)
        indices = np.linspace(start, end - 1, n, dtype=int)

        batch = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(arr) for arr in batch]

    @torch.no_grad()
    def compute_event_embeddings(
        self,
        video_path: str,
        boundaries: Sequence[int] | IntArray,
    ) -> list[FloatArray]:
        b = np.asarray(boundaries, dtype=int)
        event_embeddings_list = []

        # 각 event 별로 frame indices 미리 계산
        all_event_indices = []
        for i in range(len(b) - 1):
            start, end = int(b[i]), int(b[i + 1])
            if end <= start:
                all_event_indices.append([])
            else:
                n = min(self.frames_per_event, end - start)
                indices = np.linspace(start, end - 1, n, dtype=int).tolist()
                all_event_indices.append(indices)

        # Cache key: video + 모든 frame indices의 flat tuple
        flat_indices = tuple(idx for ev in all_event_indices for idx in ev)
        cache_key = _frame_cache_key(video_path, flat_indices)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # 새로 계산
        for indices in all_event_indices:
            if len(indices) == 0:
                event_embeddings_list.append(
                    np.zeros((1, self.model.config.projection_dim), dtype=np.float32)
                )
                continue
            
            vr = decord.VideoReader(str(Path(video_path)))
            batch = vr.get_batch(indices).asnumpy()
            frames = [Image.fromarray(arr) for arr in batch]
            
            frame_features = self.encode_images(frames)
            event_embeddings_list.append(
                frame_features.detach().cpu().numpy().astype(np.float32)
            )

        # 저장
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(event_embeddings_list, f)
        except Exception as e:
            print(f"[Warning] Cache save failed: {e}")

        return event_embeddings_list
    @torch.no_grad()
    def compute_query_relevance(
        self,
        question: str,
        options: Sequence[str],                  
        event_embeddings: list[FloatArray],      
        temperature: float = 0.07,
    ) -> FloatArray:
        """
        Compute event relevance using MAX over per-frame similarities.
        Options are NOT used (CLIP은 4지선다 정답 고르는 모델이 아님; 
        Qwen2-VL이 그 역할을 함).
        """
        if len(event_embeddings) == 0:
            return np.array([], dtype=np.float32)

        # Question만 사용 (options 제외)
        text_feature = self.encode_text(question)        # [1, D]
        text_feature = text_feature.squeeze(0)            # [D]

        # 각 event의 max-similarity 계산
        event_max_sims = []
        for event_frames_emb in event_embeddings:         # event_frames_emb: [N_frames, D]
            if event_frames_emb.shape[0] == 0:
                event_max_sims.append(0.0)
                continue
            
            # numpy → torch
            event_tensor = torch.from_numpy(event_frames_emb).to(self.device)
            event_tensor = F.normalize(event_tensor, dim=-1)
            
            # frame별 similarity 계산
            sims = event_tensor @ text_feature             # [N_frames]
            
            # event의 representative similarity = MAX (NOT mean)
            max_sim = sims.max().item()
            event_max_sims.append(max_sim)

        similarity_array = np.array(event_max_sims, dtype=np.float32)
        
        # softmax로 relevance 분포화
        similarity_tensor = torch.from_numpy(similarity_array).to(self.device)
        relevance = torch.softmax(similarity_tensor / temperature, dim=0)

        return relevance.detach().cpu().numpy().astype(np.float32)