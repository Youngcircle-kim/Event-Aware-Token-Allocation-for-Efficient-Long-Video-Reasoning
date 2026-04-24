from pathlib import Path
from typing import Sequence

import decord
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int_]


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
    ) -> FloatArray:
        b = np.asarray(boundaries, dtype=int)
        event_embeddings = []

        for i in range(len(b) - 1):
            start = int(b[i])
            end = int(b[i + 1])

            frames = self.load_event_frames(
                video_path=video_path,
                start=start,
                end=end,
                num_frames=self.frames_per_event,
            )

            if len(frames) == 0:
                event_embeddings.append(
                    torch.zeros(self.model.config.projection_dim, device=self.device)
                )
                continue

            frame_features = self.encode_images(frames)

            event_feature = frame_features.mean(dim=0)
            event_feature = F.normalize(event_feature, dim=-1)

            event_embeddings.append(event_feature)

        if len(event_embeddings) == 0:
            return np.empty((0, self.model.config.projection_dim), dtype=np.float32)

        event_embeddings_tensor = torch.stack(event_embeddings, dim=0)

        return event_embeddings_tensor.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def compute_query_relevance(
        self,
        question: str,
        options: Sequence[str],
        event_embeddings: FloatArray,
        temperature: float = 0.07,
    ) -> FloatArray:
        if event_embeddings.size == 0:
            return np.array([], dtype=np.float32)

        # MCQ에서는 question만 쓰는 것보다 options까지 같이 넣는 게 더 안정적임
        option_text = " ".join([str(opt) for opt in options])
        query_text = f"{question} {option_text}"

        text_feature = self.encode_text(query_text)

        event_tensor = torch.from_numpy(event_embeddings).to(self.device)
        event_tensor = F.normalize(event_tensor, dim=-1)

        similarity = event_tensor @ text_feature.squeeze(0)

        # softmax를 쓰면 event 간 차이가 allocation에 반영됨
        relevance = torch.softmax(similarity / temperature, dim=0)

        return relevance.detach().cpu().numpy().astype(np.float32)