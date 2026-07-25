from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class VideoFeatures:
    """
    Candidate frames and their visual embeddings.

    frame_indices:
        Original video frame indices.

    timestamps:
        Candidate frame timestamps in seconds.

    features:
        L2-normalized CLIP image features with shape [N, D].
        The tensor is stored on CPU to reduce persistent GPU usage.
    """

    frame_indices: list[int]
    timestamps: list[float]
    features: torch.Tensor

    def __post_init__(self) -> None:
        num_candidates = len(self.frame_indices)

        if len(self.timestamps) != num_candidates:
            raise ValueError(
                "frame_indices and timestamps must have the same length."
            )

        if self.features.ndim != 2:
            raise ValueError(
                f"features must have shape [N, D], got {self.features.shape}."
            )

        if self.features.shape[0] != num_candidates:
            raise ValueError(
                "The number of features must match the number of frames."
            )


class CLIPFeatureExtractor:
    """
    CLIP image/text feature extractor.

    Image and text embeddings are projected into the same embedding space,
    allowing cosine similarity between a question and frames/events.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch_size = batch_size

        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model = CLIPModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.model.eval()

    @torch.inference_mode()
    def encode_images(
        self,
        images: Sequence[Image.Image],
    ) -> torch.Tensor:
        if not images:
            raise ValueError("At least one image is required.")

        self.load()

        assert self.model is not None
        assert self.processor is not None

        feature_batches: list[torch.Tensor] = []

        for start in range(0, len(images), self.batch_size):
            batch_images = images[start : start + self.batch_size]

            inputs = self.processor(
                images=list(batch_images),
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to(
                device=self.device,
                dtype=self.dtype,
            )

            features = self.model.get_image_features(
                pixel_values=pixel_values
            )

            features = F.normalize(
                features.float(),
                dim=-1,
            )

            feature_batches.append(features.cpu())

        return torch.cat(feature_batches, dim=0)

    @torch.inference_mode()
    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        self.load()

        assert self.model is not None
        assert self.processor is not None

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        features = self.model.get_text_features(**inputs)
        features = F.normalize(features.float(), dim=-1)

        return features[0].cpu()