"""
Mock vision-language encoder.

Interface matches what open_clip / SigLIP would provide:
  - encode_frames(frames) -> (N, D) embeddings
  - encode_text(text) -> (D,) text embedding

In mock mode, "frames" are actually latent vectors from MockVideoGenerator.
We simulate the encoder by applying a fixed linear projection + normalization,
which mimics CLIP's property of producing aligned text/image embeddings.

# TODO: REAL
Replace MockCLIPEncoder with a real open_clip wrapper. The same interface
is preserved so downstream code doesn't change.
"""
from typing import List, Optional
import numpy as np


class MockCLIPEncoder:
    """Simulated CLIP/SigLIP encoder for offline development."""

    def __init__(
        self,
        input_dim: int = 64,
        embed_dim: int = 128,
        seed: int = 0,
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)
        # Fixed projection matrices (shared image/text space)
        self._image_proj = rng.normal(0, 1 / np.sqrt(input_dim),
                                       size=(input_dim, embed_dim))
        # Text projection: operates on a word-hash representation
        self._text_vocab_size = 1000
        self._text_emb = rng.normal(0, 1 / np.sqrt(embed_dim),
                                     size=(self._text_vocab_size, embed_dim))

    @staticmethod
    def _l2_normalize(x: np.ndarray, axis=-1, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)

    # --------- image/video encoding ---------
    def encode_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (N, input_dim) latent array from mock video.
                    In real mode, this would be (N, H, W, 3) pixels.
        Returns:
            (N, embed_dim) normalized embeddings.
        """
        if frames.ndim != 2 or frames.shape[-1] != self.input_dim:
            raise ValueError(
                f"Mock encoder expects (N, {self.input_dim}); got {frames.shape}"
            )
        emb = frames @ self._image_proj
        return self._l2_normalize(emb, axis=-1)

    # --------- text encoding ---------
    def encode_text(self, text: str) -> np.ndarray:
        """
        Deterministic hash-based text encoding for mock.

        Real implementation uses CLIP's text encoder; here we average
        per-token embeddings where tokens are just hashed words.
        """
        words = text.lower().split()
        if not words:
            return np.zeros(self.embed_dim)
        indices = [hash(w) % self._text_vocab_size for w in words]
        emb = self._text_emb[indices].mean(axis=0)
        return self._l2_normalize(emb, axis=-1)

    # --------- similarity ---------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two (already-normalized) vectors."""
        return float(np.dot(a, b))
