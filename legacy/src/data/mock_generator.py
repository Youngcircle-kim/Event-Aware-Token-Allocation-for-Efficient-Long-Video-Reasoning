"""
Mock data generators for offline development.

We simulate videos as sequences of "latent frame embeddings" rather than
real pixel data. This lets us test all algorithmic logic end-to-end without
any GPU or actual video files.

Each mock video is constructed by concatenating K "event blobs" in latent
space, where each blob has its own center and spread. This gives us
controllable ground-truth event structure to validate segmentation and
allocation algorithms against.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.utils.types import VideoQAExample


@dataclass
class MockVideoSpec:
    """Ground-truth spec used to generate a mock video."""
    video_id: str
    duration: float                     # seconds
    fps: int                             # frames per second (raw video)
    event_boundaries: List[float]       # seconds; len = K+1 (start of each event + end)
    event_centers: np.ndarray            # (K, D) latent centers per event
    event_spreads: np.ndarray            # (K,) std-dev within each event
    event_motion: np.ndarray             # (K,) "motion intensity" per event
    event_object_density: np.ndarray     # (K,) "object count" per event
    answer_event_idx: int                # index of event containing the GT answer
    embedding_dim: int = 64

    @property
    def num_events(self):
        return len(self.event_centers)

    @property
    def num_frames(self):
        return int(self.duration * self.fps)


class MockVideoGenerator:
    """
    Generates synthetic videos with controllable event structure.

    The video is represented as a (T, D) array of "raw frame latents"
    — these stand in for what would otherwise be pixel frames.
    A mock vision encoder (in src/models/mock_encoder.py) converts these
    to frame embeddings.
    """

    def __init__(self, embedding_dim: int = 64, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)

    def sample_video_spec(
        self,
        video_id: str,
        duration_range: Tuple[float, float] = (60.0, 300.0),
        num_events_range: Tuple[int, int] = (3, 10),
        fps: int = 30,
    ) -> MockVideoSpec:
        """Sample a random video spec with K events."""
        duration = self.rng.uniform(*duration_range)
        num_events = self.rng.integers(*num_events_range, endpoint=True)

        # Sample event boundaries (random partition of [0, duration])
        internal_boundaries = sorted(
            self.rng.uniform(0, duration, size=num_events - 1).tolist()
        )
        event_boundaries = [0.0] + internal_boundaries + [duration]

        # Sample event centers in latent space (well-separated)
        event_centers = self.rng.normal(
            0, 2.0, size=(num_events, self.embedding_dim)
        )

        # Within-event spread (how noisy is the event internally)
        event_spreads = self.rng.uniform(0.05, 0.3, size=num_events)

        # Motion intensity (some events are dynamic, others static)
        event_motion = self.rng.uniform(0.1, 1.0, size=num_events)

        # Object density
        event_object_density = self.rng.integers(1, 10, size=num_events).astype(float)

        # Randomly pick which event contains the GT answer
        answer_event_idx = int(self.rng.integers(0, num_events))

        return MockVideoSpec(
            video_id=video_id,
            duration=duration,
            fps=fps,
            event_boundaries=event_boundaries,
            event_centers=event_centers,
            event_spreads=event_spreads,
            event_motion=event_motion,
            event_object_density=event_object_density,
            answer_event_idx=answer_event_idx,
            embedding_dim=self.embedding_dim,
        )

    def render_frame_latents(
        self,
        spec: MockVideoSpec,
        sample_fps: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce raw "frame latents" at the given sampling rate.

        Returns:
            latents: (N, D) array of per-frame latents
            timestamps: (N,) array of timestamps in seconds
        """
        n_samples = int(spec.duration * sample_fps)
        timestamps = np.linspace(0, spec.duration, n_samples, endpoint=False)

        latents = np.zeros((n_samples, spec.embedding_dim))

        for i, t in enumerate(timestamps):
            # Find which event this timestamp belongs to
            event_idx = self._find_event(t, spec.event_boundaries)
            center = spec.event_centers[event_idx]
            spread = spec.event_spreads[event_idx]
            # Sample a noisy latent around the event center
            latents[i] = center + self.rng.normal(0, spread, size=spec.embedding_dim)

        return latents, timestamps

    @staticmethod
    def _find_event(t: float, boundaries: List[float]) -> int:
        """Return the event index for a given timestamp."""
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= t < boundaries[i + 1]:
                return i
        return len(boundaries) - 2  # last event


# =============================================================================
# Mock QA example generator
# =============================================================================

class MockQAGenerator:
    """Generates a VideoQAExample paired with a MockVideoSpec."""

    TEMPLATES = [
        ("What happens in scene {e}?", "Event {e} occurred."),
        ("What object appears at {t}s?", "Object {e}."),
        ("What is the main action during segment {e}?", "Action of event {e}."),
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def make_example(self, spec: MockVideoSpec) -> VideoQAExample:
        e = spec.answer_event_idx
        event_start = spec.event_boundaries[e]
        event_end = spec.event_boundaries[e + 1]
        midpoint = (event_start + event_end) / 2

        template_q, template_a = self.TEMPLATES[
            self.rng.integers(len(self.TEMPLATES))
        ]
        question = template_q.format(e=e, t=int(midpoint))

        # Generate 4 multiple-choice options; correct answer is "A" for simplicity
        correct = template_a.format(e=e)
        distractors = [
            template_a.format(e=j) for j in range(spec.num_events) if j != e
        ][:3]
        # Pad if too few events
        while len(distractors) < 3:
            distractors.append(f"Unrelated option {len(distractors)}")

        options_text = [correct] + distractors
        options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_text)]

        return VideoQAExample(
            video_id=spec.video_id,
            video_path=f"mock://{spec.video_id}",
            duration=spec.duration,
            question=question,
            options=options,
            answer="A",  # correct answer is always A in mock
            gt_timestamp=(event_start, event_end),
            task_type="mock_temporal",
            dataset="mock",
        )


# =============================================================================
# Full dataset generator
# =============================================================================

def generate_mock_dataset(
    n_examples: int = 20,
    embedding_dim: int = 64,
    seed: int = 42,
) -> List[Tuple[MockVideoSpec, VideoQAExample]]:
    """Generate a small synthetic dataset for offline testing."""
    vid_gen = MockVideoGenerator(embedding_dim=embedding_dim, seed=seed)
    qa_gen = MockQAGenerator(seed=seed + 1)
    data = []
    for i in range(n_examples):
        spec = vid_gen.sample_video_spec(video_id=f"mock_vid_{i:03d}")
        example = qa_gen.make_example(spec)
        data.append((spec, example))
    return data
