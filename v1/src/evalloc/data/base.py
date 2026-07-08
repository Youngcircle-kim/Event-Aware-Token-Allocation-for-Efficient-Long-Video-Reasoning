from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskType  = Literal["multi_choice", "open_ended"]

@dataclass
class QASample:
    sample_id: str
    video_path: Path
    question: str
    task_type: TaskType = "multi_choice"
    options: list[str] | None = None
    answer: str | None = None
    duration: float | None = None
    metadata: dict[str,  Any] = field(default_factory=dict)

    def is_multi_choice(self) -> bool:
        return self.task_type == "multi_choice"

    def is_open_ended(self) -> bool:
        return self.task_type == "open_ended"

    def has_options(self) -> bool:
        return self.options is not None and len(self.options) > 0

    def has_answer(self) -> bool:
        return self.answer is not None

    @property
    def dataset_name(self) -> str | None:
        return self.metadata.get("dataset")

    @property
    def question_type(self) -> str | None:
        return self.metadata.get("question_type")

    @property
    def evidence_timestamps(self) -> list[tuple[float, float]] | None:
        return self.metadata.get("evidence_timestamps")
    
    def validate(self, require_answer: bool = True) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required.")

        if not self.video_path:
            raise ValueError(f"video_path is required for sample {self.sample_id}.")
        
        if not self.question:
            raise ValueError(f"question is required for sample {self.sample_id}.")

        if self.is_multi_choice() and not self.has_options():
            raise ValueError(
                f"Multi-choice sample {self.sample_id} must have options."
            )
        if require_answer and not self.has_answer():
            raise ValueError(
                f"Sample {self.sample_id} must have an answer for evaluation."
            )
        
class BaseQADataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def __getitem__(self, idx: int) -> QASample:
        raise NotImplementedError