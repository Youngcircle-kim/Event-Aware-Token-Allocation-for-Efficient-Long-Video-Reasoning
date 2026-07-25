# src/evalloc/inference/prompt_builder.py

from __future__ import annotations

from dataclasses import dataclass

from evalloc.data.base import QASample


OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_multiple_choice_prompt(
    question: str,
    options: list[str],
    *,
    answer_instruction: str = "Answer using only the option letter.",
) -> str:
    """
    Build a multiple-choice prompt.

    Args:
        question:
            Question text.
        options:
            Option texts without labels.
            Example: ["Sit down", "Open the door", "Leave"]
        answer_instruction:
            Instruction appended after the choices.

    Returns:
        Formatted prompt string.
    """
    question = question.strip()

    if not question:
        raise ValueError("question must not be empty.")

    if not options:
        raise ValueError("options must not be empty.")

    if len(options) > len(OPTION_LABELS):
        raise ValueError(
            f"Too many options: {len(options)}. "
            f"Maximum supported options: {len(OPTION_LABELS)}"
        )

    formatted_options = "\n".join(
        f"{OPTION_LABELS[index]}. {option.strip()}"
        for index, option in enumerate(options)
    )

    return (
        "Watch the provided video frames in chronological order and "
        "answer the multiple-choice question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{formatted_options}\n\n"
        f"{answer_instruction.strip()}"
    )


def build_open_ended_prompt(
    question: str,
    *,
    answer_instruction: str = "Answer the question concisely.",
) -> str:
    """
    Build an open-ended QA prompt.
    """
    question = question.strip()

    if not question:
        raise ValueError("question must not be empty.")

    return (
        "Watch the provided video frames in chronological order and "
        "answer the question.\n\n"
        f"Question:\n{question}\n\n"
        f"{answer_instruction.strip()}"
    )


@dataclass(frozen=True)
class PromptBuilder:
    """
    Converts QASample objects into model-ready text prompts.

    The builder only handles text formatting. Visual inputs are attached
    separately by the model inference module.
    """

    multiple_choice_instruction: str = "Answer using only the option letter."
    open_ended_instruction: str = "Answer the question concisely."

    def build(self, sample: QASample) -> str:
        if sample.is_multi_choice():
            if not sample.has_options():
                raise ValueError(
                    f"Multiple-choice sample {sample.sample_id} has no options."
                )

            assert sample.options is not None

            return build_multiple_choice_prompt(
                question=sample.question,
                options=sample.options,
                answer_instruction=self.multiple_choice_instruction,
            )

        if sample.task_type == "open_ended":
            return build_open_ended_prompt(
                question=sample.question,
                answer_instruction=self.open_ended_instruction,
            )

        raise ValueError(
            f"Unsupported task_type '{sample.task_type}' "
            f"for sample {sample.sample_id}."
        )