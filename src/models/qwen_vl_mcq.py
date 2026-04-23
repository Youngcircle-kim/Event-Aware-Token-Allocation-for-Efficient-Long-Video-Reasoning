from __future__ import annotations

import re
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_mcq_letter(text: str, options: Optional[List[str]] = None) -> str:
    if not text:
        return "A"

    m = LETTER_RE.search(text.strip())
    if m:
        return m.group(1).upper()

    upper = text.upper().strip()
    for letter in ["A", "B", "C", "D"]:
        if upper == letter or upper.startswith(f"{letter}.") or upper.startswith(f"{letter})"):
            return letter

    if options:
        for idx, opt in enumerate(options[:4]):
            if opt and opt.lower() in text.lower():
                return chr(ord("A") + idx)

    return "A"


def build_mcq_prompt(question: str, options: List[str]) -> str:
    option_lines = []
    for i, opt in enumerate(options[:4]):
        letter = chr(ord("A") + i)
        option_lines.append(f"{letter}. {opt}")

    joined_options = "\n".join(option_lines)

    return (
        "You are answering a multiple-choice question about a video.\n"
        "Look carefully at the provided video frames and answer with only one letter: A, B, C, or D.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{joined_options}\n\n"
        "Return only the single best answer letter."
    )


class QwenVLMCQ:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 16,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[QwenVLMCQ] Loading model from: {model_name_or_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model.eval()

    def answer_mcq(
        self,
        frames: List[Image.Image],
        question: str,
        options: List[str],
    ) -> dict:
        if not frames:
            return {
                "predicted_answer": "A",
                "raw_output": "[EMPTY_FRAMES]",
            }

        prompt = build_mcq_prompt(question, options)

        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        decoded = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        pred = parse_mcq_letter(decoded, options)

        return {
            "predicted_answer": pred,
            "raw_output": decoded,
        }