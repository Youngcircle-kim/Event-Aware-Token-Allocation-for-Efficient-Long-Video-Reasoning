from __future__ import annotations

from collections.abc import Sequence

import torch


def minmax_normalize(
    values: Sequence[float] | torch.Tensor,
    *,
    eps: float = 1e-8,
    constant_value: float = 0.0,
) -> list[float]:
    tensor = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if tensor.numel() == 0:
        return []

    minimum = tensor.min()
    maximum = tensor.max()

    value_range = maximum - minimum

    if value_range.abs().item() < eps:
        return [
            constant_value
            for _ in range(tensor.numel())
        ]

    normalized = (
        tensor - minimum
    ) / (
        value_range + eps
    )

    return normalized.tolist()


def zscore_normalize(
    values: Sequence[float] | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> list[float]:
    tensor = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if tensor.numel() == 0:
        return []

    mean = tensor.mean()
    std = tensor.std(unbiased=False)

    if std.item() < eps:
        return [
            0.0
            for _ in range(tensor.numel())
        ]

    return (
        (tensor - mean) / (std + eps)
    ).tolist()


def softmax_normalize(
    values: Sequence[float] | torch.Tensor,
    *,
    temperature: float = 1.0,
) -> list[float]:
    if temperature <= 0:
        raise ValueError(
            "temperature must be positive."
        )

    tensor = torch.as_tensor(
        values,
        dtype=torch.float32,
    )

    if tensor.numel() == 0:
        return []

    probabilities = torch.softmax(
        tensor / temperature,
        dim=0,
    )

    return probabilities.tolist()