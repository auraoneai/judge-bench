from __future__ import annotations

from statistics import mean
from typing import Any


def semantic_choice(output, normal_a_id: str, normal_b_id: str) -> str:
    if output.preference == "A":
        return normal_a_id
    if output.preference == "B":
        return normal_b_id
    return "tie"


def variance(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def own_family(backend) -> str:
    return str(getattr(backend, "model_family", "unknown")).split(":", 1)[0]


def quality_label(pair: dict[str, Any]) -> str:
    return str(pair.get("quality_label", "A"))
