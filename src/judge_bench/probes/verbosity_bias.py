from __future__ import annotations
from typing import Any

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    verbose_wins=0; score_lift=0.0; total=0
    for pair in pairs:
        concise = pair["response_b"]
        verbose = concise + " " + " ".join(["additional context"] * 8)
        out=backend.score(pair["prompt"], concise, verbose)
        total += 1
        if out.preference == "B": verbose_wins += 1
        score_lift += out.score_b - out.score_a
    rate = verbose_wins / total if total else 0
    return {
        "probe": __name__.split(".")[-1],
        "pairs": total,
        "flip_rate": rate,
        "verbose_preference_rate": rate,
        "average_score_lift": score_lift / total if total else 0,
        "synthetic": True,
    }
