from __future__ import annotations
from typing import Any

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    first_preferred=0; total=0
    for pair in pairs:
        out1=backend.score(pair["prompt"], pair["response_a"], pair["response_b"])
        out2=backend.score(pair["prompt"], pair["response_b"], pair["response_a"])
        total += 2
        if out1.preference == "A": first_preferred += 1
        if out2.preference == "A": first_preferred += 1
    first_rate = first_preferred / total if total else 0
    return {
        "probe": __name__.split(".")[-1],
        "pairs": len(pairs),
        "trials": total,
        "flip_rate": abs(first_rate - 0.5) * 2,
        "first_position_rate": first_rate,
        "synthetic": True,
    }
