from __future__ import annotations
from typing import Any

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    flips=0; total=0; first=0
    for pair in pairs:
        out1=backend.score(pair["prompt"], pair["response_a"], pair["response_b"])
        out2=backend.score(pair["prompt"], pair["response_b"], pair["response_a"])
        total += 1
        if out1.preference == out2.preference: flips += 1
        if out1.preference == "A": first += 1
    return {"probe": __name__.split(".")[-1], "pairs": total, "flip_rate": flips / total if total else 0, "first_position_rate": first / total if total else 0, "synthetic": True}
