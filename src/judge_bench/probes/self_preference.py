from __future__ import annotations
from typing import Any
from .common import own_family

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    family = own_family(backend)
    own_seen=0; own_preferred=0; total=0
    for pair in pairs:
        out1=backend.score(pair["prompt"], pair["response_a"], pair["response_b"])
        total += 1
        preferred = pair.get("model_family_a") if out1.preference == "A" else pair.get("model_family_b") if out1.preference == "B" else "tie"
        if family in {pair.get("model_family_a"), pair.get("model_family_b")}:
            own_seen += 1
            if preferred == family:
                own_preferred += 1
    return {
        "probe": __name__.split(".")[-1],
        "pairs": total,
        "own_family": family,
        "own_family_cases": own_seen,
        "flip_rate": own_preferred / own_seen if own_seen else 0,
        "own_family_preference_rate": own_preferred / own_seen if own_seen else 0,
        "synthetic": True,
    }
