from __future__ import annotations
from typing import Any
from .common import variance

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    total=0; variances=[]; preference_changes=0
    paraphrase_suffixes = ["", " In short.", " Stated another way.", " Put simply.", " The same point applies."]
    for pair in pairs:
        outputs=[backend.score(pair["prompt"], pair["response_a"] + suffix, pair["response_b"]) for suffix in paraphrase_suffixes]
        total += 1
        variances.append(variance([out.score_a for out in outputs]))
        if len({out.preference for out in outputs}) > 1:
            preference_changes += 1
    return {
        "probe": __name__.split(".")[-1],
        "pairs": total,
        "flip_rate": preference_changes / total if total else 0,
        "score_variance": sum(variances) / len(variances) if variances else 0,
        "paraphrases_per_pair": len(paraphrase_suffixes),
        "synthetic": True,
    }
