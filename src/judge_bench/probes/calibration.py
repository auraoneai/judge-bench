from __future__ import annotations
from typing import Any
from .common import quality_label

def run(backend, pairs: list[dict[str, Any]], trials: int = 1) -> dict[str, Any]:
    bins = [{"count": 0, "confidence": 0.0, "accuracy": 0.0} for _ in range(5)]
    correct=0; total=0
    for pair in pairs:
        out1=backend.score(pair["prompt"], pair["response_a"], pair["response_b"])
        confidence = max(out1.score_a, out1.score_b)
        predicted = out1.preference
        is_correct = predicted == quality_label(pair)
        total += 1
        correct += 1 if is_correct else 0
        idx = min(4, max(0, int(confidence * 5)))
        bins[idx]["count"] += 1
        bins[idx]["confidence"] += confidence
        bins[idx]["accuracy"] += 1.0 if is_correct else 0.0
    ece=0.0; reliability=[]
    for item in bins:
        if not item["count"]:
            reliability.append({"count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        confidence = item["confidence"] / item["count"]
        accuracy = item["accuracy"] / item["count"]
        ece += (item["count"] / total) * abs(accuracy - confidence) if total else 0
        reliability.append({"count": item["count"], "confidence": round(confidence, 6), "accuracy": round(accuracy, 6)})
    return {
        "probe": __name__.split(".")[-1],
        "pairs": total,
        "flip_rate": ece,
        "ece": ece,
        "accuracy": correct / total if total else 0,
        "reliability_bins": reliability,
        "synthetic": True,
    }
