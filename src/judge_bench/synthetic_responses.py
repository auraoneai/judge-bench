from __future__ import annotations

def controlled_pairs(n: int = 500):
    pairs=[]
    for i in range(n):
        prompt=f"Synthetic prompt {i}: explain policy {i % 7}."
        good=f"Complete answer with step one, step two, and a concise justification {i}."
        weak=f"Partial answer {i}."
        pairs.append({"id": f"synthetic-{i}", "prompt": prompt, "response_a": good, "response_b": weak, "quality_label": "A", "synthetic": True})
    return pairs
