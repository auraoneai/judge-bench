from __future__ import annotations

def controlled_pairs(n: int = 500):
    families = ["openai", "anthropic", "google", "local"]
    pairs=[]
    for i in range(n):
        prompt=f"Synthetic prompt {i}: explain policy {i % 7}."
        good=f"Complete answer with step one, step two, and a concise justification {i}."
        weak=f"Partial answer {i}."
        pairs.append({
            "id": f"synthetic-{i}",
            "prompt": prompt,
            "response_a": good,
            "response_b": weak,
            "quality_label": "A",
            "quality_score_a": 0.9,
            "quality_score_b": 0.25,
            "model_family_a": families[i % len(families)],
            "model_family_b": families[(i + 1) % len(families)],
            "synthetic": True,
        })
    return pairs
