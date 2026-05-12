from __future__ import annotations
from .base import JudgeOutput
class BackendClient:
    cost_per_call = 0.012
    model_family = "anthropic"
    def __init__(self, model: str = "claude-4.7"):
        self.model = model
    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        score_a = min(1.0, len(response_a.split()) / max(len(response_b.split()), 1))
        score_b = min(1.0, len(response_b.split()) / max(len(response_a.split()), 1))
        pref = "A" if score_a >= score_b else "B"
        return JudgeOutput(pref, score_a, score_b, "deterministic offline placeholder", self.model_family)
