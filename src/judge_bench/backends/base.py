from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class JudgeOutput:
    preference: str
    score_a: float
    score_b: float
    rationale: str = ""
    model_family: str | None = None

class Backend(Protocol):
    model: str
    model_family: str
    cost_per_call: float
    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput: ...
