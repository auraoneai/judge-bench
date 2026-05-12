from __future__ import annotations

from .base import JudgeOutput
from .http_json import judge_prompt, parse_judge_text, post_json, require_env


class BackendClient:
    cost_per_call = 0.008
    model_family = "google"

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model

    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        payload = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": require_env("GEMINI_API_KEY")},
            {
                "contents": [{"parts": [{"text": judge_prompt(prompt, response_a, response_b)}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "preference": {"type": "STRING", "enum": ["A", "B", "tie"]},
                            "score_a": {"type": "NUMBER"},
                            "score_b": {"type": "NUMBER"},
                            "rationale": {"type": "STRING"},
                        },
                        "required": ["preference", "score_a", "score_b", "rationale"],
                    },
                },
            },
        )
        return parse_judge_text(_extract_candidate_text(payload), self.model_family)


def _extract_candidate_text(payload: dict) -> str:
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                return str(part["text"])
    raise RuntimeError("Gemini response did not include candidate text")
