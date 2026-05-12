from __future__ import annotations

from .base import JudgeOutput
from .http_json import JUDGE_SCHEMA, judge_prompt, parse_judge_text, post_json, require_env


class BackendClient:
    cost_per_call = 0.01
    model_family = "openai"

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        payload = post_json(
            "https://api.openai.com/v1/responses",
            {"Authorization": f"Bearer {require_env('OPENAI_API_KEY')}"},
            {
                "model": self.model,
                "input": judge_prompt(prompt, response_a, response_b),
                "max_output_tokens": 300,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "judge_bench_score",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            },
        )
        text = payload.get("output_text") or _extract_response_text(payload)
        return parse_judge_text(text, self.model_family)


def _extract_response_text(payload: dict) -> str:
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                return str(content["text"])
    raise RuntimeError("OpenAI response did not include output text")
