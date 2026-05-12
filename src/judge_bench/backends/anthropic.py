from __future__ import annotations

from .base import JudgeOutput
from .http_json import judge_prompt, parse_judge_text, post_json, require_env


class BackendClient:
    cost_per_call = 0.012
    model_family = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        payload = post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": require_env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": judge_prompt(prompt, response_a, response_b)}],
            },
        )
        return parse_judge_text(_extract_message_text(payload), self.model_family)


def _extract_message_text(payload: dict) -> str:
    for content in payload.get("content", []):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
    raise RuntimeError("Anthropic response did not include text content")
