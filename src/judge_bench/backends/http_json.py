from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .base import JudgeOutput

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "preference": {"type": "string", "enum": ["A", "B", "tie"]},
        "score_a": {"type": "number", "minimum": 0, "maximum": 1},
        "score_b": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["preference", "score_a", "score_b", "rationale"],
    "additionalProperties": False,
}


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for this backend")
    return value


def judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return "\n".join(
        [
            "You are evaluating two candidate responses for the supplied prompt.",
            "Return only JSON with keys: preference, score_a, score_b, rationale.",
            "Use preference A, B, or tie. Scores must be numbers from 0 to 1.",
            "",
            f"Prompt:\n{prompt}",
            "",
            f"Response A:\n{response_a}",
            "",
            f"Response B:\n{response_b}",
        ]
    )


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend request failed with HTTP {exc.code}: {body}") from exc


def parse_judge_text(text: str, model_family: str) -> JudgeOutput:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Backend did not return JSON: {text[:200]}")
        payload = json.loads(match.group(0))

    preference = str(payload.get("preference", "")).strip()
    if preference not in {"A", "B", "tie"}:
        raise RuntimeError(f"Backend returned invalid preference: {preference!r}")
    return JudgeOutput(
        preference=preference,
        score_a=float(payload["score_a"]),
        score_b=float(payload["score_b"]),
        rationale=str(payload.get("rationale", "")),
        model_family=model_family,
    )

