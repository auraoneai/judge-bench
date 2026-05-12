from __future__ import annotations

import os
from typing import Any

from .base import JudgeOutput
from .http_json import judge_prompt, parse_judge_text, post_json


DEFAULT_ENDPOINTS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
    "hf": "http://localhost:8080",
    "transformers": "",
}


class BackendClient:
    cost_per_call = 0.0
    model_family = "local"

    def __init__(
        self,
        model: str = "local-judge",
        backend: str | None = None,
        endpoint: str | None = None,
        timeout: float = 60.0,
    ):
        inferred_backend, inferred_model = _split_model(model)
        self.backend = (backend or os.environ.get("JUDGE_BENCH_LOCAL_BACKEND") or inferred_backend or "heuristic").lower()
        self.model = inferred_model
        self.endpoint = (endpoint or os.environ.get("JUDGE_BENCH_LOCAL_URL") or DEFAULT_ENDPOINTS.get(self.backend, "")).rstrip("/")
        self.timeout = timeout
        self.model_family = f"local:{self.backend}"

    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        if self.backend == "ollama":
            return self._score_ollama(prompt, response_a, response_b)
        if self.backend == "vllm":
            return self._score_openai_compatible(prompt, response_a, response_b)
        if self.backend in {"hf", "transformers"}:
            return self._score_hf(prompt, response_a, response_b)
        if self.backend != "heuristic":
            raise RuntimeError(f"Unsupported local backend: {self.backend}")
        return self._score_heuristic(response_a, response_b)

    def _score_ollama(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        payload = post_json(
            f"{self.endpoint}/api/generate",
            {},
            {
                "model": self.model,
                "prompt": judge_prompt(prompt, response_a, response_b),
                "format": "json",
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        return parse_judge_text(str(payload.get("response", "")), self.model_family)

    def _score_openai_compatible(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        payload = post_json(
            f"{self.endpoint}/chat/completions",
            _auth_headers(),
            {
                "model": self.model,
                "messages": [{"role": "user", "content": judge_prompt(prompt, response_a, response_b)}],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout,
        )
        text = _extract_chat_completion_text(payload)
        return parse_judge_text(text, self.model_family)

    def _score_hf(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        if self.endpoint:
            payload = post_json(
                f"{self.endpoint}/generate",
                _auth_headers(),
                {
                    "inputs": judge_prompt(prompt, response_a, response_b),
                    "parameters": {"max_new_tokens": 300, "return_full_text": False, "temperature": 0.0},
                },
                timeout=self.timeout,
            )
            return parse_judge_text(_extract_hf_text(payload), self.model_family)
        return self._score_transformers(prompt, response_a, response_b)

    def _score_transformers(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "JUDGE_BENCH_LOCAL_BACKEND=transformers requires the optional transformers package "
                "or JUDGE_BENCH_LOCAL_URL pointing at a local text-generation-inference server"
            ) from exc
        generator = pipeline("text-generation", model=self.model)
        output = generator(judge_prompt(prompt, response_a, response_b), max_new_tokens=300, do_sample=False)
        return parse_judge_text(_extract_hf_text(output), self.model_family)

    def _score_heuristic(self, response_a: str, response_b: str) -> JudgeOutput:
        score_a = min(1.0, len(response_a.split()) / max(len(response_b.split()), 1))
        score_b = min(1.0, len(response_b.split()) / max(len(response_a.split()), 1))
        pref = "A" if score_a >= score_b else "B"
        return JudgeOutput(pref, score_a, score_b, "deterministic offline lexical heuristic", self.model_family)


def _split_model(model: str) -> tuple[str | None, str]:
    if ":" not in model:
        return None, model
    backend, local_model = model.split(":", 1)
    if backend.lower() in {"ollama", "vllm", "hf", "transformers", "heuristic"} and local_model:
        return backend, local_model
    return None, model


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("JUDGE_BENCH_LOCAL_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _extract_chat_completion_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        if "content" in message:
            return str(message["content"])
    raise RuntimeError("Local OpenAI-compatible backend did not include message content")


def _extract_hf_text(payload: Any) -> str:
    if isinstance(payload, list) and payload:
        return _extract_hf_text(payload[0])
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "response"):
            if key in payload:
                return str(payload[key])
    raise RuntimeError("Local Hugging Face backend did not include generated text")
