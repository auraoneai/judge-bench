from .base import Backend, JudgeOutput
BACKENDS = {"openai": "judge_bench.backends.openai", "anthropic": "judge_bench.backends.anthropic", "google": "judge_bench.backends.google", "local": "judge_bench.backends.local"}
