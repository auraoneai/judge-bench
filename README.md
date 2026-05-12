# judge-bench

`judge-bench` runs synthetic diagnostics for LLM-as-judge reliability: position bias, verbosity bias, self-preference, paraphrase stability, anchoring, and calibration. It emits JSON, Markdown, and plot-ready summaries.

## Quickstart

```bash
pip install judge-bench
judge-bench run --backend openai --model gpt-4o --probes position_bias --dry-run
judge-bench run --backend local --probes all --output report.json
```

Provider backends call their public APIs directly with standard environment variables:
`OPENAI_API_KEY` for `--backend openai`, `ANTHROPIC_API_KEY` for `--backend anthropic`,
and `GEMINI_API_KEY` for `--backend google`. Non-dry runs require `--confirm-cost`.

Repeated judge calls are cached by `(model, prompt, response_a, response_b)` under `.judge-bench-cache` so paid backends do not re-run the same synthetic diagnostic pair.

The local backend can run against local model servers without API spend:

```bash
judge-bench run --backend local --model ollama:llama3.1 --probes position_bias --output report.json
JUDGE_BENCH_LOCAL_URL=http://localhost:8000/v1 judge-bench run --backend local --model vllm:meta-llama/Llama-3.1-8B-Instruct --probes position_bias --output report.json
JUDGE_BENCH_LOCAL_URL=http://localhost:8080 judge-bench run --backend local --model hf:mistral --probes position_bias --output report.json
```

Supported local modes are `ollama:<model>` for Ollama `/api/generate`, `vllm:<model>` for OpenAI-compatible `/chat/completions`, and `hf:<model>` or `transformers:<model>` for Hugging Face text generation. `JUDGE_BENCH_LOCAL_BACKEND`, `JUDGE_BENCH_LOCAL_URL`, and `JUDGE_BENCH_LOCAL_API_KEY` can override mode, endpoint, and bearer token. If no local mode is selected, `local-judge` uses a deterministic lexical heuristic for offline smoke tests.

## What This Is Not

This is not a benchmark, leaderboard, or claim of model superiority. All bundled pairs are synthetic and disclosed as such.
