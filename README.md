# judge-bench

`judge-bench` runs synthetic diagnostics for LLM-as-judge reliability: position bias, verbosity bias, self-preference, paraphrase stability, anchoring, and calibration. It emits JSON, Markdown, and plot-ready summaries.

## Quickstart

```bash
pip install judge-bench
judge-bench run --backend openai --model gpt-4o --probes position_bias --dry-run
judge-bench run --backend local --probes all --output report.json
```

Repeated judge calls are cached by `(model, prompt, response_a, response_b)` under `.judge-bench-cache` so paid backends do not re-run the same synthetic diagnostic pair.

## What This Is Not

This is not a benchmark, leaderboard, or claim of model superiority. All bundled pairs are synthetic and disclosed as such.
