# judge-bench

**LLM-as-a-judge bias and calibration testing.** Probe an LLM judge for position,
verbosity, self-preference, paraphrase, anchoring, and calibration sensitivity.

[![PyPI](https://img.shields.io/pypi/v/judge-bench)](https://pypi.org/project/judge-bench/)
[![CI](https://github.com/auraoneai/judge-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/auraoneai/judge-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/auraoneai/judge-bench/blob/main/LICENSE)

```bash
python -m pip install judge-bench==0.1.2
judge-bench run --backend local --model local-judge --probes position_bias,calibration --pairs 3 --output judge-report.json
```

Excerpt of `judge-report.json` (exit code `0`). The `local-judge` model is a
deterministic offline heuristic run on synthetic pairs; these numbers check the
pipeline and are not a benchmark:

```json
{
  "backend": "local",
  "model": "local-judge",
  "synthetic": true,
  "not_a_benchmark": true,
  "response_pairs": 3,
  "results": [
    {"probe": "position_bias", "pairs": 3, "flip_rate": 0.0, "first_position_rate": 1.0, "synthetic": true},
    {"probe": "calibration", "pairs": 3, "flip_rate": 0.0, "ece": 0.0, "accuracy": 1.0, "synthetic": true}
  ]
}
```

## What It Does and Doesn't Do

It does:

- run controlled response pairs through a chosen judge backend (hosted,
  local HTTP, local Python, or the offline `local-judge`);
- record probe-level evidence, cache repeated calls, and write JSON, Markdown,
  and SVG plot artifacts.

It doesn't:

- rank models or act as a leaderboard. All bundled response pairs are
  synthetic, and results describe only the selected prompts, backend, model,
  and probes;
- establish general model quality or production safety.

Part of the AuraOne review quality workflow:
[iaa-kit](https://auraone.ai/open/iaa-kit) →
[judge-bench](https://auraone.ai/open/judge-bench).

## Overview

`judge-bench` is for evaluation engineers choosing or monitoring an LLM-as-judge configuration. Unlike a model leaderboard, it runs controlled synthetic response pairs through a selected backend, records probe-level evidence, caches repeated calls, and produces reviewable report and plot artifacts.

## Inspectable Output

A run writes:

- `<name>.json`: backend, model, synthetic disclosure, cache hits/misses, and probe results.
- `<name>.md`: a concise diagnostic report.
- `<name>.plots.json`: plot-ready reliability and bias points.
- `<name>.svg`: deterministic plots.
- `<name>.png`: only when `matplotlib` is installed.

The built-in dry-run cost estimate uses static per-call constants for planning. It does not query provider pricing and should not be treated as a quote.

## Runtime Boundary

The default `local-judge` model is a deterministic lexical heuristic and makes no network request. `ollama:`, `vllm:`, and hosted Hugging Face modes call the configured local HTTP endpoint; `transformers:` loads a local Python model pipeline. The `openai`, `anthropic`, and `google` backends send the synthetic prompt/response pairs to those providers and require their standard API-key environment variables. Paid-backend runs require `--confirm-cost`.

Cache entries contain model inputs and outputs as local JSON files. Choose `--cache-dir` accordingly when prompts or rationales are sensitive.

## Install

```bash
python -m pip install judge-bench==0.1.2
```

For development from a clone:

```bash
python -m pip install -e .
```

## Quickstart

Run a fully offline diagnostic:

```bash
judge-bench run \
  --backend local \
  --model local-judge \
  --probes position_bias,calibration \
  --pairs 3 \
  --cache-dir .judge-bench-cache \
  --output judge-report.json
```

## Backends

- Hosted: `openai`, `anthropic`, `google`.
- Local HTTP: `ollama:<model>`, `vllm:<model>`, `hf:<model>`.
- Local Python: `transformers:<model>` with `transformers` installed.
- Offline smoke test: `local-judge`.

See [`docs/what-each-probe-measures.md`](docs/what-each-probe-measures.md), [`docs/interpreting-position-bias.md`](docs/interpreting-position-bias.md), and [`docs/methodology.md`](docs/methodology.md).

## Release Status

Registry status verified July 13, 2026: version `0.1.2` is published on PyPI and tagged `v0.1.2` in the public repository. The project is alpha software. No model-quality, superiority, or adoption claim is made.

## Limits

All bundled response pairs are synthetic. Results characterize the selected prompts, backend, model, and probe set; they do not establish general model quality or production safety.

## Next Action

Run the offline `local-judge` quickstart, inspect the probe-level Markdown and SVG evidence, then repeat with the intended production judge only after confirming the cache, data-sharing, network, and cost boundary.

---

Need an independent model-quality evaluation or benchmark program?
[Talk to AuraOne](https://auraone.ai/contact?from=open:judge-bench)
