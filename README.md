# judge-bench

Probe an LLM judge for position, verbosity, self-preference, paraphrase, anchoring, and calibration sensitivity.

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
