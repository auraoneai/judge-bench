# Changelog

## 0.1.2 - 2026-07-13

- Publish the current canonical source with refreshed discovery metadata, runtime and data boundaries, release links, and package-specific CLI or report improvements.

## 0.1.1

- Prepare hardened source-side release after CI, validation, documentation, and packaging fixes.
- Add real local backend paths for Ollama, vLLM/OpenAI-compatible servers, and Hugging Face text generation while retaining a deterministic offline heuristic for smoke tests.
- Split the six probe modules into distinct synthetic diagnostics and emit JSON/SVG/optional matplotlib plot artifacts from the runner.
- Add explicit `--pairs` and `--cache-dir` runner controls, exact per-probe dry-run call estimates, and backend-family cache keys to avoid cross-provider collisions.

## 0.1.0

- Initial open-source implementation.
