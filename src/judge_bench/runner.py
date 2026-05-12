from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backends.base import JudgeOutput
from .plots import write_plot_artifacts
from .probes import PROBES
from .synthetic_responses import controlled_pairs

def backend_for(name: str, model: str):
    mod = importlib.import_module(f"judge_bench.backends.{name}")
    return mod.BackendClient(model)

def estimate_cost(backend_name: str, probes: list[str], pairs: int, model: str | None = None) -> float:
    backend = backend_for(backend_name, model or backend_name)
    return backend.cost_per_call * len(probes) * pairs * 2


def cache_key(backend, prompt, a, b):
    return hashlib.sha256(f"{backend.model_family}\0{backend.model}\0{prompt}\0{a}\0{b}".encode()).hexdigest()


class CachedBackend:
    def __init__(self, backend: Any, cache_dir: str | Path):
        self.backend = backend
        self.model = backend.model
        self.model_family = backend.model_family
        self.cost_per_call = backend.cost_per_call
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def score(self, prompt: str, response_a: str, response_b: str) -> JudgeOutput:
        path = self.cache_dir / f"{cache_key(self.backend, prompt, response_a, response_b)}.json"
        if path.exists():
            self.hits += 1
            return JudgeOutput(**json.loads(path.read_text(encoding="utf-8")))
        self.misses += 1
        output = self.backend.score(prompt, response_a, response_b)
        path.write_text(json.dumps(asdict(output), sort_keys=True), encoding="utf-8")
        return output

def run_suite(backend_name="local", model="local-judge", probes=None, pairs=20, cache_dir=".judge-bench-cache"):
    probes = PROBES if probes in (None, ["all"], "all") else probes
    backend = CachedBackend(backend_for(backend_name, model), cache_dir)
    data = controlled_pairs(pairs)
    results = []
    for probe in probes:
        mod = importlib.import_module(f"judge_bench.probes.{probe}")
        results.append(mod.run(backend, data))
    return {
        "backend": backend_name,
        "model": model,
        "synthetic": True,
        "not_a_benchmark": True,
        "response_pairs": len(data),
        "cache": {"hits": backend.hits, "misses": backend.misses, "dir": str(cache_dir)},
        "results": results,
    }


def render_markdown(report):
    lines = [
        "# Judge Bench Diagnostic Report",
        "",
        "This is not a benchmark or leaderboard. All response pairs are synthetic diagnostics.",
        "",
    ]
    for r in report["results"]:
        lines.append(f"- {r['probe']}: flip_rate={r['flip_rate']:.3f}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="judge-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--backend", default="local")
    p.add_argument("--model", default="local-judge")
    p.add_argument("--probes", default="all")
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--cache-dir", default=".judge-bench-cache")
    p.add_argument("--output", default="report.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-cost", action="store_true")
    args = parser.parse_args(argv)
    probes = PROBES if args.probes == "all" else args.probes.split(",")
    if args.pairs <= 0:
        parser.error("--pairs must be a positive integer")
    cost = estimate_cost(args.backend, probes, args.pairs, args.model)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "model": args.model,
                    "probes": probes,
                    "response_pairs": args.pairs,
                    "expected_calls": len(probes) * args.pairs * 2,
                    "expected_cost_usd": round(cost, 2),
                    "cache_dir": args.cache_dir,
                    "requires_confirm_cost": cost > 0,
                },
                indent=2,
            )
        )
        return 0
    if cost > 0 and not args.confirm_cost:
        print(json.dumps({"error": "cost confirmation required", "expected_cost_usd": round(cost, 2)}))
        return 2
    report = run_suite(args.backend, args.model, probes, args.pairs, args.cache_dir)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    report["plots"] = write_plot_artifacts(report, args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
