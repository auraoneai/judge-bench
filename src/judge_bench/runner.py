from __future__ import annotations
import argparse, hashlib, importlib, json
from pathlib import Path
from .synthetic_responses import controlled_pairs
from .probes import PROBES

def backend_for(name: str, model: str):
    mod = importlib.import_module(f"judge_bench.backends.{name}")
    return mod.BackendClient(model)

def estimate_cost(backend_name: str, probes: list[str], pairs: int) -> float:
    backend = backend_for(backend_name, backend_name)
    return backend.cost_per_call * len(probes) * pairs * 2

def cache_key(backend, prompt, a, b):
    return hashlib.sha256(f"{backend.model}\0{prompt}\0{a}\0{b}".encode()).hexdigest()

def run_suite(backend_name="local", model="local-judge", probes=None, pairs=20, cache_dir=".judge-bench-cache"):
    probes = PROBES if probes in (None, ["all"], "all") else probes
    backend = backend_for(backend_name, model); data = controlled_pairs(pairs); results=[]
    Path(cache_dir).mkdir(exist_ok=True)
    for probe in probes:
        mod = importlib.import_module(f"judge_bench.probes.{probe}")
        results.append(mod.run(backend, data))
    return {"backend": backend_name, "model": model, "synthetic": True, "not_a_benchmark": True, "results": results}

def render_markdown(report):
    lines=["# Judge Bench Diagnostic Report", "", "This is not a benchmark or leaderboard. All response pairs are synthetic diagnostics.", ""]
    for r in report["results"]: lines.append(f"- {r['probe']}: flip_rate={r['flip_rate']:.3f}")
    return "\n".join(lines)+"\n"

def main(argv=None):
    parser=argparse.ArgumentParser(prog="judge-bench"); sub=parser.add_subparsers(dest="cmd", required=True)
    p=sub.add_parser("run"); p.add_argument("--backend", default="local"); p.add_argument("--model", default="local-judge"); p.add_argument("--probes", default="all"); p.add_argument("--output", default="report.json"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--confirm-cost", action="store_true")
    args=parser.parse_args(argv)
    probes=PROBES if args.probes == "all" else args.probes.split(',')
    cost=estimate_cost(args.backend, probes, 20)
    if args.dry_run:
        print(json.dumps({"expected_cost_usd": round(cost,2), "requires_confirm_cost": True}, indent=2)); return 0
    if cost > 0 and not args.confirm_cost:
        print(json.dumps({"error": "cost confirmation required", "expected_cost_usd": round(cost,2)})); return 2
    report=run_suite(args.backend,args.model,probes); Path(args.output).write_text(json.dumps(report, indent=2)); Path(args.output).with_suffix('.md').write_text(render_markdown(report)); print(json.dumps(report, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
