from judge_bench.backends.base import JudgeOutput
from judge_bench.runner import CachedBackend, run_suite, estimate_cost

def test_dry_cost(): assert estimate_cost("local", ["position_bias"], 10) == 0

def test_all_probes_mocked():
    report = run_suite("local", probes=["position_bias", "verbosity_bias", "self_preference", "paraphrase_stability", "anchoring", "calibration"], pairs=3)
    assert len(report["results"]) == 6 and report["not_a_benchmark"] is True

def test_cache_prevents_duplicate_backend_calls(tmp_path):
    class CountingBackend:
        model = "counting"
        model_family = "local"
        cost_per_call = 0.0
        def __init__(self): self.calls = 0
        def score(self, prompt, response_a, response_b):
            self.calls += 1
            return JudgeOutput("A", 1.0, 0.0, "cached", "local")
    raw = CountingBackend()
    backend = CachedBackend(raw, tmp_path)
    assert backend.score("p", "a", "b") == backend.score("p", "a", "b")
    assert raw.calls == 1
    assert backend.hits == 1 and backend.misses == 1
