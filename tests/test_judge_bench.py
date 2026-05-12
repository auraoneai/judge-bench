from judge_bench.runner import run_suite, estimate_cost

def test_dry_cost(): assert estimate_cost("local", ["position_bias"], 10) == 0

def test_all_probes_mocked():
    report = run_suite("local", probes=["position_bias", "verbosity_bias", "self_preference", "paraphrase_stability", "anchoring", "calibration"], pairs=3)
    assert len(report["results"]) == 6 and report["not_a_benchmark"] is True
