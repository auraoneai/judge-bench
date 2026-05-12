import judge_bench.backends.anthropic as anthropic_backend
import judge_bench.backends.google as google_backend
import judge_bench.backends.local as local_backend
import judge_bench.backends.openai as openai_backend
from judge_bench.backends.base import JudgeOutput
from judge_bench.runner import CachedBackend, estimate_cost, run_suite

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

def test_openai_backend_parses_structured_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        openai_backend,
        "post_json",
        lambda *args, **kwargs: {"output_text": '{"preference":"A","score_a":0.9,"score_b":0.2,"rationale":"clearer"}'},
    )
    output = openai_backend.BackendClient("gpt-4o").score("prompt", "good", "bad")
    assert output == JudgeOutput("A", 0.9, 0.2, "clearer", "openai")

def test_anthropic_backend_parses_message_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        anthropic_backend,
        "post_json",
        lambda *args, **kwargs: {
            "content": [{"type": "text", "text": '{"preference":"B","score_a":0.1,"score_b":0.8,"rationale":"more complete"}'}]
        },
    )
    output = anthropic_backend.BackendClient("claude-sonnet-4-20250514").score("prompt", "bad", "good")
    assert output == JudgeOutput("B", 0.1, 0.8, "more complete", "anthropic")

def test_google_backend_parses_generate_content_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        google_backend,
        "post_json",
        lambda *args, **kwargs: {
            "candidates": [{"content": {"parts": [{"text": '{"preference":"tie","score_a":0.5,"score_b":0.5,"rationale":"equivalent"}'}]}}]
        },
    )
    output = google_backend.BackendClient("gemini-2.5-flash").score("prompt", "same", "same")
    assert output == JudgeOutput("tie", 0.5, 0.5, "equivalent", "google")

def test_local_ollama_backend_posts_generate_payload(monkeypatch):
    calls = []
    monkeypatch.setattr(
        local_backend,
        "post_json",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or {"response": '{"preference":"A","score_a":0.7,"score_b":0.3,"rationale":"better"}'},
    )
    output = local_backend.BackendClient("ollama:llama3.1", endpoint="http://ollama.test").score("p", "a", "b")
    args, kwargs = calls[0]
    assert args[0] == "http://ollama.test/api/generate"
    assert args[2]["model"] == "llama3.1"
    assert args[2]["format"] == "json"
    assert kwargs["timeout"] == 60.0
    assert output == JudgeOutput("A", 0.7, 0.3, "better", "local:ollama")

def test_local_vllm_backend_posts_openai_compatible_payload(monkeypatch):
    calls = []
    monkeypatch.setenv("JUDGE_BENCH_LOCAL_API_KEY", "local-token")
    monkeypatch.setattr(
        local_backend,
        "post_json",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or {
            "choices": [
                {"message": {"content": '{"preference":"B","score_a":0.2,"score_b":0.9,"rationale":"stronger"}'}}
            ]
        },
    )
    output = local_backend.BackendClient("vllm:meta-llama/Llama-3.1-8B-Instruct", endpoint="http://vllm.test/v1").score(
        "p", "a", "b"
    )
    args, _kwargs = calls[0]
    assert args[0] == "http://vllm.test/v1/chat/completions"
    assert args[1] == {"Authorization": "Bearer local-token"}
    assert args[2]["response_format"] == {"type": "json_object"}
    assert output == JudgeOutput("B", 0.2, 0.9, "stronger", "local:vllm")

def test_local_hf_backend_parses_tgi_response(monkeypatch):
    monkeypatch.setenv("JUDGE_BENCH_LOCAL_URL", "http://tgi.test")
    monkeypatch.setattr(
        local_backend,
        "post_json",
        lambda *args, **kwargs: [{"generated_text": '{"preference":"tie","score_a":0.5,"score_b":0.5,"rationale":"same"}'}],
    )
    output = local_backend.BackendClient("hf:mistral").score("p", "a", "b")
    assert output == JudgeOutput("tie", 0.5, 0.5, "same", "local:hf")
