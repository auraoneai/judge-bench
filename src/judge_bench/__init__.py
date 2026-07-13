__all__ = ["run_suite", "estimate_calls", "estimate_cost"]

def __getattr__(name):
    if name in __all__:
        from .runner import estimate_calls, estimate_cost, run_suite
        return {"run_suite": run_suite, "estimate_calls": estimate_calls, "estimate_cost": estimate_cost}[name]
    raise AttributeError(name)
