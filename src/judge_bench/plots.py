def reliability_diagram_points(results):
    return [(i / max(len(results), 1), r.get("flip_rate", 0.0)) for i, r in enumerate(results, start=1)]
