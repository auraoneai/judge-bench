from __future__ import annotations

import json
from pathlib import Path


def reliability_diagram_points(results):
    calibration = next((result for result in results if result.get("probe") == "calibration"), None)
    if calibration and calibration.get("reliability_bins"):
        return [(item["confidence"], item["accuracy"]) for item in calibration["reliability_bins"] if item["count"]]
    return [(i / max(len(results), 1), result.get("flip_rate", 0.0)) for i, result in enumerate(results, start=1)]


def bias_points(results):
    return [(result.get("probe", "unknown"), float(result.get("flip_rate", 0.0))) for result in results]


def write_plot_artifacts(report: dict, output: str | Path) -> dict[str, str | None]:
    output = Path(output)
    stem = output.with_suffix("")
    points = {"reliability": reliability_diagram_points(report.get("results", [])), "bias": bias_points(report.get("results", []))}
    json_path = stem.with_suffix(".plots.json")
    svg_path = stem.with_suffix(".svg")
    json_path.write_text(json.dumps(points, indent=2), encoding="utf-8")
    svg_path.write_text(_svg(points["bias"]), encoding="utf-8")
    png_path = _try_matplotlib(points, stem.with_suffix(".png"))
    return {"json": str(json_path), "svg": str(svg_path), "png": str(png_path) if png_path else None}


def _try_matplotlib(points: dict, path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    labels = [label for label, _value in points["bias"]]
    values = [value for _label, value in points["bias"]]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(labels, values)
    ax.set_ylabel("diagnostic rate")
    ax.set_ylim(0, 1)
    ax.set_title("Judge Bench synthetic diagnostics")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _svg(points: list[tuple[str, float]]) -> str:
    width = max(320, 90 * len(points))
    bars = []
    for index, (label, value) in enumerate(points):
        height = int(max(0.0, min(1.0, value)) * 140)
        x = 30 + index * 85
        y = 170 - height
        bars.append(f'<rect x="{x}" y="{y}" width="48" height="{height}" fill="#2563eb" />')
        bars.append(f'<text x="{x}" y="190" font-size="10">{label}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="210" role="img">'
        '<text x="20" y="20" font-size="14">Judge Bench synthetic diagnostics</text>'
        + "".join(bars)
        + "</svg>"
    )
