"""Most exclusive segments (Segregated Trajectories) — per-segment location
quotient of the dominant group + exclusivity flag. Hand-computed ground truth."""

from __future__ import annotations

import pandas as pd
import pytest

from segx.config import DemographicAttribute, DemographicSchema, MetricsConfig, Settings
from segx.metrics.registry import get_metric


def _settings(threshold: float = 2.0) -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        ),
        metrics=MetricsConfig(exclusive_lq_threshold=threshold),
    )


def _edge_table(segments: dict[tuple[int, int, int], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (u, v, key), classes in segments.items():
        for cls in ("low", "high"):
            rows.append(
                {
                    "mode": "walk",
                    "category": "park",
                    "binding": "nearest",
                    "u": u,
                    "v": v,
                    "key": key,
                    "attribute": "income",
                    "class": cls,
                    "mass": classes[cls],
                    "n_paths": 1,
                }
            )
    return pd.DataFrame(rows)


# high is globally rare (2 of 22) but locally the whole of segment C → strongly exclusive.
_SCENARIO = {
    (0, 1, 0): {"low": 10.0, "high": 0.0},
    (1, 2, 0): {"low": 10.0, "high": 0.0},
    (2, 3, 0): {"low": 0.0, "high": 2.0},
}


def test_location_quotient_and_exclusive_flag():
    out = get_metric("segregated_trajectories")().compute(_edge_table(_SCENARIO), _settings())
    by_seg = {(r.u, r.v, r.key): (r.dominant_class, r.lq, r.exclusive) for _, r in out.iterrows()}

    # low dominant on A/B, only mildly over-represented (global share already 20/22)
    dom_a, lq_a, exc_a = by_seg[(0, 1, 0)]
    assert dom_a == "low"
    assert lq_a == pytest.approx(1.0 / (20.0 / 22.0))  # = 1.1
    assert exc_a is False
    # high dominant on C, globally rare (2/22) → LQ = 1 / (2/22) = 11 → exclusive
    dom_c, lq_c, exc_c = by_seg[(2, 3, 0)]
    assert dom_c == "high"
    assert lq_c == pytest.approx(1.0 / (2.0 / 22.0))  # = 11.0
    assert exc_c is True


def test_threshold_is_configurable():
    # with a very high threshold even the 11× segment is not "exclusive"
    out = get_metric("segregated_trajectories")().compute(_edge_table(_SCENARIO), _settings(threshold=20.0))
    assert not out["exclusive"].any()


def test_summary_counts_exclusive_per_group():
    metric = get_metric("segregated_trajectories")()
    settings = _settings()
    summary = metric.summarize(metric.compute(_edge_table(_SCENARIO), settings), settings)
    by_group = {r.dominant_class: (r.n_segments, r.n_exclusive, r.share_exclusive) for _, r in summary.iterrows()}
    assert by_group["low"] == (2, 0, pytest.approx(0.0))
    assert by_group["high"] == (1, 1, pytest.approx(1.0))


def test_empty_table():
    metric = get_metric("segregated_trajectories")()
    out = metric.compute(pd.DataFrame(), _settings())
    assert out.empty
    assert {"dominant_class", "lq", "exclusive"} <= set(out.columns)
