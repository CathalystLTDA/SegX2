"""Dynamic segregation (Dominant Trajectories) — hand-computed ground truth.

Dominance weight = ``1 − normalized_entropy`` (whole-distribution homogeneity).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from segx.config import DemographicAttribute, DemographicSchema, Settings
from segx.metrics.entropy import normalized_entropy
from segx.metrics.registry import get_metric


def _settings() -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        )
    )


def _edge_table(segments: dict[tuple[int, int, int], dict[str, float]]) -> pd.DataFrame:
    """One partition (walk/park/nearest), one attribute (income low/high)."""
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


def test_dominant_class_and_weight():
    table = _edge_table({(0, 1, 0): {"low": 3.0, "high": 1.0}, (1, 2, 0): {"low": 1.0, "high": 1.0}})
    out = get_metric("dominant_trajectories")().compute(table, _settings())
    by_seg = {(r.u, r.v, r.key): (r.dominant_class, r.dominance) for _, r in out.iterrows()}

    # seg (0,1,0): low dominates 3 vs 1; dominance = 1 − h([3,1])
    dom_a, w_a = by_seg[(0, 1, 0)]
    assert dom_a == "low"
    assert w_a == pytest.approx(1.0 - normalized_entropy([3.0, 1.0], n_classes=2))
    # seg (1,2,0): tie 1/1 → first class wins, dominance 0 (perfectly mixed)
    dom_b, w_b = by_seg[(1, 2, 0)]
    assert dom_b == "low"
    assert w_b == pytest.approx(0.0)


def test_no_mass_segment_has_no_winner():
    table = _edge_table({(0, 1, 0): {"low": 0.0, "high": 0.0}})
    out = get_metric("dominant_trajectories")().compute(table, _settings())
    row = out.iloc[0]
    assert row.dominant_class is None
    assert math.isnan(row.dominance)


def test_summary_is_mean_dominance():
    table = _edge_table({(0, 1, 0): {"low": 3.0, "high": 1.0}, (1, 2, 0): {"low": 1.0, "high": 1.0}})
    metric = get_metric("dominant_trajectories")()
    summary = metric.summarize(metric.compute(table, _settings()), _settings())
    expected = (1.0 - normalized_entropy([3.0, 1.0], n_classes=2) + 0.0) / 2
    assert summary.iloc[0]["dominance"] == pytest.approx(expected)


def test_empty_table():
    metric = get_metric("dominant_trajectories")()
    out = metric.compute(pd.DataFrame(), _settings())
    assert out.empty
    assert "dominant_class" in out.columns and "dominance" in out.columns
