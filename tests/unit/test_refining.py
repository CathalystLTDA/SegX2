"""Experienced segregation (1 − h) and Intergroup contact (Gini–Simpson) as
registered metrics over the P4 edge table. Hand-computed ground truth."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from segx.config import DemographicAttribute, DemographicSchema, Settings
from segx.metrics.entropy import gini_simpson, normalized_entropy
from segx.metrics.registry import get_metric


def _settings() -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        )
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


_TABLE = _edge_table({(0, 1, 0): {"low": 3.0, "high": 1.0}, (1, 2, 0): {"low": 1.0, "high": 1.0}})


def test_experienced_segregation_per_segment_and_summary():
    metric = get_metric("experienced_segregation")()
    settings = _settings()
    out = metric.compute(_TABLE, settings)
    by_seg = {(r.u, r.v, r.key): r.experienced_segregation for _, r in out.iterrows()}
    seg_a = 1.0 - normalized_entropy([3.0, 1.0], n_classes=2)
    assert by_seg[(0, 1, 0)] == pytest.approx(seg_a)
    assert by_seg[(1, 2, 0)] == pytest.approx(0.0)  # perfectly mixed
    summary = metric.summarize(out, settings)
    assert summary.iloc[0]["experienced_segregation"] == pytest.approx((seg_a + 0.0) / 2)


def test_intergroup_contact_per_segment_and_summary():
    metric = get_metric("intergroup_contact")()
    settings = _settings()
    out = metric.compute(_TABLE, settings)
    by_seg = {(r.u, r.v, r.key): r.intergroup_contact for _, r in out.iterrows()}
    assert by_seg[(0, 1, 0)] == pytest.approx(gini_simpson([3.0, 1.0]))  # 0.375
    assert by_seg[(1, 2, 0)] == pytest.approx(0.5)
    summary = metric.summarize(out, settings)
    assert summary.iloc[0]["intergroup_contact"] == pytest.approx((0.375 + 0.5) / 2)


def test_no_mass_segment_is_nan():
    table = _edge_table({(0, 1, 0): {"low": 0.0, "high": 0.0}})
    for name in ("experienced_segregation", "intergroup_contact"):
        out = get_metric(name)().compute(table, _settings())
        assert math.isnan(out.iloc[0][name])


def test_both_metrics_registered():
    from segx.metrics.registry import available_metrics

    for name in (
        "experienced_segregation",
        "intergroup_contact",
        "exposure",
        "dominant_trajectories",
        "segregated_trajectories",
    ):
        assert name in available_metrics()
