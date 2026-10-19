"""Social exposure — per-group mean segment diversity over traversed segments,
weighted by the group's presence. Hand-computed ground truth."""

from __future__ import annotations

import pandas as pd
import pytest

from segx.config import DemographicAttribute, DemographicSchema, Settings
from segx.metrics.entropy import shannon_entropy
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


def test_per_group_exposure_is_presence_weighted_mean_of_segment_entropy():
    # seg A: {3 low, 1 high} → H_A;  seg B: {1 low, 1 high} → H_B = 1 bit
    table = _edge_table({(0, 1, 0): {"low": 3.0, "high": 1.0}, (1, 2, 0): {"low": 1.0, "high": 1.0}})
    out = get_metric("exposure")().compute(table, _settings())
    exposure = {r["class"]: r.exposure for _, r in out.iterrows()}

    h_a = shannon_entropy([3.0, 1.0])
    h_b = shannon_entropy([1.0, 1.0])
    # low present 3 on A, 1 on B → weighted mean
    assert exposure["low"] == pytest.approx((3.0 * h_a + 1.0 * h_b) / 4.0)
    # high present 1 on A, 1 on B
    assert exposure["high"] == pytest.approx((1.0 * h_a + 1.0 * h_b) / 2.0)


def test_group_absent_from_all_traversed_segments_is_nan():
    # high never appears (mass 0 everywhere) → its exposure is undefined, not 0
    table = _edge_table({(0, 1, 0): {"low": 4.0, "high": 0.0}, (1, 2, 0): {"low": 2.0, "high": 0.0}})
    out = get_metric("exposure")().compute(table, _settings())
    exposure = {r["class"]: r.exposure for _, r in out.iterrows()}
    # every traversed segment is single-class → H is 0 for low (defined), high absent → NaN
    assert exposure["low"] == pytest.approx(0.0)
    assert pd.isna(exposure["high"])


def test_empty_table():
    out = get_metric("exposure")().compute(pd.DataFrame(), _settings())
    assert out.empty
    assert "class" in out.columns and "exposure" in out.columns
