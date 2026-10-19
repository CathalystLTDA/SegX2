"""Cross-mode exposure tests (P7.1) — hand-computable throughout.

The whole fixture is one segment, ``(0, 1, 0)``, ``layer="street"``, with a single
``income`` attribute over ``{low, high}`` and two modes depositing on it:

======  =====  ======
        low    high
======  =====  ======
walk    3      1
transit 0      4
======  =====  ======

With ``factor = 0.5``:

* observer **walk** sees ``(3 + 0.5·0, 1 + 0.5·4) = (3, 3)`` -> ``H = 1.0`` bit
  exactly (plain ``exposure`` would give ``H(3, 1) = 0.8113``);
* observer **transit** sees ``(0 + 0.5·3, 4 + 0.5·1) = (1.5, 4.5) ≡ (1, 3)`` ->
  ``H = 0.8113``.

Two clean numbers, and they demonstrate the asymmetry the report has to state:
the small mode is dominated by the large one, not the other way round.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from segx.config import (
    CrossModeExposureConfig,
    DemographicAttribute,
    DemographicSchema,
    MetricsConfig,
    Settings,
)
from segx.metrics.cross_mode_exposure import (
    CrossModeExposureMetric,
    cross_mode_combination_counts,
    cross_mode_stats,
)
from segx.metrics.exposure import ExposureMetric

H_3_3 = 1.0
H_3_1 = 0.8112781244591328


def _settings(factor: float = 0.5, overrides: dict[str, float] | None = None) -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        ),
        metrics=MetricsConfig(
            cross_mode=CrossModeExposureConfig(factor=factor, overrides=dict(overrides or {})),
        ),
    )


def _rows(records: list[tuple]) -> pd.DataFrame:
    """``(mode, u, v, key, layer, class, mass)`` -> the accumulator's frame shape."""
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "category": "commute",
                "binding": "od",
                "u": u,
                "v": v,
                "key": key,
                "layer": layer,
                "attribute": "income",
                "class": cls,
                "mass": mass,
                "n_paths": 1,
            }
            for mode, u, v, key, layer, cls, mass in records
        ]
    )


def _one_segment() -> pd.DataFrame:
    return _rows(
        [
            ("walk", 0, 1, 0, "street", "low", 3.0),
            ("walk", 0, 1, 0, "street", "high", 1.0),
            ("transit", 0, 1, 0, "street", "low", 0.0),
            ("transit", 0, 1, 0, "street", "high", 4.0),
        ]
    )


def _by_mode_class(frame: pd.DataFrame, column: str) -> dict[tuple[str, str], float]:
    return {(str(r["mode"]), str(r["class"])): float(r[column]) for _, r in frame.iterrows()}


# --------------------------------------------------------------------------- #
# The formulation                                                              #
# --------------------------------------------------------------------------- #


def test_hand_computed_ground_truth_and_the_asymmetry() -> None:
    got = _by_mode_class(CrossModeExposureMetric().compute(_one_segment(), _settings()), "cross_mode_exposure")
    # walk: effective (3, 3) -> a perfectly mixed bit, up from H(3, 1) = 0.8113
    assert got[("walk", "low")] == pytest.approx(H_3_3)
    assert got[("walk", "high")] == pytest.approx(H_3_3)
    # transit: effective (1.5, 4.5) == (1, 3) -> dominated by the pedestrians
    assert got[("transit", "high")] == pytest.approx(H_3_1)
    # `low` has no transit presence on this segment, so transit's `low` group has
    # nothing to average over — NaN, never 0.0 (invariant #2)
    assert math.isnan(got[("transit", "low")])


def test_a_factor_of_zero_reproduces_exposure_row_for_row() -> None:
    """The free sanity check: no cross term means the plain P5 measure."""
    table = _one_segment()
    settings = _settings(factor=0.0)
    cross = _by_mode_class(CrossModeExposureMetric().compute(table, settings), "cross_mode_exposure")
    plain = _by_mode_class(ExposureMetric().compute(table, settings), "exposure")
    assert cross.keys() == plain.keys()
    for key, value in plain.items():
        if math.isnan(value):
            assert math.isnan(cross[key])
        else:
            assert cross[key] == pytest.approx(value)


def test_a_factor_of_one_pools_the_modes_completely() -> None:
    """Every observer then sees the same crowd, so `H_cross` is mode-independent."""
    got = _by_mode_class(
        CrossModeExposureMetric().compute(_one_segment(), _settings(factor=1.0)), "cross_mode_exposure"
    )
    pooled = 0.954434002924965  # H(3, 5) -- the same crowd for both observers
    assert got[("walk", "low")] == pytest.approx(pooled)
    assert got[("walk", "high")] == pytest.approx(pooled)
    assert got[("transit", "high")] == pytest.approx(pooled)


def test_cross_mode_exposure_can_fall_below_exposure() -> None:
    """Counterintuitive but correct: a more concentrated neighbour *lowers* H."""
    table = _rows(
        [
            ("walk", 0, 1, 0, "street", "low", 2.0),
            ("walk", 0, 1, 0, "street", "high", 2.0),  # perfectly mixed on its own
            ("transit", 0, 1, 0, "street", "low", 8.0),  # single-class
            ("transit", 0, 1, 0, "street", "high", 0.0),
        ]
    )
    settings = _settings()
    cross = _by_mode_class(CrossModeExposureMetric().compute(table, settings), "cross_mode_exposure")
    plain = _by_mode_class(ExposureMetric().compute(table, settings), "exposure")
    assert plain[("walk", "low")] == pytest.approx(1.0)
    assert cross[("walk", "low")] < plain[("walk", "low")]


# --------------------------------------------------------------------------- #
# The layer filter (D4)                                                        #
# --------------------------------------------------------------------------- #


def test_a_rail_segment_is_excluded_by_its_layer_not_by_its_node_ids() -> None:
    """A ``rail_line`` row is given **positive** ids colliding with a street row on
    purpose: if the exclusion depended on the negative-id convention of synthetic
    transit nodes rather than on ``layer``, this would mix them."""
    table = _rows(
        [
            ("walk", 0, 1, 0, "street", "low", 3.0),
            ("walk", 0, 1, 0, "street", "high", 1.0),
            ("transit", 0, 1, 0, "rail_line", "low", 0.0),
            ("transit", 0, 1, 0, "rail_line", "high", 4.0),
        ]
    )
    settings = _settings()
    # ...and because nothing mixed, the coverage warning fires — correctly: this
    # run produced no cross-mode co-presence at all. The warning names the dead
    # pair (P7.2): a global rate would have been carried by whichever pair *did*
    # join, which is exactly how the Philadelphia `walk ∩ drive` zero hid.
    with pytest.warns(UserWarning, match=r"transit\|walk"):
        cross = _by_mode_class(CrossModeExposureMetric().compute(table, settings), "cross_mode_exposure")
    plain = _by_mode_class(ExposureMetric().compute(table, settings), "exposure")
    # the pedestrian does not meet the metro rider under the street...
    assert cross[("walk", "low")] == pytest.approx(plain[("walk", "low")])
    # ...and the rider in the tunnel keeps their own uncontaminated composition
    assert cross[("transit", "high")] == pytest.approx(plain[("transit", "high")])


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


def test_overrides_are_order_independent() -> None:
    a = _settings(overrides={"transit|walk": 0.9})
    b = _settings(overrides={"walk|transit": 0.9})
    table = _one_segment()
    got_a = _by_mode_class(CrossModeExposureMetric().compute(table, a), "cross_mode_exposure")
    got_b = _by_mode_class(CrossModeExposureMetric().compute(table, b), "cross_mode_exposure")
    assert got_a == pytest.approx(got_b, nan_ok=True)
    assert a.metrics.cross_mode.overrides == {"transit|walk": 0.9}  # stored canonically
    assert a.metrics.cross_mode.factor_for("walk", "transit") == pytest.approx(0.9)
    assert a.metrics.cross_mode.factor_for("walk", "walk") == 1.0
    assert a.metrics.cross_mode.factor_for("walk", "drive") == pytest.approx(0.5)  # falls back to `factor`


def test_an_override_survives_the_settings_override_round_trip() -> None:
    """``Settings._with_overrides`` dumps and re-validates, so a non-canonical key
    would have to be canonicalised twice — or leak through."""
    settings = Settings().model_copy()
    updated = settings._with_overrides({"metrics.cross_mode.overrides": {"walk|transit": 0.7}})
    assert updated.metrics.cross_mode.overrides == {"transit|walk": 0.7}
    assert updated._with_overrides({"metrics.cross_mode.factor": 0.25}).metrics.cross_mode.overrides == {
        "transit|walk": 0.7
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"walk|bicycle": 0.5}, "unknown mode"),
        ({"walk|walk": 0.5}, "same mode twice"),
        ({"walk": 0.5}, "mode_a|mode_b"),
        ({"walk|transit": -0.1}, ">= 0"),
    ],
)
def test_bad_overrides_are_rejected(overrides: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CrossModeExposureConfig(overrides=overrides)


def test_a_negative_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="factor must be >= 0"):
        CrossModeExposureConfig(factor=-0.5)


# --------------------------------------------------------------------------- #
# The coverage audit                                                           #
# --------------------------------------------------------------------------- #


def test_join_rate_counts_multimode_segments() -> None:
    table = pd.concat(
        [
            _one_segment(),
            _rows(
                [
                    ("walk", 1, 2, 0, "street", "low", 1.0),
                    ("walk", 1, 2, 0, "street", "high", 1.0),
                ]
            ),
        ],
        ignore_index=True,
    )
    stats = cross_mode_stats(table)
    assert stats["n_modes_present"] == 2
    assert stats["n_segments_total"] == 2
    assert stats["n_segments_multimode"] == 1
    assert stats["cross_mode_join_rate"] == pytest.approx(0.5)


def test_combination_counts_break_the_join_rate_out_by_which_modes_met() -> None:
    """One row per *exact* mode set, so a low join rate can be diagnosed.

    Segments here: (0,1) walk+transit, (1,2) walk only, (2,3) all three, (3,4)
    drive only, and (9,9) a rail_line segment that the pooling excludes and this
    table must still *show* rather than drop."""
    table = pd.concat(
        [
            _one_segment(),  # (0, 1, 0): walk + transit
            _rows(
                [
                    ("walk", 1, 2, 0, "street", "low", 1.0),
                    ("walk", 2, 3, 0, "street", "low", 1.0),
                    ("drive", 2, 3, 0, "street", "low", 1.0),
                    ("transit", 2, 3, 0, "street", "low", 1.0),
                    ("drive", 3, 4, 0, "street", "high", 1.0),
                    ("transit", 9, 9, 0, "rail_line", "high", 1.0),
                ]
            ),
        ],
        ignore_index=True,
    )
    got = cross_mode_combination_counts(table)
    counts = {(str(r["layer"]), str(r["modes"])): int(r["n_segments"]) for _, r in got.iterrows()}
    assert counts == {
        ("rail_line", "transit"): 1,
        ("street", "drive"): 1,
        ("street", "walk"): 1,
        ("street", "transit+walk"): 1,
        ("street", "drive+transit+walk"): 1,
    }
    # every street segment is counted exactly once, so the street rows reconcile
    # with `cross_mode_stats` rather than telling a different story
    street = got[got["layer"] == "street"]
    stats = cross_mode_stats(table)
    assert int(street["n_segments"].sum()) == stats["n_segments_total"]
    assert int(street.loc[street["n_modes"] > 1, "n_segments"].sum()) == stats["n_segments_multimode"]
    # sorted single-mode first, so the table reads top to bottom
    assert list(street["n_modes"]) == sorted(street["n_modes"])


def test_combination_counts_of_an_empty_table_are_an_empty_frame() -> None:
    empty = pd.DataFrame(
        columns=["mode", "category", "binding", "u", "v", "key", "layer", "attribute", "class", "mass"]
    )
    out = cross_mode_combination_counts(empty)
    assert out.empty
    assert list(out.columns) == ["layer", "category", "binding", "modes", "n_modes", "n_segments"]


def test_two_modes_that_never_share_a_segment_warn() -> None:
    """The failure mode is not a zero — it is `cross_mode_exposure == exposure`,
    which reads as a finding. Saying so is what saves a wasted week."""
    table = _rows(
        [
            ("walk", 0, 1, 0, "street", "low", 3.0),
            ("walk", 0, 1, 0, "street", "high", 1.0),
            ("transit", 5, 6, 0, "street", "low", 0.0),
            ("transit", 5, 6, 0, "street", "high", 4.0),
        ]
    )
    assert cross_mode_stats(table)["cross_mode_join_rate"] == 0.0
    assert cross_mode_stats(table)["pair_join_rates"]["transit|walk"]["n_shared"] == 0
    with pytest.warns(UserWarning, match=r"transit\|walk"):
        got = CrossModeExposureMetric().compute(table, _settings())
    plain = _by_mode_class(ExposureMetric().compute(table, _settings()), "exposure")
    assert _by_mode_class(got, "cross_mode_exposure") == pytest.approx(plain, nan_ok=True)


def test_a_single_mode_table_does_not_warn(recwarn) -> None:
    table = _rows(
        [
            ("walk", 0, 1, 0, "street", "low", 3.0),
            ("walk", 0, 1, 0, "street", "high", 1.0),
        ]
    )
    CrossModeExposureMetric().compute(table, _settings())
    assert [w for w in recwarn if "cross_mode" in str(w.message)] == []


def test_an_empty_table_is_an_empty_frame() -> None:
    empty = pd.DataFrame(
        columns=["mode", "category", "binding", "u", "v", "key", "layer", "attribute", "class", "mass"]
    )
    out = CrossModeExposureMetric().compute(empty, _settings())
    assert out.empty
    assert list(out.columns) == ["mode", "category", "binding", "attribute", "class", "cross_mode_exposure"]
    assert cross_mode_stats(empty)["n_modes_present"] == 0


def test_the_metric_is_registered_and_summarizes_like_exposure() -> None:
    from segx.metrics.registry import available_metrics, get_metric

    assert "cross_mode_exposure" in available_metrics()
    metric = get_metric("cross_mode_exposure")()
    settings = _settings()
    per_segment = metric.compute(_one_segment(), settings)
    summary = metric.summarize(per_segment, settings)
    assert list(summary.columns) == ["mode", "category", "binding", "attribute", "cross_mode_exposure"]
    walk = summary[summary["mode"] == "walk"].iloc[0]
    assert float(walk["cross_mode_exposure"]) == pytest.approx(H_3_3)


# --------------------------------------------------------------------------- #
# Corridor pooling (P7.2)                                                      #
# --------------------------------------------------------------------------- #


def _corridor_rows(records: list[tuple]) -> pd.DataFrame:
    """``(mode, u, v, key, corridor, length_m, class, mass)`` -> frame shape.

    The post-P7.2 accumulator output: same as :func:`_rows` plus the canonical
    street space each segment sits in and how long it is.
    """
    return pd.DataFrame(
        [
            {
                "mode": mode,
                "category": "commute",
                "binding": "od",
                "u": u,
                "v": v,
                "key": key,
                "layer": "street",
                "corridor": corridor,
                "length_m": length,
                "attribute": "income",
                "class": cls,
                "mass": mass,
                "n_paths": 1,
            }
            for mode, u, v, key, corridor, length, cls, mass in records
        ]
    )


def test_a_car_and_a_pedestrian_on_different_ways_meet_through_the_corridor() -> None:
    """The defect this phase exists for. The car is on the roadway `(1, 2, 0)`, the
    pedestrian on the separately-mapped sidewalk `(10, 11, 0)` — different OSM ways,
    so the pre-P7.2 `(u, v, key)` join found nothing and the drive number was a
    fiction. Sharing a corridor, the hand-computed `H(3, 3) = 1.0` comes back."""
    table = _corridor_rows(
        [
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "low", 3.0),
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "high", 1.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "low", 0.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "high", 4.0),
        ]
    )
    got = _by_mode_class(CrossModeExposureMetric().compute(table, _settings()), "cross_mode_exposure")
    assert got[("walk", "low")] == pytest.approx(H_3_3)
    assert got[("drive", "high")] == pytest.approx(H_3_1)

    stats = cross_mode_stats(table)
    assert stats["pair_join_rates"]["drive|walk"]["n_shared"] == 1
    assert stats["cross_mode_join_rate"] == pytest.approx(1.0)


def test_one_edge_per_mode_per_corridor_reproduces_the_pre_corridor_result() -> None:
    """Non-regression: where the ids already matched, adding corridors changes
    nothing. The length-weighted mean over a single edge is that edge's mass."""
    plain = _one_segment()
    with_corridor = _corridor_rows(
        [
            ("walk", 0, 1, 0, "e:0:1:0", 100.0, "low", 3.0),
            ("walk", 0, 1, 0, "e:0:1:0", 100.0, "high", 1.0),
            ("transit", 0, 1, 0, "e:0:1:0", 100.0, "low", 0.0),
            ("transit", 0, 1, 0, "e:0:1:0", 100.0, "high", 4.0),
        ]
    )
    before = _by_mode_class(CrossModeExposureMetric().compute(plain, _settings()), "cross_mode_exposure")
    after = _by_mode_class(CrossModeExposureMetric().compute(with_corridor, _settings()), "cross_mode_exposure")
    assert after == pytest.approx(before, nan_ok=True)


def test_pooling_uses_the_length_weighted_mean_not_the_sum() -> None:
    """A block's sidewalk is three OSM edges and its roadway is one. Summing would
    hand the driver three times the pedestrian mass — a number about how OSM was
    segmented, not about the city. The mean is the same 3/1 the driver would see
    from a single sidewalk edge, so `H(1.5, 4.5) = H(1, 3)` still holds."""
    sidewalk = [
        ("walk", 10 + i, 11 + i, 0, "e:1:2:0", 30.0, cls, mass)
        for i in range(3)
        for cls, mass in (("low", 3.0), ("high", 1.0))
    ]
    table = _corridor_rows(
        [
            *sidewalk,
            ("drive", 1, 2, 0, "e:1:2:0", 90.0, "low", 0.0),
            ("drive", 1, 2, 0, "e:1:2:0", 90.0, "high", 4.0),
        ]
    )
    got = _by_mode_class(CrossModeExposureMetric().compute(table, _settings()), "cross_mode_exposure")
    assert got[("drive", "high")] == pytest.approx(H_3_1)
    assert got[("walk", "low")] == pytest.approx(H_3_3)


def test_length_weighting_favours_the_longer_stretch() -> None:
    """Two sidewalk edges in one corridor, 90 m carrying 4 low and 10 m carrying 0.
    The weighted mean is 3.6, not the unweighted 2.0 — so an OSM stub at a corner
    cannot drag a whole block's pedestrian mass down by half."""
    table = _corridor_rows(
        [
            ("walk", 10, 11, 0, "e:1:2:0", 90.0, "low", 4.0),
            ("walk", 10, 11, 0, "e:1:2:0", 90.0, "high", 0.0),
            ("walk", 11, 12, 0, "e:1:2:0", 10.0, "low", 0.0),
            ("walk", 11, 12, 0, "e:1:2:0", 10.0, "high", 0.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "low", 0.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "high", 10.0),
        ]
    )
    got = _by_mode_class(CrossModeExposureMetric().compute(table, _settings()), "cross_mode_exposure")
    # driver sees (0 + 0.5·3.6, 10 + 0.5·0) = (1.8, 10)
    expected = -sum(p * math.log2(p) for p in (1.8 / 11.8, 10.0 / 11.8))
    assert got[("drive", "high")] == pytest.approx(expected)


def test_factor_zero_still_reproduces_exposure_with_corridors() -> None:
    """The property that guarantees the corridor never leaked into the P5 metrics:
    with no cross term, pooling scope is irrelevant and this is `exposure`."""
    table = _corridor_rows(
        [
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "low", 3.0),
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "high", 1.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "low", 0.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "high", 4.0),
        ]
    )
    settings = _settings(factor=0.0)
    cross = _by_mode_class(CrossModeExposureMetric().compute(table, settings), "cross_mode_exposure")
    plain = _by_mode_class(ExposureMetric().compute(table, settings), "exposure")
    assert cross == pytest.approx(plain, nan_ok=True)


def test_a_rail_segment_in_its_own_corridor_stays_uncontaminated() -> None:
    """Rail is excluded from corridor matching upstream, so it arrives with a
    corridor of its own; the layer filter then keeps it out of the pool too."""
    table = _corridor_rows(
        [
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "low", 3.0),
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "high", 1.0),
            ("transit", -1, -2, 0, "own:transit:-1:-2:0", 100.0, "low", 0.0),
            ("transit", -1, -2, 0, "own:transit:-1:-2:0", 100.0, "high", 4.0),
        ]
    )
    table.loc[table["mode"].eq("transit"), "layer"] = "rail_line"
    settings = _settings()
    cross = _by_mode_class(CrossModeExposureMetric().compute(table, settings), "cross_mode_exposure")
    plain = _by_mode_class(ExposureMetric().compute(table, settings), "exposure")
    assert cross[("transit", "high")] == pytest.approx(plain[("transit", "high")])
    assert cross[("walk", "low")] == pytest.approx(plain[("walk", "low")])


def test_combination_counts_group_by_corridor() -> None:
    """The acceptance table. Two OSM ways, one corridor -> one `drive+walk` row."""
    table = _corridor_rows(
        [
            ("walk", 10, 11, 0, "e:1:2:0", 100.0, "low", 3.0),
            ("drive", 1, 2, 0, "e:1:2:0", 100.0, "high", 4.0),
        ]
    )
    out = cross_mode_combination_counts(table)
    assert out["modes"].tolist() == ["drive+walk"]
    assert int(out["n_segments"].iloc[0]) == 1
