"""P7.5: OD destination spread — hand-computed ground truth on a tiny OD table.

Every expectation here is worked out by hand in the comments, in the style of
``tests/unit/test_entropy.py``: the whole point of the analysis is that its
numbers are read as findings, so they have to be checkable without running it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from segx.analysis.od_spread import (
    analyse_od_spread,
    class_destination_overlap,
    class_flows,
    destination_spread,
    od_spread_unavailable_reason,
    summarize_destination_spread,
)
from segx.config import (
    AnalysisConfig,
    DemographicAttribute,
    DemographicSchema,
    ODColumnMapping,
    ODSpreadConfig,
    Settings,
)
from segx.errors import SchemaError
from segx.io.od_pairs import load_od_pairs

# Porto Alegre-ish lon/lat, so the auto-UTM working CRS is 32722 and the
# destination offsets are metres.
ORIGIN_LON, ORIGIN_LAT = -51.20, -30.00


def _settings(**kwargs) -> Settings:
    """LODES-shaped schema: the class counts live on the OD file itself."""
    schema = DemographicSchema(
        attributes=[
            DemographicAttribute(
                name="income",
                kind="categorical",
                categories=["low", "high"],
                class_columns=["SE01", "SE02"],
            )
        ],
        attribute_source="od",
    )
    return Settings(demographics=schema, od_columns=ODColumnMapping(), **kwargs)


def _write(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(origin: str, dest: str, dx_deg: float, se01: float, se02: float) -> dict:
    return {
        "origin_id": origin,
        "dest_id": dest,
        "origin_lon": ORIGIN_LON,
        "origin_lat": ORIGIN_LAT,
        "dest_lon": ORIGIN_LON + dx_deg,
        "dest_lat": ORIGIN_LAT,
        "SE01": se01,
        "SE02": se02,
    }


def _load(tmp_path, rows: list[dict], settings: Settings):
    path = tmp_path / "od.csv"
    _write(path, rows)
    return load_od_pairs(path, settings, working_crs="EPSG:32722")


# --------------------------------------------------------------------------- #
# Destination identity                                                        #
# --------------------------------------------------------------------------- #


def test_dest_id_from_column(tmp_path) -> None:
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 1, 0), _row("a", "w2", 0.02, 1, 0)], settings)
    assert list(loaded.gdf["dest_id"]) == ["w1", "w2"]


def test_dest_id_falls_back_to_id_ref(tmp_path) -> None:
    """LODES names no `dest_id`; the `wGEOID` that placed the endpoint *is* it."""
    settings = _settings()
    path = tmp_path / "od.csv"
    _write(
        path,
        [
            {"origin_lon": ORIGIN_LON, "origin_lat": ORIGIN_LAT, "dest_lon": -51.1, "dest_lat": -30.0, "wGEOID": "42"},
        ],
    )
    settings = settings.model_copy(update={"od_columns": ODColumnMapping(dest_id_ref="wGEOID")})
    loaded = load_od_pairs(path, settings, working_crs="EPSG:32722")
    assert list(loaded.gdf["dest_id"]) == ["42"]


def test_dest_id_falls_back_to_coordinates(tmp_path) -> None:
    """Two flows ending in the same place are one destination, not two rows."""
    settings = _settings()
    path = tmp_path / "od.csv"
    _write(
        path,
        [
            {"origin_lon": ORIGIN_LON, "origin_lat": ORIGIN_LAT, "dest_lon": -51.1, "dest_lat": -30.0},
            {"origin_lon": -51.19, "origin_lat": -30.0, "dest_lon": -51.1, "dest_lat": -30.0},
        ],
    )
    loaded = load_od_pairs(path, settings, working_crs="EPSG:32722")
    ids = list(loaded.gdf["dest_id"])
    assert ids[0] == ids[1]
    assert ids[0].startswith("@")


# --------------------------------------------------------------------------- #
# Availability                                                                #
# --------------------------------------------------------------------------- #


def test_unavailable_when_classes_live_on_demographics(tmp_path) -> None:
    """The measure is a tautology when every flow carries the origin's whole
    composition — so it says so rather than producing identical rows per class."""
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 1, 1)], settings)
    from_demographics = settings.model_copy(
        update={"demographics": settings.demographics.model_copy(update={"attribute_source": "demographics"})}
    )
    reason = od_spread_unavailable_reason(loaded, from_demographics)
    assert reason is not None
    assert "attribute_source" in reason
    with pytest.raises(SchemaError, match="attribute_source"):
        analyse_od_spread(loaded, from_demographics)


def test_available_for_a_lodes_shaped_file(tmp_path) -> None:
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 1, 1)], settings)
    assert od_spread_unavailable_reason(loaded, settings) is None


def test_disabled_by_config(tmp_path) -> None:
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 1, 1)], settings)
    off = settings.model_copy(update={"analysis": AnalysisConfig(od_spread=ODSpreadConfig(enabled=False))})
    assert od_spread_unavailable_reason(loaded, off) == "analysis.od_spread.enabled is false"


# --------------------------------------------------------------------------- #
# The flow table                                                              #
# --------------------------------------------------------------------------- #


def test_class_flows_drops_zero_mass_and_sums_duplicates(tmp_path) -> None:
    """A zero-count class on a flow is not a destination of that class; and the
    same (origin, destination) appearing twice is still one destination."""
    settings = _settings()
    loaded = _load(
        tmp_path,
        [
            _row("a", "w1", 0.01, 2, 0),  # low only
            _row("a", "w1", 0.01, 3, 0),  # same pair again -> summed
            _row("a", "w2", 0.02, 0, 5),  # high only
        ],
        settings,
    )
    flows = class_flows(loaded, settings)
    low = flows[flows["class"] == "low"]
    assert len(low) == 1  # w1 only, the two rows collapsed
    assert low["mass"].iloc[0] == pytest.approx(5.0)
    assert set(flows[flows["class"] == "high"]["dest_id"]) == {"w2"}


def test_min_mass_filters_the_long_tail(tmp_path) -> None:
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 1, 0), _row("a", "w2", 0.02, 9, 0)], settings)
    filtered = settings.model_copy(update={"analysis": AnalysisConfig(od_spread=ODSpreadConfig(min_mass=1.0))})
    flows = class_flows(loaded, filtered)
    assert set(flows[flows["class"] == "low"]["dest_id"]) == {"w2"}


# --------------------------------------------------------------------------- #
# Per-origin measures — hand-computed                                         #
# --------------------------------------------------------------------------- #


def test_destination_spread_hand_computed(tmp_path) -> None:
    """One origin, low goes 3/1 to two places, high goes 2/2/2/2 to four.

    low:  p = (0.75, 0.25)     H = -0.75·log₂0.75 − 0.25·log₂0.25 = 0.811278
          effective = 2^H = 1.7548 · top_share = 0.75 · Σp² = 0.625
    high: p = (0.25 ×4)        H = 2 → effective = 4 · top_share = 0.25 · Σp² = 0.25
    """
    settings = _settings()
    loaded = _load(
        tmp_path,
        [
            _row("a", "w1", 0.01, 3, 2),
            _row("a", "w2", 0.02, 1, 2),
            _row("a", "w3", 0.03, 0, 2),
            _row("a", "w4", 0.04, 0, 2),
        ],
        settings,
    )
    per_origin = destination_spread(class_flows(loaded, settings), settings).set_index("class")

    low = per_origin.loc["low"]
    assert low["mass"] == pytest.approx(4.0)
    assert low["n_destinations"] == pytest.approx(2.0)
    assert low["effective_destinations"] == pytest.approx(2**0.8112781244591328)
    assert low["top_share"] == pytest.approx(0.75)
    assert low["concentration"] == pytest.approx(0.625)

    high = per_origin.loc["high"]
    assert high["n_destinations"] == pytest.approx(4.0)
    assert high["effective_destinations"] == pytest.approx(4.0)
    assert high["concentration"] == pytest.approx(0.25)


def test_absent_class_is_nan_not_zero(tmp_path) -> None:
    """Nobody of this class lives here is not this class goes nowhere (invariant #2)."""
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 4, 0)], settings)
    per_origin = destination_spread(class_flows(loaded, settings), settings).set_index("class")
    high = per_origin.loc["high"]
    assert high["mass"] == 0.0
    assert math.isnan(high["n_destinations"])
    assert math.isnan(high["effective_destinations"])


def test_single_destination_is_one_not_undefined(tmp_path) -> None:
    """A class that really does go to exactly one place scores 1, not NaN — that
    is the maximally confined reading the measure exists to report."""
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.01, 7, 1)], settings)
    per_origin = destination_spread(class_flows(loaded, settings), settings).set_index("class")
    assert per_origin.loc["low", "n_destinations"] == pytest.approx(1.0)
    assert per_origin.loc["low", "effective_destinations"] == pytest.approx(1.0)
    assert per_origin.loc["low", "top_share"] == pytest.approx(1.0)


def test_radius_of_gyration_is_zero_for_one_destination(tmp_path) -> None:
    """Distance from home and size of the activity space are different things: a
    class commuting far to a single place has a large mean distance and no spread."""
    settings = _settings()
    loaded = _load(tmp_path, [_row("a", "w1", 0.05, 6, 0)], settings)
    low = destination_spread(class_flows(loaded, settings), settings).set_index("class").loc["low"]
    assert low["radius_of_gyration_m"] == pytest.approx(0.0, abs=1e-6)
    assert low["mean_distance_m"] > 4000.0  # ~0.05° of longitude at this latitude


def test_rarefaction_removes_the_group_size_confound(tmp_path) -> None:
    """The measured Philadelphia failure, in miniature.

    Both classes spread themselves identically — every worker to their own
    workplace — but ``high`` has five times the workers, so it touches five times
    the destinations. The raw count reports a 5× gap that is not there; the
    rarefied count, taken at the same head-count, reports none.

    The exact value pins the *hypergeometric* form: two workers of a two-person
    class drawn without replacement always reach both of their workplaces, so
    both classes score exactly 2.0. The with-replacement approximation scores low
    1.5 and high 1.9 here — it charges the smaller class for collisions between
    people who are distinct, which is the group-size bias creeping back in
    through the correction.
    """
    settings = _settings()
    rows = [_row("a", f"w{i}", 0.01 * (i + 1), 1 if i < 2 else 0, 1) for i in range(10)]
    per_origin = destination_spread(class_flows(_load(tmp_path, rows, settings), settings), settings).set_index("class")

    assert per_origin.loc["low", "n_destinations"] == pytest.approx(2.0)
    assert per_origin.loc["high", "n_destinations"] == pytest.approx(10.0)
    # Same n (the smaller class's 2 workers), and both are uniform over their own
    # destinations, so the two rarefied counts must agree.
    assert per_origin.loc["low", "rarefaction_n"] == pytest.approx(2.0)
    assert per_origin.loc["high", "rarefaction_n"] == pytest.approx(2.0)
    assert per_origin.loc["low", "rarefied_destinations"] == pytest.approx(2.0)
    assert per_origin.loc["high", "rarefied_destinations"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Overlap                                                                     #
# --------------------------------------------------------------------------- #


def test_overlap_disjoint_destinations(tmp_path) -> None:
    """Two classes, three destinations each, none shared: the segregation case
    that per-class counts alone cannot see (both would report 3)."""
    settings = _settings()
    rows = [_row("a", f"w{i}", 0.01 * (i + 1), 1 if i < 3 else 0, 0 if i < 3 else 1) for i in range(6)]
    overlap = class_destination_overlap(class_flows(_load(tmp_path, rows, settings), settings), settings)
    row = overlap.iloc[0]
    assert row["n_shared"] == 0
    assert row["jaccard"] == pytest.approx(0.0)
    assert row["cosine"] == pytest.approx(0.0)
    assert row["shared_share_a"] == pytest.approx(0.0)


def test_overlap_identical_destinations(tmp_path) -> None:
    settings = _settings()
    rows = [_row("a", f"w{i}", 0.01 * (i + 1), 1, 1) for i in range(4)]
    overlap = class_destination_overlap(class_flows(_load(tmp_path, rows, settings), settings), settings)
    row = overlap.iloc[0]
    assert row["n_shared"] == 4
    assert row["jaccard"] == pytest.approx(1.0)
    assert row["cosine"] == pytest.approx(1.0)
    assert row["shared_share_a"] == pytest.approx(1.0)
    assert row["shared_share_b"] == pytest.approx(1.0)


def test_overlap_is_directional(tmp_path) -> None:
    """A small class fully inside a large one's destination set is not the same
    fact as the large one being inside the small one's — one number cannot say it.

    low goes only to w1. high spreads 1 worker over w1..w4. So *all* of low's flow
    lands where high also goes (shared_share_a = 1.0) while only a quarter of
    high's lands where low goes (shared_share_b = 0.25).
    """
    settings = _settings()
    rows = [_row("a", f"w{i + 1}", 0.01 * (i + 1), 4 if i == 0 else 0, 1) for i in range(4)]
    overlap = class_destination_overlap(class_flows(_load(tmp_path, rows, settings), settings), settings)
    row = overlap.iloc[0]
    assert (row["class_a"], row["class_b"]) == ("low", "high")
    assert row["shared_share_a"] == pytest.approx(1.0)
    assert row["shared_share_b"] == pytest.approx(0.25)
    assert row["jaccard"] == pytest.approx(0.25)


def test_overlap_skips_origins_where_a_class_is_absent(tmp_path) -> None:
    """An overlap with a class that is not there is undefined, not zero."""
    settings = _settings()
    rows = [_row("a", "w1", 0.01, 3, 0), _row("b", "w1", 0.01, 1, 1)]
    overlap = class_destination_overlap(class_flows(_load(tmp_path, rows, settings), settings), settings)
    assert list(overlap["origin_id"]) == ["b"]


# --------------------------------------------------------------------------- #
# Summary + end to end                                                        #
# --------------------------------------------------------------------------- #


def test_summary_is_mass_weighted(tmp_path) -> None:
    """Two origins, one with 1 worker reaching 1 place, one with 9 reaching 3.

    The per-block mean would be 2.0; the mass-weighted mean — the spread of the
    typical *person* — is (1·1 + 9·3)/10 = 2.8.
    """
    settings = _settings()
    rows = [
        {**_row("a", "w1", 0.01, 1, 0)},
        {**_row("b", "w1", 0.01, 3, 0)},
        {**_row("b", "w2", 0.02, 3, 0)},
        {**_row("b", "w3", 0.03, 3, 0)},
    ]
    flows = class_flows(_load(tmp_path, rows, settings), settings)
    summary = summarize_destination_spread(destination_spread(flows, settings), flows).set_index("class")
    assert summary.loc["low", "n_origins"] == 2
    assert summary.loc["low", "mass_total"] == pytest.approx(10.0)
    assert summary.loc["low", "n_destinations"] == pytest.approx(2.8)
    # Citywide, low reaches three distinct workplaces across both origins.
    assert summary.loc["low", "n_destinations_pooled"] == pytest.approx(3.0)


def test_paired_summary_uses_only_shared_origins(tmp_path) -> None:
    settings = _settings()
    rows = [
        _row("a", "w1", 0.01, 2, 0),  # low only -> not paired
        _row("b", "w1", 0.01, 2, 2),  # both present -> paired
        _row("b", "w2", 0.02, 2, 2),
    ]
    flows = class_flows(_load(tmp_path, rows, settings), settings)
    summary = summarize_destination_spread(destination_spread(flows, settings), flows).set_index("class")
    assert summary.loc["low", "n_origins"] == 2
    assert summary.loc["low", "n_origins_paired"] == 1
    assert summary.loc["high", "n_origins_paired"] == 1


def test_analyse_od_spread_end_to_end(tmp_path) -> None:
    settings = _settings()
    rows = [_row("a", f"w{i}", 0.01 * (i + 1), 2, 1) for i in range(3)]
    result = analyse_od_spread(_load(tmp_path, rows, settings), settings)
    assert result.stats["n_od_rows"] == 3
    assert result.stats["n_origins"] == 1
    assert result.stats["distance_basis"] == "euclidean_input_geometry"
    assert set(result.summary["class"]) == {"low", "high"}
    assert not result.overlap.empty
    assert len(result.overlap_summary) == 1


def test_warns_when_every_origin_has_one_destination(tmp_path) -> None:
    """An OD file with no origin id gives one origin per row — every measure is
    then trivially 1, which looks exactly like a real result in a spreadsheet."""
    settings = _settings()
    path = tmp_path / "od.csv"
    _write(
        path,
        [
            {k: v for k, v in _row("ignored", "w1", 0.01, 1, 1).items() if k not in ("origin_id", "dest_id")},
            {k: v for k, v in _row("ignored", "w2", 0.02, 1, 1).items() if k not in ("origin_id", "dest_id")},
        ],
    )
    loaded = load_od_pairs(path, settings, working_crs="EPSG:32722")
    with pytest.warns(UserWarning, match="single destination"):
        analyse_od_spread(loaded, settings)
