"""GTFS reader tests (P7) against the in-repo ``tests/fixtures/mini_gtfs`` feed.

The fixture is deliberately small enough that every derived quantity is
hand-checkable, and deliberately *awkward* in the ways real feeds are: a weekend
service that must not leak into a weekday run, trips outside the analysis window,
a short-turn trip that must not win the modal stop sequence, one hop whose running
time differs from the rest, and a route that leaves the study area and comes back.

Ground truth for ``BUS1`` (weekday, the default window 06:00-09:00 = 180 min):

* in-window trips: ``t0`` 06:00, ``t1`` 07:00, ``t2`` 07:30, ``t3`` 08:00, ``t4`` 08:30
  -> **5** (``t0`` sits exactly on the inclusive lower bound; ``t5`` 09:15 is outside;
  ``w1`` is Saturday-only)
* headway = 180 min / 5 trips = **36.0 min**
* modal sequence = ``S1 S2 S3 S4`` (5 trips) beats the short turn ``S1 S2 S3`` (1)
* mean ``S1->S2`` hop = (3 + 9 + 3 + 3 + 3 + 3) / 6 = **4.0 min**; the rest are 3.0
  (hop times average over *all* the day's trips, never the window's subset, so
  widening the window does not move them)

``BUS2`` carries the lower-bound coverage the wider default window would otherwise
lose: ``u0`` departs 05:30 (outside), ``u1`` 07:05 and ``u2`` 08:05 (inside) -> **2**
trips, headway 180 / 2 = **90.0 min**.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from segx.config import Settings, TransitConfig
from segx.errors import GTFSError, GTFSSchemaError
from segx.io.gtfs import (
    LoadedGTFS,
    active_service_ids,
    apply_overlay,
    build_patterns,
    build_stops_gdf,
    clip_patterns_to_boundary,
    feed_hash,
    gtfs_time_to_minutes,
    normalize_route_type,
    overlay_hash,
    read_cached_feed,
    read_feed_tables,
    transit_cache_key,
    write_cached_feed,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_gtfs"
WORKING_CRS = "EPSG:32618"


def _settings(**transit_kwargs) -> Settings:
    return Settings(transit=TransitConfig(enabled=True, **transit_kwargs))


def _patterns(settings: Settings | None = None):
    tables = read_feed_tables(FIXTURE)
    return build_patterns(tables, settings or _settings())


def _row(frame, pattern_id: str):
    return frame[frame["pattern_id"] == pattern_id].iloc[0]


# --------------------------------------------------------------------------- #
# Reading                                                                      #
# --------------------------------------------------------------------------- #


def test_reads_a_directory_and_a_zip_identically(tmp_path: Path) -> None:
    archive = tmp_path / "feed.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for member in FIXTURE.glob("*.txt"):
            zf.write(member, arcname=f"nested/{member.name}")  # agencies often nest one level
    from_dir = read_feed_tables(FIXTURE)
    from_zip = read_feed_tables(archive)
    assert sorted(from_dir) == sorted(from_zip)
    assert from_dir["stops.txt"].equals(from_zip["stops.txt"])


def test_ids_are_read_as_strings() -> None:
    """A stop id of ``"007"`` must not become ``7`` — every join depends on it."""
    tables = read_feed_tables(FIXTURE)
    assert tables["stops.txt"]["stop_id"].map(type).eq(str).all()


def test_missing_required_file_is_a_typed_schema_error(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    for name in ("stops.txt", "routes.txt", "trips.txt"):  # no stop_times.txt
        (partial / name).write_bytes((FIXTURE / name).read_bytes())
    with pytest.raises(GTFSSchemaError) as excinfo:
        read_feed_tables(partial)
    assert excinfo.value.code == "transit.gtfs_schema"
    assert "stop_times.txt" in excinfo.value.details["missing"]


def test_unreadable_zip_is_a_typed_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip at all")
    with pytest.raises(GTFSError) as excinfo:
        read_feed_tables(broken)
    assert excinfo.value.code == "transit.gtfs"


def test_missing_feed_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(GTFSError):
        read_feed_tables(tmp_path / "nope.zip")


def test_feed_hash_changes_when_content_changes(tmp_path: Path) -> None:
    a = tmp_path / "a.zip"
    b = tmp_path / "b.zip"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert feed_hash(a) != feed_hash(b)
    assert feed_hash(a) == feed_hash(a)


# --------------------------------------------------------------------------- #
# Time / calendar                                                              #
# --------------------------------------------------------------------------- #


def test_gtfs_times_past_midnight_are_not_wrapped() -> None:
    import pandas as pd

    got = gtfs_time_to_minutes(pd.Series(["07:14:00", "25:10:00", "", "00:00:30"]))
    assert got[0] == pytest.approx(434.0)
    assert got[1] == pytest.approx(25 * 60 + 10)  # a 25:10 departure sorts after 23:50
    assert pd.isna(got[2])
    assert got[3] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("day", "expected"),
    [("weekday", {"WD"}), ("saturday", {"SA"}), ("sunday", set())],
)
def test_active_service_ids_reads_calendar(day: str, expected: set[str]) -> None:
    assert active_service_ids(read_feed_tables(FIXTURE), day) == expected


def test_active_service_ids_is_none_without_a_calendar() -> None:
    tables = read_feed_tables(FIXTURE)
    tables.pop("calendar.txt")
    assert active_service_ids(tables, "weekday") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("1", 1), ("3", 3), ("11", 11), ("700", 3), ("401", 1), ("900", 0), ("nonsense", 3)],
)
def test_extended_route_types_collapse_onto_basic_ones(raw: str, expected: int) -> None:
    assert normalize_route_type(raw) == expected


# --------------------------------------------------------------------------- #
# Pattern derivation                                                           #
# --------------------------------------------------------------------------- #


def test_headway_counts_only_in_window_weekday_trips() -> None:
    patterns, _ = _patterns()
    bus = _row(patterns, "BUS1::0")
    assert bus["n_trips"] == pytest.approx(5.0)  # t0..t4; not t5 (window) nor w1 (Saturday)
    assert bus["headway_min"] == pytest.approx(36.0)  # 180 min / 5


def test_the_window_lower_bound_is_inclusive() -> None:
    """``t0`` departs at exactly 06:00, the default window's start — it counts."""
    patterns, _ = _patterns(_settings(service_window=("06:00", "06:30")))
    assert _row(patterns, "BUS1::0")["n_trips"] == pytest.approx(1.0)  # t0 only
    assert _row(patterns, "BUS1::0")["headway_min"] == pytest.approx(30.0)  # 30 / 1


def test_service_window_is_configurable() -> None:
    """A different window is a different headway — one config value, per D3."""
    patterns, _ = _patterns(_settings(service_window=("07:00", "08:00")))
    assert _row(patterns, "BUS1::0")["n_trips"] == pytest.approx(2.0)  # t1, t2
    assert _row(patterns, "BUS1::0")["headway_min"] == pytest.approx(30.0)  # 60 / 2


def test_saturday_run_sees_only_the_saturday_trip() -> None:
    patterns, _ = _patterns(_settings(service_day="saturday"))
    assert _row(patterns, "BUS1::0")["n_trips"] == pytest.approx(1.0)
    assert "MET1::0" not in set(patterns["pattern_id"])  # metro is weekday-only here


def test_modal_stop_sequence_beats_the_short_turn() -> None:
    _, pattern_stops = _patterns()
    sequence = list(pattern_stops[pattern_stops["pattern_id"] == "BUS1::0"]["stop_id"])
    assert sequence == ["S1", "S2", "S3", "S4"]


def test_inter_stop_times_are_means_over_trips() -> None:
    _, pattern_stops = _patterns()
    hops = pattern_stops[pattern_stops["pattern_id"] == "BUS1::0"].set_index("seq")["travel_time_min"]
    assert hops[0] == 0.0  # first stop of a pattern has no incoming hop
    assert hops[1] == pytest.approx(4.0)  # (3 + 9 + 3 + 3 + 3 + 3) / 6
    assert hops[2] == pytest.approx(3.0)
    assert hops[3] == pytest.approx(3.0)


def test_route_type_reaches_the_pattern_table() -> None:
    patterns, _ = _patterns()
    settings = _settings()
    assert _row(patterns, "BUS1::0")["route_type"] == 3
    assert _row(patterns, "MET1::0")["route_type"] == 1
    assert settings.transit.layer_for_route_type(3) == "street"
    assert settings.transit.layer_for_route_type(1) == "rail_line"


# --------------------------------------------------------------------------- #
# The service window (P7.1)                                                 #
# --------------------------------------------------------------------------- #

_CALENDAR = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "WD,1,1,1,1,1,0,0,20260101,20261231\n"
)
_ROUTES = "route_id,route_short_name,route_type\nR1,R1,3\n"
_STOPS = "stop_id,stop_name,stop_lat,stop_lon\nP1,P1,39.95,-75.16\nP2,P2,39.95,-75.158\nP3,P3,39.95,-75.156\n"


def _tiny_feed(tmp_path: Path, *, trips: str, stop_times: str, frequencies: str | None = None) -> Path:
    """A hand-written 3-stop feed, so a window edge case is one file to read."""
    root = tmp_path / "tiny"
    root.mkdir()
    (root / "calendar.txt").write_text(_CALENDAR, encoding="utf-8")
    (root / "routes.txt").write_text(_ROUTES, encoding="utf-8")
    (root / "stops.txt").write_text(_STOPS, encoding="utf-8")
    (root / "trips.txt").write_text(trips, encoding="utf-8")
    (root / "stop_times.txt").write_text(stop_times, encoding="utf-8")
    if frequencies is not None:
        (root / "frequencies.txt").write_text(frequencies, encoding="utf-8")
    return root


def _times(trip_id: str, *departures: str) -> str:
    stops = ["P1", "P2", "P3"]
    return "".join(
        f"{trip_id},{t},{t},{stop},{i + 1}\n" for i, (t, stop) in enumerate(zip(departures, stops, strict=True))
    )


_HEADER = "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"


def test_the_default_window_starts_at_six() -> None:
    """The early commute is real: 07:00 excluded the 6 a.m. bus entirely."""
    assert Settings().transit.service_window == ("06:00", "09:00")
    assert Settings().transit.window_minutes == pytest.approx(180.0)


@pytest.mark.parametrize("spelling", [("22:00", "02:00"), ("22:00", "26:00")])
def test_both_spellings_of_a_night_window_mean_the_same_thing(spelling: tuple[str, str]) -> None:
    with pytest.warns(UserWarning, match="crosses midnight"):
        transit = TransitConfig(enabled=True, service_window=spelling)
    assert transit.window_bounds_min == (22 * 60, 26 * 60)
    assert transit.window_minutes == pytest.approx(240.0)
    # the raw strings survive for the report; only the derived bounds normalise
    assert transit.service_window == spelling


def test_the_two_night_spellings_share_one_cache_entry() -> None:
    """Otherwise the same run computes — and stores — the same tables twice."""
    bbox = (-75.17, 39.94, -75.15, 39.96)
    assert transit_cache_key(bbox, "feedA", "weekday", ("22:00", "02:00")) == transit_cache_key(
        bbox, "feedA", "weekday", ("22:00", "26:00")
    )


def test_a_window_longer_than_a_service_day_is_an_error() -> None:
    """('00:00', '48:00') is legal HH:MM and silently halves every headway."""
    with pytest.raises(ValueError, match="24 h"):
        TransitConfig(enabled=True, service_window=("00:00", "48:00"))


def test_a_zero_length_window_is_an_error() -> None:
    with pytest.raises(ValueError, match="same time of day"):
        TransitConfig(enabled=True, service_window=("07:00", "07:00"))


def test_a_wide_window_warns_but_is_allowed() -> None:
    """An all-day accessibility study is legitimate (invariant #3) — but one
    average headway over peak *and* off-peak describes neither."""
    with pytest.warns(UserWarning, match="blends peak and off-peak"):
        transit = TransitConfig(enabled=True, service_window=("06:00", "12:00"))
    assert transit.window_minutes == pytest.approx(360.0)


def test_a_trip_crossing_midnight_is_placed_by_its_first_departure(tmp_path: Path) -> None:
    """23:50 -> 00:05 -> 00:20 departs at 1430, not at 5 (``.first()`` vs ``.min()``)."""
    feed = _tiny_feed(
        tmp_path,
        trips="route_id,service_id,trip_id,direction_id\nR1,WD,x1,0\n",
        stop_times=_HEADER + _times("x1", "23:50:00", "00:05:00", "00:20:00"),
    )
    patterns, _ = build_patterns(read_feed_tables(feed), _settings(service_window=("23:00", "23:59")))
    assert _row(patterns, "R1::0")["n_trips"] == pytest.approx(1.0)


def test_a_frequency_row_crossing_midnight_does_not_delete_the_line(tmp_path: Path) -> None:
    """Regression: '06:00' -> '01:00' under the default window used to clip to a
    negative overlap, clamp to zero trips, and drop the whole pattern — a silent
    zero that removed a real line from the graph (invariant #2)."""
    feed = _tiny_feed(
        tmp_path,
        trips="route_id,service_id,trip_id,direction_id\nR1,WD,f1,0\n",
        stop_times=_HEADER + _times("f1", "06:00:00", "06:03:00", "06:06:00"),
        frequencies="trip_id,start_time,end_time,headway_secs\nf1,06:00:00,01:00:00,600\n",
    )
    patterns, _ = build_patterns(read_feed_tables(feed), _settings())
    assert "R1::0" in set(patterns["pattern_id"])
    assert _row(patterns, "R1::0")["n_trips"] == pytest.approx(18.0)  # 180 min of overlap / 10 min
    assert _row(patterns, "R1::0")["headway_min"] == pytest.approx(10.0)


def test_a_night_window_matches_plain_small_hours_departures(tmp_path: Path) -> None:
    """A 01:30 departure counts in a 22:00-02:00 window, like a 25:30 one does."""
    feed = _tiny_feed(
        tmp_path,
        trips="route_id,service_id,trip_id,direction_id\nR1,WD,n1,0\n",
        stop_times=_HEADER + _times("n1", "01:30:00", "01:33:00", "01:36:00"),
    )
    with pytest.warns(UserWarning, match="crosses midnight"):
        settings = _settings(service_window=("22:00", "02:00"))
    patterns, _ = build_patterns(read_feed_tables(feed), settings)
    assert _row(patterns, "R1::0")["n_trips"] == pytest.approx(1.0)


def test_mixing_both_midnight_encodings_on_one_line_warns(tmp_path: Path) -> None:
    """Without dates the two cannot be told apart, so the doubled trip count is
    *labelled* rather than silently corrected."""
    feed = _tiny_feed(
        tmp_path,
        trips="route_id,service_id,trip_id,direction_id\nR1,WD,e1,0\nR1,WD,n1,0\n",
        stop_times=(
            _HEADER
            + _times("e1", "25:30:00", "25:33:00", "25:36:00")
            + _times("n1", "01:30:00", "01:33:00", "01:36:00")
        ),
    )
    with pytest.warns(UserWarning, match="crosses midnight"):
        settings = _settings(service_window=("22:00", "02:00"))
    with pytest.warns(UserWarning, match="n_trips_matched_by_midnight_shift"):
        patterns, _ = build_patterns(read_feed_tables(feed), settings)
    assert _row(patterns, "R1::0")["n_trips"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# CRS + boundary clipping (invariant #1)                                       #
# --------------------------------------------------------------------------- #


def test_stops_are_reprojected_to_the_working_crs() -> None:
    stops = build_stops_gdf(read_feed_tables(FIXTURE)["stops.txt"], working_crs=WORKING_CRS)
    assert str(stops.crs) == WORKING_CRS
    assert stops.geometry.x.max() > 1000  # metres, not degrees
    assert set(stops["stop_id"]) == {"S1", "S2", "S3", "S4", "M1", "M2", "X1"}


def _clipped():
    tables = read_feed_tables(FIXTURE)
    patterns, pattern_stops = build_patterns(tables, _settings())
    stops = build_stops_gdf(tables["stops.txt"], working_crs=WORKING_CRS)
    inside = gpd.GeoSeries([Point(-75.17, 39.94), Point(-75.15, 39.96)], crs="EPSG:4326").to_crs(WORKING_CRS)
    return clip_patterns_to_boundary(stops, patterns, pattern_stops, box(*inside.total_bounds))


def test_out_of_boundary_stops_are_dropped() -> None:
    stops, _, pattern_stops = _clipped()
    assert "X1" not in set(stops["stop_id"])
    assert "X1" not in set(pattern_stops["stop_id"])


def test_a_line_leaving_the_boundary_is_split_not_bridged() -> None:
    """BUS2 runs S1-S2-X1-S3-S4. Dropping X1 must not create a phantom S2->S3 hop."""
    _, patterns, pattern_stops = _clipped()
    runs = {
        pid: list(group.sort_values("seq")["stop_id"])
        for pid, group in pattern_stops.groupby("pattern_id")
        if pid.startswith("BUS2")
    }
    assert sorted(runs.values()) == [["S1", "S2"], ["S3", "S4"]]
    # both halves inherit the original's headway and route type (180 min / u1, u2;
    # u0 at 05:30 is outside the default window)
    for pid in runs:
        assert _row(patterns, pid)["headway_min"] == pytest.approx(90.0)
        assert _row(patterns, pid)["route_type"] == 3
    # and each run restarts its own sequence numbering at 0
    for _pid, group in pattern_stops[pattern_stops["pattern_id"].str.startswith("BUS2")].groupby("pattern_id"):
        assert list(group["seq"]) == list(range(len(group)))
        assert group.sort_values("seq").iloc[0]["travel_time_min"] == 0.0


# --------------------------------------------------------------------------- #
# Cache key + round trip                                                       #
# --------------------------------------------------------------------------- #


def test_cache_key_covers_every_input_that_shapes_the_tables() -> None:
    bbox = (-75.17, 39.94, -75.15, 39.96)
    base = transit_cache_key(bbox, "feedA", "weekday", ("07:00", "09:00"))
    assert base == transit_cache_key(bbox, "feedA", "weekday", ("07:00", "09:00"))
    assert base != transit_cache_key(bbox, "feedB", "weekday", ("07:00", "09:00"))
    assert base != transit_cache_key(bbox, "feedA", "saturday", ("07:00", "09:00"))
    assert base != transit_cache_key(bbox, "feedA", "weekday", ("14:00", "16:00"))
    assert base != transit_cache_key(bbox, "feedA", "weekday", ("07:00", "09:00"), "overlay1")
    assert base != transit_cache_key((-75.18, 39.94, -75.15, 39.96), "feedA", "weekday", ("07:00", "09:00"))


def test_cached_feed_round_trips(tmp_path: Path) -> None:
    stops, patterns, pattern_stops = _clipped()
    feed = LoadedGTFS(
        stops=stops,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source="gtfs",
        feed_hash="abc123",
        service_day="weekday",
        service_window=("07:00", "09:00"),
        crs=WORKING_CRS,
    )
    write_cached_feed(feed, tmp_path / "entry")
    again = read_cached_feed(tmp_path / "entry")
    assert again.source == "gtfs"
    assert again.feed_hash == "abc123"
    assert again.service_window == ("07:00", "09:00")
    assert again.n_patterns == feed.n_patterns
    assert set(again.stops["stop_id"]) == set(feed.stops["stop_id"])


def test_metadata_stamps_the_source() -> None:
    stops, patterns, pattern_stops = _clipped()
    feed = LoadedGTFS(
        stops=stops,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source="osm_assumed",
        feed_hash="x",
        service_day="weekday",
        service_window=("07:00", "09:00"),
        crs=WORKING_CRS,
    )
    # D1: an approximate run must be *labelled*, not silently indistinguishable.
    assert feed.metadata()["transit_source"] == "osm_assumed"


# --------------------------------------------------------------------------- #
# Editing overlay (D5 seam)                                                    #
# --------------------------------------------------------------------------- #


def _feed_for_overlay() -> LoadedGTFS:
    stops, patterns, pattern_stops = _clipped()
    return LoadedGTFS(
        stops=stops,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source="gtfs",
        feed_hash="base",
        service_day="weekday",
        service_window=("07:00", "09:00"),
        crs=WORKING_CRS,
    )


def _overlay_file(tmp_path: Path, records: list[dict], geometries: list) -> Path:
    path = tmp_path / "overlay.geojson"
    gpd.GeoDataFrame(records, geometry=geometries, crs=WORKING_CRS).to_file(path, driver="GeoJSON")
    return path


def test_overlay_removes_a_stop_from_the_feed(tmp_path: Path) -> None:
    feed = _feed_for_overlay()
    path = _overlay_file(tmp_path, [{"action": "remove_stop", "stop_id": "S2"}], [Point(0, 0)])
    edited = apply_overlay(feed, path, working_crs=WORKING_CRS)
    assert "S2" not in set(edited.stops["stop_id"])
    assert "S2" not in set(edited.pattern_stops["stop_id"])
    # BUS2's first run was S1-S2; losing S2 leaves one stop, so the run is dropped
    assert "BUS2::0#0" not in set(edited.patterns["pattern_id"])


def test_overlay_adds_and_moves_stops(tmp_path: Path) -> None:
    feed = _feed_for_overlay()
    target = feed.stops.set_index("stop_id").geometry.loc["S1"]
    new_point = Point(target.x + 500, target.y + 500)
    path = _overlay_file(
        tmp_path,
        [
            {"action": "add_stop", "stop_id": "NEW", "stop_name": "Proposed Station"},
            {"action": "move_stop", "stop_id": "S3"},
        ],
        [new_point, Point(target.x + 10, target.y + 10)],
    )
    edited = apply_overlay(feed, path, working_crs=WORKING_CRS)
    assert "NEW" in set(edited.stops["stop_id"])
    moved = edited.stops.set_index("stop_id").geometry.loc["S3"]
    assert moved.x == pytest.approx(target.x + 10)


def test_overlay_rejects_an_unknown_action(tmp_path: Path) -> None:
    feed = _feed_for_overlay()
    path = _overlay_file(tmp_path, [{"action": "blow_up_city", "stop_id": "S1"}], [Point(0, 0)])
    with pytest.raises(GTFSSchemaError):
        apply_overlay(feed, path, working_crs=WORKING_CRS)


def test_overlay_hash_is_content_addressed(tmp_path: Path) -> None:
    a = tmp_path / "a.geojson"
    b = tmp_path / "b.geojson"
    a.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    b.write_text('{"type": "FeatureCollection", "features": [1]}', encoding="utf-8")
    assert overlay_hash(None) is None
    assert overlay_hash(a) != overlay_hash(b)
    with pytest.raises(GTFSError):
        overlay_hash(tmp_path / "missing.geojson")
