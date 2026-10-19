"""OSM fallback tests (P7) — the source used when no GTFS feed is supplied.

Two things must hold, and they matter more than the parsing details:

1. the fallback produces the **same shape** as the GTFS reader, so
   :func:`segx.network.transit.build_transit_graph` has one input contract;
2. every result is stamped ``transit_source = "osm_assumed"``, because OSM carries
   **no timetables** and the headways were therefore *invented* from config. A
   headway we invented and a headway we measured look identical in a GeoPackage —
   labelling is the only thing that keeps them apart (D1).

The Overpass call itself is integration territory; the payload *parser* is pure
and is exercised here against a canned response, which is why the fallback talks
to Overpass directly rather than through OSMnx (``features_from_polygon`` drops
relation member roles and order — exactly what a transit graph needs).
"""

from __future__ import annotations

import pytest

from segx.config import NetworkConfig, Settings, TransitConfig
from segx.network.transit import build_transit_graph
from segx.network.transit_osm import (
    OSM_ROUTE_TYPES,
    assumed_headway_min,
    build_osm_feed,
    overpass_query,
    parse_overpass_relations,
)
from segx.validation.synthetic_city import EDGE_M, SYNTHETIC_CRS, build_grid_graph, node_colrow

# The grid is a metric fixture, so the OSM feed is built with the working CRS as
# its own "geographic" CRS — that keeps the stop coordinates on the grid nodes.
_GRID_AS_GEOGRAPHIC = SYNTHETIC_CRS


def _settings(**transit_kwargs) -> Settings:
    return Settings(
        network=NetworkConfig(modes=["walk", "transit"]),
        transit=TransitConfig(enabled=True, source="osm", **transit_kwargs),
    )


def _node(osm_id: int, grid_node: int) -> dict:
    col, row = node_colrow(grid_node)
    return {"type": "node", "id": osm_id, "lon": col * EDGE_M, "lat": row * EDGE_M, "tags": {"name": f"stop{osm_id}"}}


def _payload() -> dict:
    return {
        "elements": [
            _node(101, 0),
            _node(102, 1),
            _node(103, 2),
            _node(104, 3),
            _node(201, 12),
            _node(202, 15),
            {
                "type": "relation",
                "id": 900,
                "tags": {"type": "route", "route": "bus", "name": "Bottom Row Bus"},
                "members": [
                    {"type": "node", "ref": 101, "role": "stop"},
                    {"type": "node", "ref": 102, "role": "stop"},
                    {"type": "way", "ref": 555, "role": ""},
                    {"type": "node", "ref": 103, "role": "stop"},
                    {"type": "node", "ref": 104, "role": "stop"},
                ],
            },
            {
                "type": "relation",
                "id": 901,
                "tags": {"type": "route", "route": "subway", "name": "Top Row Metro"},
                "members": [
                    {"type": "node", "ref": 201, "role": "platform"},
                    {"type": "node", "ref": 202, "role": "platform"},
                ],
            },
        ]
    }


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #


def test_parses_ordered_stop_members_and_ignores_ways() -> None:
    records = {r["route_id"]: r for r in parse_overpass_relations(_payload())}
    assert set(records) == {"osm900", "osm901"}
    bus = records["osm900"]
    assert bus["route"] == "bus"
    assert [stop_id for stop_id, *_ in bus["stops"]] == ["101", "102", "103", "104"]
    assert records["osm901"]["route"] == "subway"


def test_platform_members_are_used_only_when_there_are_no_stop_members() -> None:
    """Mixing the two roles would double every stop of the line."""
    payload = _payload()
    relation = payload["elements"][-2]  # the bus
    relation["members"].append({"type": "node", "ref": 101, "role": "platform"})
    bus = next(r for r in parse_overpass_relations(payload) if r["route_id"] == "osm900")
    assert [stop_id for stop_id, *_ in bus["stops"]] == ["101", "102", "103", "104"]


def test_a_repeated_terminus_on_a_loop_route_is_not_duplicated() -> None:
    payload = _payload()
    payload["elements"][-2]["members"].append({"type": "node", "ref": 101, "role": "stop"})
    bus = next(r for r in parse_overpass_relations(payload) if r["route_id"] == "osm900")
    assert [stop_id for stop_id, *_ in bus["stops"]] == ["101", "102", "103", "104"]


def test_relations_with_fewer_than_two_stops_are_dropped() -> None:
    payload = {
        "elements": [
            _node(101, 0),
            {
                "type": "relation",
                "id": 902,
                "tags": {"type": "route", "route": "bus"},
                "members": [{"type": "node", "ref": 101, "role": "stop"}],
            },
        ]
    }
    assert parse_overpass_relations(payload) == []


def test_non_transit_route_relations_are_ignored() -> None:
    payload = _payload()
    payload["elements"].append(
        {
            "type": "relation",
            "id": 903,
            "tags": {"type": "route", "route": "hiking"},
            "members": [{"type": "node", "ref": 101, "role": "stop"}, {"type": "node", "ref": 102, "role": "stop"}],
        }
    )
    assert {r["route_id"] for r in parse_overpass_relations(payload)} == {"osm900", "osm901"}


def test_query_pins_the_snapshot_date_when_configured() -> None:
    bbox = (-75.17, 39.94, -75.15, 39.96)
    assert "[date:" not in overpass_query(bbox, None)
    pinned = overpass_query(bbox, "2026-05-01")
    assert '[date:"2026-05-01T00:00:00Z"]' in pinned
    assert "39.94,-75.17,39.96,-75.15" in pinned  # Overpass wants (S,W,N,E)


# --------------------------------------------------------------------------- #
# Feed construction                                                            #
# --------------------------------------------------------------------------- #


def test_osm_feed_matches_the_gtfs_shape_and_is_labelled_assumed() -> None:
    settings = _settings()
    feed = build_osm_feed(
        parse_overpass_relations(_payload()),
        settings,
        working_crs=SYNTHETIC_CRS,
        geographic_crs=_GRID_AS_GEOGRAPHIC,
    )
    assert feed.source == "osm_assumed"
    assert feed.metadata()["transit_source"] == "osm_assumed"
    assert list(feed.patterns.columns) == [
        "pattern_id",
        "route_id",
        "direction_id",
        "route_type",
        "route_name",
        "headway_min",
        "n_trips",
    ]
    assert list(feed.pattern_stops.columns) == ["pattern_id", "seq", "stop_id", "travel_time_min"]
    assert set(feed.patterns["route_type"]) == {OSM_ROUTE_TYPES["bus"], OSM_ROUTE_TYPES["subway"]}


def test_headways_come_from_config_not_from_data() -> None:
    """Invariant #3: nothing about transit is a literal in engine code."""
    settings = _settings()
    settings.transit.osm_assumed_headway_min = {"bus": 12.0, "subway": 4.0}
    feed = build_osm_feed(
        parse_overpass_relations(_payload()),
        settings,
        working_crs=SYNTHETIC_CRS,
        geographic_crs=_GRID_AS_GEOGRAPHIC,
    )
    by_route = feed.patterns.set_index("route_id")["headway_min"]
    assert by_route.loc["osm900"] == pytest.approx(12.0)
    assert by_route.loc["osm901"] == pytest.approx(4.0)
    assert assumed_headway_min(3, settings) == pytest.approx(12.0)


def test_trip_count_and_headway_stay_consistent_with_the_window() -> None:
    """``n_trips × headway_min == window_minutes`` is the identity the whole
    frequency model rests on; the OSM branch invents the headway, so the trip
    count has to be *derived* from the window rather than from a literal."""
    settings = _settings(service_window=("06:00", "09:00"))
    feed = build_osm_feed(
        parse_overpass_relations(_payload()),
        settings,
        working_crs=SYNTHETIC_CRS,
        geographic_crs=_GRID_AS_GEOGRAPHIC,
    )
    window_minutes = settings.transit.window_minutes
    assert window_minutes == pytest.approx(180.0)
    products = feed.patterns["n_trips"] * feed.patterns["headway_min"]
    assert products.round(9).eq(round(window_minutes, 9)).all()


def test_inter_stop_times_are_estimated_from_distance() -> None:
    """OSM has no schedule, so a hop is its straight-line distance at an assumed
    speed — modest on purpose, so a bus never looks faster than it could be."""
    feed = build_osm_feed(
        parse_overpass_relations(_payload()),
        _settings(),
        working_crs=SYNTHETIC_CRS,
        geographic_crs=_GRID_AS_GEOGRAPHIC,
    )
    hops = feed.pattern_stops[feed.pattern_stops["pattern_id"] == "osm900::0"].set_index("seq")["travel_time_min"]
    assert hops[0] == 0.0
    # 100 m at the assumed 15 km/h bus speed = 0.4 min
    assert hops[1] == pytest.approx(EDGE_M * 60.0 / (15.0 * 1000.0))
    assert (feed.pattern_stops["travel_time_min"] >= 0).all()


def test_the_osm_feed_builds_a_working_transit_graph() -> None:
    """The point of the shared shape: the graph builder needs no OSM-specific path."""
    settings = _settings()
    feed = build_osm_feed(
        parse_overpass_relations(_payload()),
        settings,
        working_crs=SYNTHETIC_CRS,
        geographic_crs=_GRID_AS_GEOGRAPHIC,
    )
    grid = build_grid_graph()
    graph = build_transit_graph(feed, grid, grid, settings, working_crs=SYNTHETIC_CRS)
    assert graph.graph["transit"]["transit_source"] == "osm_assumed"
    assert graph.graph["transit"]["n_rides"] == 4  # 3 bus hops + 1 metro hop
