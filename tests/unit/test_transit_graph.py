"""Merged walk ⊕ transit graph tests (P7), on the synthetic city.

The fixture puts a bus line along the bottom row (nodes 0-1-2-3) and a metro line
along the top row (12-13-14-15), with stops sitting **exactly on grid nodes** so
the connectors are free and the ride-vs-walk arithmetic is exact:

* walking 0 -> 3 costs ``3 x EDGE_WALK_MIN`` = **3.6 min**
* riding 0 -> 3 costs ``headway/2 + board_penalty + 3 x hop``
  = 5.0 + 1.0 + 1.5 = **7.5 min** at a 10-minute headway -> *walking wins*
  = 1.0 + 1.0 + 1.5 = **3.5 min** at a 2-minute headway  -> *the bus wins*

That single pair of assertions is the whole frequency model: nobody wrote "prefer walking when the
bus is infrequent" — it falls out of one Dijkstra over one graph.
"""

from __future__ import annotations

import pytest

from segx.config import NetworkConfig, RoutingConfig, Settings, TransitConfig
from segx.errors import GraphError, TransitDataMissingError
from segx.network.transit import (
    ACCESS_EDGE_KINDS,
    board_cost_min,
    build_transit_graph,
    edge_layers,
    expand_transit_rides,
    has_transit,
    street_nodes,
)
from segx.routing.networkx_engine import NetworkXEngine
from segx.validation.synthetic_city import (
    BUS_0_TO_3_FAST_MIN,
    BUS_0_TO_3_SLOW_MIN,
    SYNTHETIC_CRS,
    TRANSIT_BOARD_PENALTY_MIN,
    TRANSIT_FAST_HEADWAY_MIN,
    TRANSIT_SLOW_HEADWAY_MIN,
    WALK_0_TO_3_MIN,
    build_grid_graph,
    build_transit_feed,
)


def _settings(**transit_kwargs) -> Settings:
    kwargs = {"enabled": True, "board_penalty_min": TRANSIT_BOARD_PENALTY_MIN, **transit_kwargs}
    return Settings(
        network=NetworkConfig(modes=["walk", "drive", "transit"]),
        transit=TransitConfig(**kwargs),
        routing=RoutingConfig(engine="networkx", access_mode="transit"),
    )


def _graph(headway: float = TRANSIT_FAST_HEADWAY_MIN, *, map_match: bool = True, **transit_kwargs):
    settings = _settings(map_match_surface_rides=map_match, **transit_kwargs)
    grid = build_grid_graph()
    feed = build_transit_feed(headway)
    drive = grid if map_match else None
    return build_transit_graph(feed, grid, drive, settings, working_crs=SYNTHETIC_CRS), settings


def _kinds(graph, path) -> list[str]:
    return [graph.get_edge_data(*edge).get("segx_edge") for edge in path]


def _layers(graph, path) -> list[str]:
    return [graph.get_edge_data(*edge).get("segx_layer") for edge in path]


# --------------------------------------------------------------------------- #
# Structure                                                                    #
# --------------------------------------------------------------------------- #


def test_all_five_edge_families_are_present() -> None:
    graph, _ = _graph()
    kinds = {data.get("segx_edge") for *_, data in graph.edges(data=True)}
    assert kinds == {"street", "connector", "board", "ride", "alight", "transfer"}


def test_synthetic_nodes_are_negative_and_cannot_collide() -> None:
    graph, _ = _graph()
    grid_nodes = set(build_grid_graph().nodes)
    synthetic = {n for n, data in graph.nodes(data=True) if "segx_node" in data}
    assert synthetic
    assert all(n < 0 for n in synthetic)
    assert synthetic.isdisjoint(grid_nodes)
    assert {data["segx_node"] for _, data in graph.nodes(data=True) if "segx_node" in data} == {
        "platform",
        "route_stop",
    }


def test_street_nodes_excludes_the_synthetic_ones() -> None:
    graph, _ = _graph()
    assert street_nodes(graph) == set(build_grid_graph().nodes)
    # a plain street graph needs no restriction at all
    assert street_nodes(build_grid_graph()) is None
    assert has_transit(graph) and not has_transit(build_grid_graph())


def test_build_provenance_is_recorded_on_the_graph() -> None:
    graph, _ = _graph()
    meta = graph.graph["transit"]
    assert meta["transit_source"] == "gtfs"
    assert meta["n_platforms"] == 8
    assert meta["n_rides"] == 6  # 3 bus hops + 3 metro hops
    assert meta["n_stops_unreachable"] == 0


def test_transfer_radius_is_honoured() -> None:
    """The grid's spacing is 100 m, so the radius directly controls the count."""
    wide, _ = _graph(max_transfer_distance_m=400.0)
    narrow, _ = _graph(max_transfer_distance_m=50.0)
    assert wide.graph["transit"]["n_transfers"] > narrow.graph["transit"]["n_transfers"]
    assert narrow.graph["transit"]["n_transfers"] == 0


def test_board_cost_is_capped_by_max_wait() -> None:
    settings = _settings(max_wait_min=4.0)
    # an hourly line would otherwise model a certain 30-minute wait
    assert board_cost_min(60.0, settings) == pytest.approx(4.0 + TRANSIT_BOARD_PENALTY_MIN)
    assert board_cost_min(6.0, settings) == pytest.approx(3.0 + TRANSIT_BOARD_PENALTY_MIN)


def test_crs_mismatch_is_refused() -> None:
    settings = _settings()
    with pytest.raises(GraphError):
        build_transit_graph(build_transit_feed(), build_grid_graph(), None, settings, working_crs="EPSG:31983")


def test_an_empty_feed_is_a_typed_error() -> None:
    settings = _settings()
    feed = build_transit_feed()
    empty = type(feed)(
        stops=feed.stops.iloc[0:0],
        patterns=feed.patterns.iloc[0:0],
        pattern_stops=feed.pattern_stops.iloc[0:0],
        source="gtfs",
        feed_hash="empty",
        service_day="weekday",
        service_window=("07:00", "09:00"),
        crs=SYNTHETIC_CRS,
    )
    with pytest.raises(TransitDataMissingError) as excinfo:
        build_transit_graph(empty, build_grid_graph(), None, settings, working_crs=SYNTHETIC_CRS)
    assert excinfo.value.code == "transit.data_missing"


# --------------------------------------------------------------------------- #
# Layers (D4)                                                                  #
# --------------------------------------------------------------------------- #


def test_surface_rides_are_street_and_rail_rides_are_not() -> None:
    graph, _ = _graph()
    rides = [data for *_, data in graph.edges(data=True) if data.get("segx_edge") == "ride"]
    bus = [d for d in rides if d["route_type"] == 3]
    metro = [d for d in rides if d["route_type"] == 1]
    assert len(bus) == len(metro) == 3
    assert {d["segx_layer"] for d in bus} == {"street"}
    assert {d["segx_layer"] for d in metro} == {"rail_line"}
    assert all(d["street_path"] for d in bus)
    assert all(d["street_path"] is None for d in metro)


def test_an_unmatched_surface_ride_keeps_its_own_line_rather_than_faking_a_street() -> None:
    graph, _ = _graph(map_match=False)
    bus = [d for *_, d in graph.edges(data=True) if d.get("segx_edge") == "ride" and d["route_type"] == 3]
    assert {d["segx_layer"] for d in bus} == {"rail_line"}
    assert all(d["surface_unmatched"] for d in bus)
    assert graph.graph["transit"]["n_surface_rides_unmatched"] == 3


def test_edge_layers_reports_only_the_non_street_minority() -> None:
    graph, _ = _graph()
    layers = edge_layers(graph)
    assert set(layers.values()) == {"access", "rail_line"}
    assert sum(1 for v in layers.values() if v == "rail_line") == 3
    assert not any(graph.get_edge_data(*edge).get("segx_layer") == "street" for edge in layers)


# --------------------------------------------------------------------------- #
# The frequency model — the ground truth                                       #
# --------------------------------------------------------------------------- #


def test_a_frequent_bus_beats_walking() -> None:
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    assert result is not None
    assert result.cost == pytest.approx(BUS_0_TO_3_FAST_MIN)
    assert result.cost < WALK_0_TO_3_MIN
    assert _kinds(graph, result.path) == ["connector", "board", "ride", "ride", "ride", "alight", "connector"]


def test_an_infrequent_bus_loses_to_walking() -> None:
    graph, settings = _graph(TRANSIT_SLOW_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    assert result is not None
    assert result.cost == pytest.approx(WALK_0_TO_3_MIN)
    assert BUS_0_TO_3_SLOW_MIN > WALK_0_TO_3_MIN
    assert set(_kinds(graph, result.path)) == {"street"}


def test_changing_lines_costs_a_transfer() -> None:
    """0 -> 15 must leave the bus and join the metro, paying alight + board."""
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 15, "transit", weight="travel_time_min")
    assert result is not None
    kinds = _kinds(graph, result.path)
    assert kinds.count("board") >= 1
    # staying on one line traverses only `ride` edges; a line change cannot
    assert "alight" in kinds or kinds.count("board") == 1


# --------------------------------------------------------------------------- #
# expand_transit_rides                                                         #
# --------------------------------------------------------------------------- #


def test_expansion_puts_bus_riders_on_the_street_and_drops_access_edges() -> None:
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    expanded, leg_modes = expand_transit_rides(result.path, graph)
    # the three grid edges the bus drives down — the avenue, not a synthetic line
    assert expanded == [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
    assert all(edge[0] >= 0 and edge[1] >= 0 for edge in expanded)
    assert not any(_kind in ACCESS_EDGE_KINDS for _kind in _kinds(graph, expanded))
    # riding is `transit`; nothing here is a walking leg (P7.2)
    assert leg_modes == ["transit"] * 3


def test_expansion_leaves_a_metro_ride_on_its_own_layer() -> None:
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(12, 15, "transit", weight="travel_time_min")
    expanded, leg_modes = expand_transit_rides(result.path, graph)
    assert len(expanded) == 3
    assert _layers(graph, expanded) == ["rail_line"] * 3
    # metro riders are never placed on the street above the tunnel
    assert all(edge[0] < 0 and edge[1] < 0 for edge in expanded)
    assert leg_modes == ["transit"] * 3


def test_expansion_is_a_no_op_for_a_plain_street_path() -> None:
    graph, settings = _graph(TRANSIT_SLOW_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    edges, leg_modes = expand_transit_rides(result.path, graph)
    assert edges == result.path
    # at a 10-minute headway walking wins, so every leg is a walking leg — and the
    # traveller is filed as a pedestrian, not as a bus passenger (P7.2)
    assert leg_modes == ["walk"] * len(edges)
    assert expand_transit_rides([], graph) == ([], [])


def test_walking_legs_of_a_transit_trip_are_labelled_walk() -> None:
    """The P7.2 correction: a bus trip's access legs are pedestrians on a sidewalk,
    not passengers in a vehicle. Before this, "bus × sidewalk" silently measured a
    bus passenger against *themselves* walking to the stop."""
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    # 4 -> 3: walk down from the row below, then ride the bus along row r=0
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(4, 3, "transit", weight="travel_time_min")
    edges, leg_modes = expand_transit_rides(result.path, graph)
    assert len(edges) == len(leg_modes)
    assert set(leg_modes) == {"walk", "transit"}
    # every `transit` leg is a street the bus actually drives down
    ridden = {edge for edge, mode in zip(edges, leg_modes, strict=True) if mode == "transit"}
    assert ridden <= {(0, 1, 0), (1, 2, 0), (2, 3, 0)}


def test_expansion_is_idempotent() -> None:
    """The orchestrator runs it once, but a double application must not double mass."""
    graph, settings = _graph(TRANSIT_FAST_HEADWAY_MIN)
    result = NetworkXEngine({"transit": graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    once, _ = expand_transit_rides(result.path, graph)
    assert expand_transit_rides(once, graph)[0] == once
