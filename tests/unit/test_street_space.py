"""Canonical street-space (corridor) resolution — P7.2.

Hand-built geometry, no OSM: a roadway block, the sidewalk running beside it, a
crossing cutting across it, and a park path far away. The whole fix rests on
those four cases resolving the way a person looking at a map would say they
should.
"""

from __future__ import annotations

import networkx as nx
import pytest

from segx.config import Settings
from segx.network.registry import GraphRegistry
from segx.network.street_space import edges_as_gdf, resolve_street_space

CRS = "EPSG:32618"


def _settings(**street_space) -> Settings:
    return Settings.model_validate({"network": {"street_space": street_space}}) if street_space else Settings()


def _graph(nodes: dict[int, tuple[float, float]], edges: list[tuple[int, int, dict]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = CRS
    for nid, (x, y) in nodes.items():
        graph.add_node(nid, x=float(x), y=float(y))
    for u, v, data in edges:
        length = ((nodes[v][0] - nodes[u][0]) ** 2 + (nodes[v][1] - nodes[u][1]) ** 2) ** 0.5
        graph.add_edge(u, v, key=0, length=length, **data)
    return graph


def _drive_graph() -> nx.MultiDiGraph:
    """One east-west roadway block, 100 m long, from (0,0) to (100,0)."""
    return _graph({1: (0.0, 0.0), 2: (100.0, 0.0)}, [(1, 2, {"highway": "residential"})])


def _walk_graph() -> nx.MultiDiGraph:
    """A sidewalk beside the roadway, a crossing over it, and a distant park path.

    * 10 → 11: sidewalk, parallel, offset 5 m north — must join the roadway.
    * 11 → 12: crossing, perpendicular — must not.
    * 20 → 21: park path 100 m north, parallel but far — must not.
    """
    nodes = {
        10: (10.0, 5.0),
        11: (90.0, 5.0),
        12: (90.0, -5.0),
        20: (10.0, 100.0),
        21: (90.0, 100.0),
    }
    return _graph(
        nodes,
        [
            (10, 11, {"highway": "footway"}),
            (11, 12, {"highway": "footway"}),
            (20, 21, {"highway": "footway"}),
        ],
    )


def _resolve(settings: Settings | None = None, walk: nx.MultiDiGraph | None = None):
    graphs = GraphRegistry()
    graphs.register("drive", _drive_graph())
    graphs.register("walk", walk if walk is not None else _walk_graph())
    return resolve_street_space(graphs, settings or _settings(), working_crs=CRS)


# --------------------------------------------------------------------------- #
# The four cases                                                               #
# --------------------------------------------------------------------------- #


def test_a_parallel_sidewalk_joins_the_roadway_corridor() -> None:
    """The whole point: the car is on way A and the pedestrian on way B, five
    metres apart, and they share a street. No id remap can see that."""
    space = _resolve()
    roadway = space.get((1, 2, 0))
    sidewalk = space.get((10, 11, 0))
    assert roadway is not None and sidewalk is not None
    assert sidewalk.corridor_id == roadway.corridor_id
    assert sidewalk.matched is True


def test_a_perpendicular_crossing_does_not_join_the_street_it_crosses() -> None:
    """Its midpoint is well inside the tolerance; only the bearing test keeps it
    out. Without that test every corner would pull its cross street in."""
    space = _resolve()
    crossing = space.get((11, 12, 0))
    assert crossing.corridor_id != space.get((1, 2, 0)).corridor_id
    assert crossing.matched is False
    assert crossing.corridor_id.startswith("own:walk:")


def test_a_distant_parallel_path_gets_its_own_corridor() -> None:
    """A park path is a real street space, not a failure — invariant #2."""
    space = _resolve()
    park = space.get((20, 21, 0))
    assert park.matched is False
    assert park.corridor_id != space.get((1, 2, 0)).corridor_id


def test_the_anchor_is_its_own_corridor_and_keeps_its_length() -> None:
    space = _resolve()
    roadway = space.get((1, 2, 0))
    assert roadway.matched is True
    assert roadway.length_m == pytest.approx(100.0)
    assert space.get((10, 11, 0)).length_m == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# Tuning and exclusions                                                        #
# --------------------------------------------------------------------------- #


def test_tolerance_below_the_offset_rejects_the_sidewalk() -> None:
    space = _resolve(_settings(tolerance_m=2.0))
    assert space.get((10, 11, 0)).matched is False


def test_bearing_tolerance_of_ninety_degrees_lets_the_crossing_in() -> None:
    """Not a recommendation — a guard that the bearing test is what does the work,
    so a mis-tuned `max_bearing_deg` fails loudly in the audit rather than quietly
    inflating co-presence."""
    space = _resolve(_settings(max_bearing_deg=90.0))
    assert space.get((11, 12, 0)).corridor_id == space.get((1, 2, 0)).corridor_id


@pytest.mark.parametrize(
    ("tags", "settings_kwargs"),
    [
        ({"highway": "corridor"}, {}),
        ({"highway": "steps"}, {}),
        ({"highway": "footway", "tunnel": "yes"}, {}),
    ],
)
def test_indoor_and_underground_edges_never_join_a_street(tags, settings_kwargs) -> None:
    """An underground concourse runs directly beneath the avenue and is not
    co-present with it. Philadelphia's walk graph really does contain one."""
    walk = _graph({10: (10.0, 5.0), 11: (90.0, 5.0)}, [(10, 11, tags)])
    space = _resolve(_settings(**settings_kwargs), walk=walk)
    assert space.get((10, 11, 0)).matched is False


def test_rail_edges_never_join_a_street_corridor() -> None:
    """P7 D4, restated in corridor terms: a metro tunnel under an avenue must not
    be pooled with the traffic on it. The exclusion keys on `segx_layer`, never on
    the negative-id convention of synthetic transit nodes."""
    walk = _graph(
        {10: (10.0, 5.0), 11: (90.0, 5.0)},
        [(10, 11, {"highway": "footway", "segx_layer": "rail_line"})],
    )
    space = _resolve(walk=walk)
    assert space.get((10, 11, 0)).matched is False


def test_excluding_a_highway_value_is_configurable() -> None:
    """Invariant #3 — the exclusion list is config, not a literal in the code."""
    walk = _graph({10: (10.0, 5.0), 11: (90.0, 5.0)}, [(10, 11, {"highway": "service"})])
    assert _resolve(walk=walk).get((10, 11, 0)).matched is True
    space = _resolve(_settings(exclude_highway=["service"]), walk=walk)
    assert space.get((10, 11, 0)).matched is False


# --------------------------------------------------------------------------- #
# Contract                                                                     #
# --------------------------------------------------------------------------- #


def test_anchor_priority_falls_back_to_a_mode_that_is_present() -> None:
    """A walk-only run has no roadway layer; the corridors then come from `walk`
    and every edge still gets a defined corridor."""
    graphs = GraphRegistry()
    graphs.register("walk", _walk_graph())
    space = resolve_street_space(graphs, _settings(), working_crs=CRS)
    assert space.stats["anchor_mode"] == "walk"
    assert all(info.corridor_id for info in space.corridors.values())


def test_resolution_is_deterministic() -> None:
    a = _resolve().corridors
    b = _resolve().corridors
    assert a == b


def test_a_reciprocal_pair_is_one_segment() -> None:
    """`canonical_edge` folds direction, so a two-way street is one corridor
    member, not two — otherwise the length-weighted pool would double-count it."""
    drive = _drive_graph()
    drive.add_edge(2, 1, key=0, length=100.0, highway="residential")
    graphs = GraphRegistry()
    graphs.register("drive", drive)
    graphs.register("walk", _walk_graph())
    space = resolve_street_space(graphs, _settings(), working_crs=CRS)
    assert space.stats["n_anchors"] == 1


def test_stats_report_the_pair_that_matters() -> None:
    space = _resolve()
    assert space.stats["corridors_per_mode_pair"]["drive|walk"] == 1
    assert space.stats["n_corridors_own"] == 2  # the crossing and the park path
    assert space.params["tolerance_m"] == pytest.approx(20.0)


def test_edges_as_gdf_falls_back_to_the_node_chord() -> None:
    """OSMnx only sets `geometry` on edges simplified from a multi-node way; a
    straight block has none and *is* its chord."""
    gdf = edges_as_gdf(_drive_graph(), "drive", CRS)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.length == pytest.approx(100.0)
    assert gdf.crs.to_string() == CRS


def test_bearings_are_the_chord_orientation_folded_to_half_a_turn() -> None:
    """One value per geometry, in order, from its first and last vertex only — a
    curved street's interior vertices do not change which way it runs, and a
    street mapped the other way round has the same bearing."""
    import math

    from shapely.geometry import LineString

    from segx.network.street_space import _bearings_deg

    geoms = [
        LineString([(0, 0), (10, 0)]),
        LineString([(10, 0), (0, 0)]),  # reversed: same street
        LineString([(0, 0), (3, 9), (0, 10)]),  # curved: chord runs straight up
        LineString([(1, 1), (4, 5)]),
    ]
    got = _bearings_deg(geoms)
    assert got[0] == 0.0 and got[1] == 0.0
    assert got[2] == pytest.approx(90.0)
    assert got[3] == math.degrees(math.atan2(4, 3)) % 180.0
