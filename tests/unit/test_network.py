"""P1: network builder (projection + travel-time weights), cache key plumbing,
and point snapping. Live OSMnx fetches are integration territory (P3+)."""

from __future__ import annotations

import math

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from segx.config import NetworkConfig, Settings
from segx.network.builder import (
    attach_travel_times,
    build_mode_graph,
    parse_maxspeed_kmh,
    project_graph,
)
from segx.network.cache import cache_entry, cache_key, read_graph, read_manifest, write_graph
from segx.network.snapping import snap_points

# --------------------------------------------------------------------------- #
# parse_maxspeed_kmh                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (50, 50.0),
        (50.0, 50.0),
        ("50", 50.0),
        ("50 km/h", 50.0),
        ("50km/h", 50.0),
        ("30 mph", 30.0 * 1.609344),
        ("not a speed", None),
        (["50", "40"], 40.0),  # take the minimum
        (["x", "y"], None),
        ([], None),
    ],
)
def test_parse_maxspeed_kmh(value, expected) -> None:
    got = parse_maxspeed_kmh(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# project_graph + attach_travel_times                                          #
# --------------------------------------------------------------------------- #


def _wgs84_graph() -> nx.MultiDiGraph:
    """A 2-node, 1-edge MultiDiGraph near Porto Alegre, OSMnx-shaped."""
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-51.20, y=-30.00)
    g.add_node(2, x=-51.19, y=-30.00)
    # length is in metres (OSMnx computes great-circle on fetch)
    line = LineString([(-51.20, -30.00), (-51.19, -30.00)])
    g.add_edge(1, 2, length=965.0, maxspeed="50", geometry=line)
    return g


def test_project_graph_to_utm() -> None:
    g = _wgs84_graph()
    p = project_graph(g, "EPSG:32722")
    assert p.graph["crs"] == "EPSG:32722"
    # Porto Alegre easting ~470_000 m
    assert 400_000 < p.nodes[1]["x"] < 600_000
    assert 6_000_000 < p.nodes[1]["y"] < 7_500_000
    # the original graph is untouched (project_graph copies)
    assert g.nodes[1]["x"] == pytest.approx(-51.20)


def test_project_graph_no_op_when_same_crs() -> None:
    g = _wgs84_graph()
    p = project_graph(g, "EPSG:4326")
    assert p.nodes[1]["x"] == pytest.approx(-51.20)


def test_project_graph_requires_crs_metadata() -> None:
    g = nx.MultiDiGraph()
    g.add_node(1, x=0, y=0)
    with pytest.raises(ValueError, match="CRS"):
        project_graph(g, "EPSG:32722")


def test_attach_travel_times_walk() -> None:
    g = _wgs84_graph()  # length 965 m
    attach_travel_times(g, "walk", Settings())  # walk @ 5 km/h
    # 965 m @ 5 km/h = 965 / (5000/60) min = 11.58 min
    edge_data = next(iter(g[1][2].values()))
    assert edge_data["travel_time_min"] == pytest.approx(965.0 * 60.0 / 5000.0)


def test_attach_travel_times_drive_uses_posted_speed() -> None:
    g = _wgs84_graph()  # maxspeed = "50"
    attach_travel_times(g, "drive", Settings())
    edge_data = next(iter(g[1][2].values()))
    # 965 m @ 50 km/h = 1.158 min
    assert edge_data["travel_time_min"] == pytest.approx(965.0 * 60.0 / 50000.0)
    assert edge_data["effective_speed_kmh"] == 50.0


def test_attach_travel_times_drive_falls_back_when_no_maxspeed() -> None:
    g = _wgs84_graph()
    # strip maxspeed from the edge
    for *_, data in g.edges(data=True):
        data.pop("maxspeed", None)
    s = Settings(network=NetworkConfig(drive_speed_fallback_kmh=40.0))
    attach_travel_times(g, "drive", s)
    edge_data = next(iter(g[1][2].values()))
    assert edge_data["effective_speed_kmh"] == 40.0
    assert edge_data["travel_time_min"] == pytest.approx(965.0 * 60.0 / 40_000.0)


def _reciprocal_metric_graph() -> nx.MultiDiGraph:
    """3 nodes in a metric CRS; edge (1,2) present in both directions + (2,3)."""
    g = nx.MultiDiGraph(crs="EPSG:32722")
    for n, x in [(1, 0.0), (2, 100.0), (3, 200.0)]:
        g.add_node(n, x=x, y=0.0)
    g.add_edge(1, 2, length=100.0)
    g.add_edge(2, 1, length=100.0)  # reciprocal of (1,2)
    g.add_edge(2, 3, length=100.0)
    return g


def test_build_mode_graph_undirected_collapses_reciprocal_edges() -> None:
    s = Settings(network=NetworkConfig(undirected=True))
    g = build_mode_graph(_reciprocal_metric_graph(), "walk", s, working_crs="EPSG:32722")
    assert not g.is_directed()
    assert g.number_of_edges() == 2  # (1,2) and (2,3); the reciprocal (2,1) merged into (1,2)
    for *_, data in g.edges(data=True):  # weights still land on the merged segments
        assert data["travel_time_min"] == pytest.approx(100.0 * 60.0 / 5000.0)


def test_build_mode_graph_directed_by_default_keeps_both_directions() -> None:
    s = Settings(network=NetworkConfig(undirected=False))
    g = build_mode_graph(_reciprocal_metric_graph(), "walk", s, working_crs="EPSG:32722")
    assert g.is_directed()
    assert g.number_of_edges() == 3  # both directions of (1,2) preserved


def test_attach_travel_times_rejects_missing_length() -> None:
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=0, y=0)
    g.add_node(2, x=0.001, y=0)
    g.add_edge(1, 2)  # no length attribute
    with pytest.raises(ValueError, match="length"):
        attach_travel_times(g, "walk", Settings())


def test_attach_travel_times_rejects_unsupported_mode() -> None:
    g = _wgs84_graph()
    with pytest.raises(ValueError, match="teleport"):
        attach_travel_times(g, "teleport", Settings())


def test_attach_travel_times_walks_transit_street_edges_and_leaves_transit_edges() -> None:
    """P7: on a merged graph only the street edges are (re)weighted at walk speed;
    board/ride/etc. already carry times derived from headways and schedules."""
    g = _wgs84_graph()
    for *_, data in g.edges(data=True):
        data["segx_layer"] = "street"
    g.add_node(-1, x=0, y=0)
    g.add_edge(1, -1, length=0.0, segx_layer="access", segx_edge="board", travel_time_min=6.0)
    attach_travel_times(g, "transit", Settings())
    assert g[1][-1][0]["travel_time_min"] == 6.0  # untouched
    street = next(iter(g[1][2].values()))
    assert street["travel_time_min"] > 0


def test_build_mode_graph_projects_and_weights() -> None:
    raw = _wgs84_graph()
    built = build_mode_graph(raw, "walk", Settings(), working_crs="EPSG:32722")
    assert built.graph["crs"] == "EPSG:32722"
    assert "travel_time_min" in next(iter(built[1][2].values()))


# --------------------------------------------------------------------------- #
# cache plumbing                                                               #
# --------------------------------------------------------------------------- #


def test_cache_key_is_deterministic() -> None:
    bbox = (-51.25, -30.05, -51.15, -29.95)
    k1 = cache_key(bbox, "2026-05-01", "walk")
    k2 = cache_key(bbox, "2026-05-01", "walk")
    assert k1 == k2
    # any input change → different key
    assert cache_key(bbox, "2026-05-02", "walk") != k1
    assert cache_key(bbox, "2026-05-01", "drive") != k1
    assert cache_key((-51.25, -30.05, -51.15, -29.94), "2026-05-01", "walk") != k1
    # tiny floating-point drift past the rounding decimals is absorbed
    drifted = (-51.25 + 1e-9, -30.05, -51.15, -29.95)
    assert cache_key(drifted, "2026-05-01", "walk") == k1


def test_cache_entry_paths(tmp_path) -> None:
    e = cache_entry(tmp_path, "abc123")
    assert e.root == tmp_path / "osm" / "abc123"
    assert e.nodes_path.name == "nodes.parquet"
    assert e.edges_path.name == "edges.parquet"
    assert e.manifest_path.name == "manifest.json"
    assert e.exists is False


def test_write_read_graph_roundtrips_list_valued_columns(tmp_path) -> None:
    """OSMnx merges ways into list-valued ``osmid``/``maxspeed`` when it
    simplifies; Parquet rejects a column mixing list and scalar cells. The cache
    must JSON-encode such columns and restore them exactly on read, so that e.g.
    a ``maxspeed`` list still parses to a real drive speed."""
    pytest.importorskip("osmnx")
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-75.16, y=39.95)
    g.add_node(2, x=-75.15, y=39.96)
    g.add_node(3, x=-75.14, y=39.95)
    # one edge with list-valued attrs (merged ways), one with plain scalars
    g.add_edge(
        1,
        2,
        key=0,
        length=100.0,
        osmid=[1, 2],
        maxspeed=["30", "40"],
        geometry=LineString([(-75.16, 39.95), (-75.15, 39.96)]),
    )
    g.add_edge(
        2, 3, key=0, length=120.0, osmid=99, maxspeed="50", geometry=LineString([(-75.15, 39.96), (-75.14, 39.95)])
    )

    bbox = (-75.16, 39.95, -75.14, 39.96)
    entry = cache_entry(tmp_path, cache_key(bbox, "2026-06-01", "drive"))
    write_graph(g, entry, bbox=bbox, snapshot_date="2026-06-01", network_type="drive")
    assert entry.exists
    assert "osmid" in read_manifest(entry)["json_columns"]["edges"]

    back = read_graph(entry)
    assert back.edges[1, 2, 0]["osmid"] == [1, 2]
    assert back.edges[1, 2, 0]["maxspeed"] == ["30", "40"]
    assert back.edges[1, 2, 0]["length"] == pytest.approx(100.0)
    assert back.edges[2, 3, 0]["osmid"] == 99
    assert back.edges[2, 3, 0]["maxspeed"] == "50"


# --------------------------------------------------------------------------- #
# Snapping                                                                     #
# --------------------------------------------------------------------------- #


def _grid_graph() -> nx.Graph:
    """4×4 grid at 100 m spacing — same layout as synthetic_city. Working CRS."""
    g = nx.Graph()
    for r in range(4):
        for c in range(4):
            g.add_node(r * 4 + c, x=c * 100.0, y=r * 100.0)
    return g


def test_snap_points_finds_nearest_node() -> None:
    g = _grid_graph()
    # five m east of node 0 → should snap to node 0 with distance 5
    pts = gpd.GeoDataFrame(
        {"id": ["p"]},
        geometry=[Point(5.0, 0.0)],
        crs="EPSG:32722",
    )
    s = Settings()  # tolerance_m default = 250
    result = snap_points(pts, g, s, working_crs="EPSG:32722")
    row = result.gdf.iloc[0]
    assert row["node_id"] == 0
    assert row["snap_distance_m"] == pytest.approx(5.0)
    assert row["over_tolerance"] is False or row["over_tolerance"] == False  # noqa: E712
    assert result.n_over_tolerance == 0


def test_snap_points_flags_over_tolerance() -> None:
    g = _grid_graph()
    # 500 m east of the grid → nearest is node 3 (col=3,row=0) at (300,0) → 200 m
    # tighten tolerance to 100 so this trips the audit flag
    pts = gpd.GeoDataFrame(
        {"id": ["far"]},
        geometry=[Point(500.0, 0.0)],
        crs="EPSG:32722",
    )
    s = Settings(network=NetworkConfig(snap_tolerance_m=100.0))
    result = snap_points(pts, g, s, working_crs="EPSG:32722")
    assert result.n_over_tolerance == 1
    assert bool(result.gdf.iloc[0]["over_tolerance"]) is True


def test_snap_points_empty_input() -> None:
    g = _grid_graph()
    pts = gpd.GeoDataFrame({"id": []}, geometry=gpd.GeoSeries([], crs="EPSG:32722"))
    result = snap_points(pts, g, Settings(), working_crs="EPSG:32722")
    assert len(result.gdf) == 0
    assert result.n_over_tolerance == 0


def test_snap_points_rejects_crs_mismatch() -> None:
    g = _grid_graph()
    pts = gpd.GeoDataFrame({"id": ["p"]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="working CRS"):
        snap_points(pts, g, Settings(), working_crs="EPSG:32722")


def test_snap_points_diagonal_distance() -> None:
    g = _grid_graph()
    # point at (50, 50) — equidistant to nodes 0, 1, 4, 5; distance = 50*sqrt(2)
    pts = gpd.GeoDataFrame(
        {"id": ["mid"]},
        geometry=[Point(50.0, 50.0)],
        crs="EPSG:32722",
    )
    result = snap_points(pts, g, Settings(), working_crs="EPSG:32722")
    assert result.gdf.iloc[0]["snap_distance_m"] == pytest.approx(50.0 * math.sqrt(2))
    assert result.gdf.iloc[0]["node_id"] in {0, 1, 4, 5}  # any tie is fine
