"""Street graphs from a local ``.osm.pbf`` extract (segx.network.pbf).

The extracts here are tiny PBFs written by pyosmium in ``tmp_path``: a 3x3
street grid in Porto Alegre with one footway, one motorway and one private
road, so the OSMnx walk/drive filters have something to tell apart.
"""

from __future__ import annotations

import networkx as nx
import osmium
import pytest

from segx.config import NetworkConfig, Settings
from segx.errors import InputFileNotFoundError, OSMExtractError
from segx.network import pbf
from segx.network.builder import fetch_raw_graph, osm_source
from segx.network.cache import cache_key

# 3x3 grid, ~110 m spacing; node id = 1 + row * 3 + col.
_LON0, _LAT0, _STEP = -51.2300, -30.0300, 0.001
_AREA = (_LON0 - 0.0005, _LAT0 - 0.0005, _LON0 + 2 * _STEP + 0.0005, _LAT0 + 2 * _STEP + 0.0005)


def _node(row: int, col: int) -> int:
    return 1 + row * 3 + col


def _write_pbf(path, ways, *, box=(-52.0, -31.0, -50.0, -29.0), timestamp="2026-10-20T20:23:47Z"):
    header = osmium.io.Header()
    if box is not None:
        header.add_box(osmium.osm.Box(osmium.osm.Location(box[0], box[1]), osmium.osm.Location(box[2], box[3])))
    if timestamp is not None:
        header.set("osmosis_replication_timestamp", timestamp)
    writer = osmium.SimpleWriter(str(path), header=header)
    try:
        for row in range(3):
            for col in range(3):
                writer.add_node(
                    osmium.osm.mutable.Node(id=_node(row, col), location=(_LON0 + col * _STEP, _LAT0 + row * _STEP))
                )
        for way_id, refs, tags in ways:
            writer.add_way(osmium.osm.mutable.Way(id=way_id, nodes=refs, tags=tags))
    finally:
        writer.close()
    return path


_GRID_WAYS = [
    # rows: residential streets
    (101, [_node(0, 0), _node(0, 1), _node(0, 2)], {"highway": "residential"}),
    (102, [_node(1, 0), _node(1, 1), _node(1, 2)], {"highway": "residential"}),
    (103, [_node(2, 0), _node(2, 1), _node(2, 2)], {"highway": "residential", "access": "private"}),
    # columns
    (201, [_node(0, 0), _node(1, 0), _node(2, 0)], {"highway": "footway"}),
    (202, [_node(0, 1), _node(1, 1), _node(2, 1)], {"highway": "motorway"}),
    (203, [_node(0, 2), _node(1, 2), _node(2, 2)], {"highway": "tertiary"}),
    # not a street at all
    (301, [_node(0, 0), _node(1, 1)], {"waterway": "stream"}),
]


@pytest.fixture
def grid_pbf(tmp_path):
    pbf._EXTRACTS.clear()
    yield _write_pbf(tmp_path / "grid.osm.pbf", _GRID_WAYS)
    pbf._EXTRACTS.clear()


def _settings(path=None, *, simplify=False) -> Settings:
    return Settings(network=NetworkConfig(osm_pbf_url=str(path) if path else None, simplify=simplify))


def _way_ids(graph: nx.MultiDiGraph) -> set[int]:
    out: set[int] = set()
    for *_, data in graph.edges(data=True):
        osmid = data["osmid"]
        out.update(osmid if isinstance(osmid, list) else [osmid])
    return out


# --------------------------------------------------------------------------- #
# Overpass filter evaluation                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("network_type", ["all", "all_public", "bike", "drive", "drive_service", "walk"])
def test_every_osmnx_network_filter_parses(network_type) -> None:
    # Pins the private OSMnx helper this module relies on.
    conditions = pbf.network_conditions(network_type)
    assert ("highway", None, "") in conditions


def test_unsupported_filter_syntax_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        pbf.parse_way_filter('["highway"](if: t["lanes"] > 2)')


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"highway": "residential"}, True),
        ({"highway": "residential", "service": "parking_aisle"}, True),  # "!~" on an unrelated value
        ({"highway": "footway"}, False),  # excluded by regex
        ({"highway": "residential", "access": "private"}, False),
        ({"name": "Rua da Praia"}, False),  # no highway tag
        ({"highway": "primary", "area": "yes"}, False),
    ],
)
def test_way_matches_uses_overpass_semantics(tags, expected) -> None:
    conditions = pbf.parse_way_filter(
        '["highway"]["area"!~"yes"]["access"!~"private"]["highway"!~"footway|path"]["service"!~"private"]'
    )
    assert pbf.way_matches(tags, conditions) is expected


def test_equality_operators() -> None:
    assert pbf.way_matches({"lanes": "2"}, pbf.parse_way_filter('["lanes"="2"]'))
    assert not pbf.way_matches({"lanes": "2"}, pbf.parse_way_filter('["lanes"!="2"]'))
    assert pbf.way_matches({"highway": "primary"}, pbf.parse_way_filter('["highway"~"prim"]'))


# --------------------------------------------------------------------------- #
# Reading and building                                                         #
# --------------------------------------------------------------------------- #


def test_header_gives_box_and_snapshot_date(grid_pbf) -> None:
    header = pbf.read_header(grid_pbf)
    assert header.bbox_wgs84 == pytest.approx((-52.0, -31.0, -50.0, -29.0))
    assert header.timestamp == "2026-10-20"


def test_read_highways_keeps_streets_only(grid_pbf) -> None:
    extract = pbf.read_highways(grid_pbf, _AREA)
    assert {w[0] for w in extract.ways} == {101, 102, 103, 201, 202, 203}
    assert set(extract.nodes) == {_node(r, c) for r in range(3) for c in range(3)}


def test_walk_and_drive_select_different_ways(grid_pbf) -> None:
    settings = _settings(grid_pbf)
    walk = pbf.graph_from_pbf(grid_pbf, _AREA, "walk", settings)
    drive = pbf.graph_from_pbf(grid_pbf, _AREA, "drive", settings)
    # walk: no motorway, no private; footway allowed.
    assert _way_ids(walk) == {101, 102, 201, 203}
    # drive: no footway, no private; motorway allowed.
    assert _way_ids(drive) == {101, 102, 202, 203}
    for graph in (walk, drive):
        assert graph.graph["crs"].lower() == "epsg:4326"
        assert all(data["length"] > 0 for *_, data in graph.edges(data=True))
        assert all("street_count" in data for _, data in graph.nodes(data=True))


def test_walk_is_bidirectional_and_drive_respects_oneway(tmp_path) -> None:
    pbf._EXTRACTS.clear()
    ways = [(101, [_node(0, 0), _node(0, 1), _node(0, 2)], {"highway": "residential", "oneway": "yes"})]
    path = _write_pbf(tmp_path / "oneway.osm.pbf", ways)
    settings = _settings(path)
    drive = pbf.graph_from_pbf(path, _AREA, "drive", settings)
    walk = pbf.graph_from_pbf(path, _AREA, "walk", settings)
    assert drive.has_edge(_node(0, 0), _node(0, 1)) and not drive.has_edge(_node(0, 1), _node(0, 0))
    assert walk.has_edge(_node(0, 1), _node(0, 0))


def test_simplify_merges_degree_two_nodes(tmp_path) -> None:
    pbf._EXTRACTS.clear()
    ways = [(101, [_node(0, 0), _node(0, 1), _node(0, 2)], {"highway": "residential"})]
    path = _write_pbf(tmp_path / "line.osm.pbf", ways)
    graph = pbf.graph_from_pbf(path, _AREA, "drive", _settings(path, simplify=True))
    assert set(graph.nodes) == {_node(0, 0), _node(0, 2)}


def test_extract_that_misses_the_area_is_a_typed_error(tmp_path) -> None:
    pbf._EXTRACTS.clear()
    path = _write_pbf(tmp_path / "elsewhere.osm.pbf", _GRID_WAYS, box=(-48.0, -27.0, -47.0, -26.0))
    with pytest.raises(OSMExtractError, match="does not cover") as ei:
        pbf.read_highways(path, _AREA)
    assert ei.value.code == "network.osm_extract"


def test_extract_without_streets_in_the_area_is_a_typed_error(grid_pbf) -> None:
    far = (-51.10, -29.90, -51.09, -29.89)  # inside the header box, outside the grid
    with pytest.raises(OSMExtractError, match="no streets"):
        pbf.read_highways(grid_pbf, far)


def test_extract_without_header_box_still_reads(tmp_path) -> None:
    pbf._EXTRACTS.clear()
    path = _write_pbf(tmp_path / "nobox.osm.pbf", _GRID_WAYS, box=None, timestamp=None)
    assert pbf.read_header(path) == pbf.PBFHeader(bbox_wgs84=None, timestamp=None)
    assert len(pbf.read_highways(path, _AREA).ways) == 6


def test_missing_file_is_input_file_not_found(tmp_path) -> None:
    with pytest.raises(InputFileNotFoundError):
        pbf.read_header(tmp_path / "nope.osm.pbf")


# --------------------------------------------------------------------------- #
# Builder + cache wiring                                                       #
# --------------------------------------------------------------------------- #


def test_fetch_raw_graph_reads_the_pbf_without_overpass(grid_pbf, monkeypatch) -> None:
    import osmnx as ox

    def _no_overpass(*args, **kwargs):
        raise AssertionError("Overpass must not be contacted when osm_pbf_url is set")

    monkeypatch.setattr(ox, "graph_from_bbox", _no_overpass)
    graph = fetch_raw_graph(_AREA, "walk", _settings(grid_pbf))
    assert _way_ids(graph) == {101, 102, 201, 203}


def test_cache_key_separates_pbf_from_overpass(grid_pbf) -> None:
    source = osm_source(_settings(grid_pbf))
    assert source is not None and source.startswith("pbf:grid.osm.pbf:")
    assert osm_source(_settings()) is None
    legacy = cache_key(_AREA, None, "walk")
    assert cache_key(_AREA, None, "walk", source=None) == legacy
    assert cache_key(_AREA, None, "walk", source=source) != legacy


def test_snapshot_date_comes_from_the_pbf_unless_pinned(grid_pbf) -> None:
    from segx.network.builder import osm_snapshot_date

    assert osm_snapshot_date(_settings(grid_pbf)) == "2026-10-20"
    pinned = Settings(network=NetworkConfig(osm_pbf_url=str(grid_pbf), osm_snapshot_date="2026-01-01"))
    assert osm_snapshot_date(pinned) == "2026-01-01"
    assert osm_snapshot_date(_settings()) is None
