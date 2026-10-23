"""Street graphs read from a local OpenStreetMap extract (``.osm.pbf``).

The offline counterpart of the Overpass download in
:func:`segx.network.builder.fetch_raw_graph`, used when ``network.osm_pbf_url``
points to a file (a Geofabrik / BBBike extract). The result is meant to be the
same graph Overpass would have given, so everything downstream is unchanged:

* ways are selected with **OSMnx's own** ``network_type`` filter
  (``osmnx._overpass._get_network_filter``), evaluated here with Overpass
  semantics (``["k"!~"re"]`` = tag absent or not matching);
* the graph is assembled by OSMnx's Overpass-response parser
  (``osmnx.graph._create_graph``) from a synthesised response, then truncated,
  reduced to its largest component and simplified exactly as
  ``osmnx.graph_from_polygon`` does (0.5 km buffer, truncate, simplify,
  truncate again).

Reading the file is the slow part (~1 min and ~2 GB for a 400 MB regional
extract), so one pass collects every ``highway`` way touching the buffered
study area and keeps it in memory for the process; the per-mode filters run on
that subset, and walk + drive cost one read. The built graphs are then cached
on disk like Overpass ones (:mod:`segx.network.cache`).

Two OSMnx private helpers are used on purpose — they are what keeps the PBF
graph identical to the Overpass one. ``tests/unit/test_pbf.py`` pins them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from segx.errors import InputFileNotFoundError, OSMExtractError
from segx.progress import check_cancel

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

    from segx.config import Settings
    from segx.progress import CancelCheck


#: OSMnx buffers the requested polygon by this much before truncating (graph_from_polygon).
_BUFFER_M = 500
#: How many ways pass between two cancellation checks while reading the file.
_CANCEL_EVERY = 50_000

#: One Overpass tag condition: ``["key"]``, ``["key"="v"]``, ``["key"~"re"]``, ``["key"!~"re"]``.
_CONDITION = re.compile(r'\["([^"]+)"(?:(!?~|!?=)"([^"]*)")?\]')


# --------------------------------------------------------------------------- #
# Overpass way filters                                                         #
# --------------------------------------------------------------------------- #


def parse_way_filter(way_filter: str) -> list[tuple[str, str | None, str]]:
    """Split an Overpass way filter into ``(key, op, value)`` conditions (ANDed).

    ``op`` is ``None`` for a bare ``["key"]`` (tag must exist). Raises
    ``ValueError`` on anything else, so an OSMnx filter change fails loudly
    instead of silently selecting other ways.
    """
    conditions = [(m[1], m[2], m[3] or "") for m in _CONDITION.finditer(way_filter)]
    if _CONDITION.sub("", way_filter).strip():
        raise ValueError(f"unsupported Overpass way filter: {way_filter!r}")
    return conditions


def way_matches(tags: dict[str, str], conditions: list[tuple[str, str | None, str]]) -> bool:
    """Whether a way with *tags* passes every condition (Overpass semantics)."""
    for key, op, value in conditions:
        tag = tags.get(key)
        if op is None:
            ok = tag is not None
        elif op == "=":
            ok = tag == value
        elif op == "!=":
            ok = tag != value
        elif op == "~":
            ok = tag is not None and re.search(value, tag) is not None
        else:  # "!~": absent, or present and not matching
            ok = tag is None or re.search(value, tag) is None
        if not ok:
            return False
    return True


def network_conditions(network_type: str) -> list[tuple[str, str | None, str]]:
    """The conditions OSMnx's Overpass query uses for *network_type*."""
    from osmnx import _overpass

    return parse_way_filter(_overpass._get_network_filter(network_type))


# --------------------------------------------------------------------------- #
# Reading the file                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PBFHeader:
    """What the file says about itself: its coverage and when it was cut."""

    bbox_wgs84: tuple[float, float, float, float] | None  # (minx, miny, maxx, maxy); None = not stated
    timestamp: str | None  # ISO date of the snapshot, e.g. "2026-10-20"; None = not stated


@dataclass(frozen=True, slots=True)
class HighwayExtract:
    """Every ``highway`` way touching an area, with the coordinates of all its nodes."""

    nodes: dict[int, tuple[float, float]]  # node id -> (lon, lat)
    ways: list[tuple[int, list[int], dict[str, str]]]  # (way id, node ids, tags)


def _pbf_path(pbf: str | Path) -> Path:
    path = Path(pbf)
    if not path.is_file():
        raise InputFileNotFoundError(
            f"OpenStreetMap extract not found: {path}",
            details={"path": str(path), "setting": "network.osm_pbf_url"},
        )
    return path


def read_header(pbf: str | Path) -> PBFHeader:
    """The bounding box and snapshot date stored in the file header (cheap)."""
    import osmium

    reader = osmium.io.Reader(str(_pbf_path(pbf)), osmium.osm.osm_entity_bits.NOTHING)
    try:
        header = reader.header()
        box = header.box()
        bbox = (box.bottom_left.lon, box.bottom_left.lat, box.top_right.lon, box.top_right.lat) if box.valid() else None
        stamp = header.get("osmosis_replication_timestamp") or header.get("timestamp")
    finally:
        reader.close()
    timestamp = None
    if stamp:
        try:
            timestamp = datetime.fromisoformat(stamp.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            timestamp = None
    return PBFHeader(bbox_wgs84=bbox, timestamp=timestamp)


def source_id(pbf: str | Path) -> str:
    """A cache-key token for the file: name + size + header date.

    Not the full path (the same extract moved elsewhere is the same data) and
    not the mtime (copying a file changes it).
    """
    path = _pbf_path(pbf)
    return f"pbf:{path.name}:{path.stat().st_size}:{read_header(path).timestamp}"


def _check_coverage(header: PBFHeader, bbox_wgs84: tuple[float, float, float, float], pbf: Path) -> None:
    if header.bbox_wgs84 is None:
        return  # not stated: the empty-extract check below still catches a wrong region
    hminx, hminy, hmaxx, hmaxy = header.bbox_wgs84
    minx, miny, maxx, maxy = bbox_wgs84
    if minx < hminx or miny < hminy or maxx > hmaxx or maxy > hmaxy:
        raise OSMExtractError(
            f"The OpenStreetMap extract {pbf.name} does not cover the study area: "
            f"the file spans {header.bbox_wgs84}, the area needs {bbox_wgs84}. "
            "Use an extract of a region that contains the whole boundary.",
            details={"path": str(pbf), "pbf_bbox": list(header.bbox_wgs84), "area_bbox": list(bbox_wgs84)},
        )


#: In-process memo of the last extract read: (path, size, bbox) -> extract.
_EXTRACTS: dict[tuple[str, int, tuple[float, ...]], HighwayExtract] = {}


def read_highways(
    pbf: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
    *,
    cancel: CancelCheck | None = None,
) -> HighwayExtract:
    """Every ``highway`` way with at least one node inside *bbox_wgs84*.

    Whole ways are kept, nodes outside the box included — the same thing the
    Overpass ``way(poly); (._;>;);`` query returns; truncation happens later on
    the graph. Memoised for the process (one entry: the last area read).
    """
    import osmium

    path = _pbf_path(pbf)
    memo_key = (str(path.resolve()), path.stat().st_size, tuple(round(c, 6) for c in bbox_wgs84))
    if memo_key in _EXTRACTS:
        return _EXTRACTS[memo_key]
    _check_coverage(read_header(path), bbox_wgs84, path)

    minx, miny, maxx, maxy = bbox_wgs84
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[tuple[int, list[int], dict[str, str]]] = []
    processor = osmium.FileProcessor(str(path), osmium.osm.NODE | osmium.osm.WAY)
    processor = processor.with_locations().with_filter(osmium.filter.KeyFilter("highway"))
    seen = 0
    for obj in processor:
        if not obj.is_way():
            continue
        seen += 1
        if seen % _CANCEL_EVERY == 0:
            check_cancel(cancel, stage="build_networks")
        refs = [(nd.ref, nd.location) for nd in obj.nodes]
        if not any(loc.valid() and minx <= loc.lon <= maxx and miny <= loc.lat <= maxy for _, loc in refs):
            continue
        if not all(loc.valid() for _, loc in refs):
            continue  # a node the extract cut off: the way cannot be drawn whole
        for ref, loc in refs:
            nodes[ref] = (loc.lon, loc.lat)
        ways.append((obj.id, [ref for ref, _ in refs], {t.k: t.v for t in obj.tags}))

    if not ways:
        raise OSMExtractError(
            f"The OpenStreetMap extract {path.name} has no streets inside the study area "
            f"{bbox_wgs84}. Use an extract of a region that contains the boundary.",
            details={"path": str(path), "area_bbox": list(bbox_wgs84)},
        )
    _EXTRACTS.clear()
    extract = HighwayExtract(nodes=nodes, ways=ways)
    _EXTRACTS[memo_key] = extract
    return extract


# --------------------------------------------------------------------------- #
# Building the graph                                                           #
# --------------------------------------------------------------------------- #


def overpass_response(extract: HighwayExtract, network_type: str) -> dict[str, Any]:
    """The Overpass JSON OSMnx would have received for *network_type* over the extract."""
    conditions = network_conditions(network_type)
    elements: list[dict[str, Any]] = []
    used: set[int] = set()
    for way_id, refs, tags in extract.ways:
        if way_matches(tags, conditions):
            elements.append({"type": "way", "id": way_id, "nodes": refs, "tags": tags})
            used.update(refs)
    for node_id in used:
        lon, lat = extract.nodes[node_id]
        elements.append({"type": "node", "id": node_id, "lat": lat, "lon": lon})
    return {"elements": elements}


def graph_from_pbf(
    pbf: str | Path,
    bbox_wgs84: tuple[float, float, float, float],
    network_type: str,
    settings: Settings,
    *,
    cancel: CancelCheck | None = None,
) -> nx.MultiDiGraph:
    """The ``network_type`` graph for *bbox_wgs84*, as ``ox.graph_from_bbox`` builds it.

    Mirrors ``osmnx.graph_from_polygon`` (defaults ``retain_all=False``,
    ``truncate_by_edge=False``) with the download swapped for the file read.
    """
    import networkx
    from osmnx import projection, simplification, stats, truncate, utils_geo
    from osmnx import settings as ox_settings
    from osmnx.graph import _create_graph

    polygon = utils_geo.bbox_to_poly(bbox_wgs84)
    poly_proj, crs_utm = projection.project_geometry(polygon)
    poly_buff, _ = projection.project_geometry(poly_proj.buffer(_BUFFER_M), crs=crs_utm, to_latlong=True)

    extract = read_highways(pbf, tuple(poly_buff.bounds), cancel=cancel)
    response = overpass_response(extract, network_type)
    if not response["elements"]:
        raise OSMExtractError(
            f"The OpenStreetMap extract has no {network_type} streets inside the study area {bbox_wgs84}.",
            details={"path": str(pbf), "network_type": network_type, "area_bbox": list(bbox_wgs84)},
        )

    check_cancel(cancel, stage="build_networks")
    bidirectional = network_type in ox_settings.bidirectional_network_types
    graph_buff = _create_graph([response], bidirectional)
    graph_buff = truncate.truncate_graph_polygon(graph_buff, poly_buff, truncate_by_edge=False)
    graph_buff = truncate.largest_component(graph_buff, strongly=False)
    if settings.network.simplify:
        graph_buff = simplification.simplify_graph(graph_buff)
    graph = truncate.truncate_graph_polygon(graph_buff, polygon, truncate_by_edge=False)
    graph = truncate.largest_component(graph, strongly=False)
    # Street counts from the buffered graph, so an edge-of-area intersection keeps its true degree.
    networkx.set_node_attributes(
        graph, stats.count_streets_per_node(graph_buff, nodes=graph.nodes), name="street_count"
    )
    return graph


__all__ = [
    "HighwayExtract",
    "PBFHeader",
    "graph_from_pbf",
    "network_conditions",
    "overpass_response",
    "parse_way_filter",
    "read_header",
    "read_highways",
    "source_id",
    "way_matches",
]
