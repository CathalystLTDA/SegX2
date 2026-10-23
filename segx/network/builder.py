"""Build per-mode street graphs: fetch (or load from cache), project to the
working CRS, attach mode-specific travel-time weights.

Three layers, separable on purpose:

* :func:`fetch_raw_graph` — the only place that hits the network (live OSMnx),
  or reads the local ``.osm.pbf`` extract when ``network.osm_pbf_url`` is set.
  Called once per ``(boundary, mode)`` cache miss.
* :func:`project_graph` and :func:`attach_travel_times` — pure NetworkX/pyproj
  operations on an in-memory graph, no IO. Unit-testable with hand-built graphs.
* :func:`load_or_fetch_graph` — the cache-aware orchestration glue: cache hit →
  read; miss → fetch (warning if ``osm_snapshot_date`` is unpinned) →
  cache → return.

The mode contract:

* ``walk`` — every edge gets ``travel_time_min = length_m / (walk_speed_kmh * 1000 / 60)``.
* ``drive`` — uses the edge's OSM ``maxspeed`` (parsed leniently); falls back to
  ``network.drive_speed_fallback_kmh`` when the tag is missing or unparseable.

Both modes assume each edge already carries a ``length`` attribute in metres,
which is true of OSMnx-fetched graphs and which the synthetic test fixtures
also set explicitly.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import networkx as nx
import pyproj
from shapely.ops import transform as shp_transform

from segx.errors import GraphError, ModeNotAvailableError
from segx.network.cache import cache_entry, cache_key, read_graph, write_graph
from segx.network.retry import with_overpass_retries

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from segx.config import Settings
    from segx.io.boundary import LoadedBoundary
    from segx.progress import CancelCheck


# --------------------------------------------------------------------------- #
# Live fetch (the only place that hits the network)                            #
# --------------------------------------------------------------------------- #


def fetch_raw_graph(
    bbox_wgs84: tuple[float, float, float, float],
    network_type: str,
    settings: Settings,
    *,
    cancel: CancelCheck | None = None,
) -> nx.MultiDiGraph:
    """Fetch a raw OSMnx graph for *bbox_wgs84* in WGS84.

    Returned graph carries ``graph.graph['crs'] = 'EPSG:4326'`` and edges already
    have ``length`` in metres (OSMnx computes great-circle lengths on fetch).
    Transient Overpass failures are retried with backoff (:mod:`segx.network.retry`).
    With ``network.osm_pbf_url`` set, the graph is read from that local extract
    instead (:mod:`segx.network.pbf`) and Overpass is never contacted.
    """
    if settings.network.osm_pbf_url is not None:
        from segx.network.pbf import graph_from_pbf

        return graph_from_pbf(settings.network.osm_pbf_url, bbox_wgs84, network_type, settings, cancel=cancel)

    import osmnx as ox

    minx, miny, maxx, maxy = bbox_wgs84
    return with_overpass_retries(
        lambda: ox.graph_from_bbox(
            bbox=(minx, miny, maxx, maxy),
            network_type=network_type,
            simplify=settings.network.simplify,
        ),
        what=f"the {network_type} street network",
        settings=settings,
        cancel=cancel,
    )


# --------------------------------------------------------------------------- #
# Projection                                                                   #
# --------------------------------------------------------------------------- #


def project_graph(graph: nx.MultiDiGraph, working_crs: str) -> nx.MultiDiGraph:
    """Reproject ``graph`` (node coords + edge geometries) to *working_crs*.

    Done with pyproj directly so the function operates on any NetworkX graph
    that exposes ``graph.graph['crs']``, ``node['x']`` and ``node['y']`` —
    we don't need to involve OSMnx for the projection step itself.
    """
    src_crs = graph.graph.get("crs")
    if src_crs is None:
        raise GraphError("graph has no CRS metadata at graph.graph['crs']")
    if pyproj.CRS.from_user_input(src_crs) == pyproj.CRS.from_user_input(working_crs):
        return graph.copy()

    transformer = pyproj.Transformer.from_crs(src_crs, working_crs, always_xy=True)
    project_xy = transformer.transform

    g = graph.copy()
    g.graph["crs"] = working_crs

    for _, data in g.nodes(data=True):
        x, y = project_xy(data["x"], data["y"])
        data["x"] = float(x)
        data["y"] = float(y)

    def _proj_2d(x: float, y: float, z: float | None = None):
        nx_, ny_ = project_xy(x, y)
        return (nx_, ny_) if z is None else (nx_, ny_, z)

    for *_, data in g.edges(data=True):
        geom = data.get("geometry")
        if geom is not None:
            data["geometry"] = shp_transform(_proj_2d, geom)

    return g


# --------------------------------------------------------------------------- #
# Travel-time weighting                                                        #
# --------------------------------------------------------------------------- #


def parse_maxspeed_kmh(value: object) -> float | None:
    """Best-effort parse of OSM ``maxspeed`` — supports plain numbers,
    ``"50"``, ``"50 km/h"``, ``"30 mph"`` and lists (returns the minimum).

    Returns ``None`` for missing / unparseable values; callers fall back to
    ``drive_speed_fallback_kmh``.
    """
    if value is None:
        return None
    if isinstance(value, list):
        parsed = [v for v in (parse_maxspeed_kmh(x) for x in value) if v is not None]
        return min(parsed) if parsed else None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if not s:
        return None
    if s.endswith("mph"):
        try:
            return float(s.removesuffix("mph").strip()) * 1.609344
        except ValueError:
            return None
    if s.endswith("km/h"):
        s = s.removesuffix("km/h").strip()
    try:
        return float(s)
    except ValueError:
        return None


def attach_travel_times(
    graph: nx.MultiDiGraph,
    mode: str,
    settings: Settings,
) -> None:
    """Set ``travel_time_min`` on every edge, in place.

    Edges without a ``length`` attribute raise — every graph leaving the
    network layer must carry travel-time weights, and a missing length means
    the caller didn't pass a properly built OSMnx graph.
    """
    walk_kmh = settings.network.walk_speed_kmh
    drive_fallback_kmh = settings.network.drive_speed_fallback_kmh

    def _minutes(length_m: float, speed_kmh: float) -> float:
        return length_m * 60.0 / (speed_kmh * 1000.0)

    if mode == "walk":
        for *_, data in graph.edges(data=True):
            length = data.get("length")
            if length is None:
                raise GraphError("edge missing 'length' attribute; cannot weight walk graph")
            data["travel_time_min"] = _minutes(float(length), walk_kmh)
    elif mode == "drive":
        for *_, data in graph.edges(data=True):
            length = data.get("length")
            if length is None:
                raise GraphError("edge missing 'length' attribute; cannot weight drive graph")
            posted = parse_maxspeed_kmh(data.get("maxspeed"))
            speed = posted if (posted is not None and posted > 0) else drive_fallback_kmh
            data["travel_time_min"] = _minutes(float(length), speed)
            data["effective_speed_kmh"] = speed
    elif mode == "transit":
        # P7: the merged graph's transit edges (connector / board / ride / alight /
        # transfer) already carry the travel times :mod:`segx.network.transit`
        # computed from headways and inter-stop times — re-deriving them from
        # ``length`` would be meaningless (a board edge has no length). Only the
        # inherited *street* edges are (re)weighted, at walking speed.
        for *_, data in graph.edges(data=True):
            if data.get("segx_layer", "street") != "street":
                continue
            if data.get("segx_edge") == "ride":
                continue  # a surface ride keeps its scheduled running time
            length = data.get("length")
            if length is None:
                raise GraphError("edge missing 'length' attribute; cannot weight transit street edge")
            data["travel_time_min"] = _minutes(float(length), walk_kmh)
    else:
        raise ModeNotAvailableError(
            f"unsupported mode {mode!r}; supported modes are 'walk', 'drive' and 'transit'",
            details={"mode": mode, "supported": ["walk", "drive", "transit"]},
        )


def to_undirected_graph(graph: nx.MultiDiGraph) -> nx.MultiGraph:
    """Collapse a directed street graph to undirected, merging each reciprocal
    edge pair into **one** segment (a street is one map line).

    Uses NetworkX ``to_undirected(reciprocal=False)`` — reciprocal edges that
    share a key (the usual OSM case, key ``0``) merge into a single undirected
    edge; the rare reciprocal pair with *distinct* keys stays as parallel edges,
    an accepted approximation. Routing is unaffected: the engines rebuild a
    ``MultiDiGraph`` view (``routing/_paths.as_multidigraph``) that re-materialises
    both directions with equal weights, so shortest paths are unchanged — only the
    segment identity (and thus the output map) becomes undirected. Deliberately
    NetworkX-only (no OSMnx) so it is testable on synthetic graphs.

    NB: for a ``drive`` graph this discards one-way restrictions — the user opts
    in via ``network.undirected``.
    """
    return graph.to_undirected(reciprocal=False)


def build_mode_graph(
    raw_graph: nx.MultiDiGraph,
    mode: str,
    settings: Settings,
    *,
    working_crs: str,
) -> nx.MultiDiGraph | nx.MultiGraph:
    """Project a raw OSMnx graph to *working_crs* and weight it for *mode*.

    When ``settings.network.undirected`` is set, the projected graph is collapsed
    to undirected (:func:`to_undirected_graph`) *before* travel-time weighting, so
    the weights land on the merged segments.
    """
    projected = project_graph(raw_graph, working_crs)
    graph = to_undirected_graph(projected) if settings.network.undirected else projected
    attach_travel_times(graph, mode, settings)
    return graph


# --------------------------------------------------------------------------- #
# Cache-aware orchestration                                                    #
# --------------------------------------------------------------------------- #


def osm_source(settings: Settings) -> str | None:
    """Cache-key token for where OSM data comes from: ``None`` = live Overpass,
    else the local extract's identity (:func:`segx.network.pbf.source_id`)."""
    if settings.network.osm_pbf_url is None:
        return None
    from segx.network.pbf import source_id

    return source_id(settings.network.osm_pbf_url)


def load_or_fetch_graph(
    boundary: LoadedBoundary,
    mode: str,
    settings: Settings,
    cache_dir: Path | None = None,
    *,
    cancel: CancelCheck | None = None,
) -> nx.MultiDiGraph:
    """Return a built, projected, weighted graph for *mode* — from cache if
    available, otherwise via :func:`fetch_raw_graph`.

    A run without ``network.osm_snapshot_date`` set is allowed but emits a
    ``UserWarning`` — the result will not be reproducible.

    ``mode == "transit"`` (P7) is not an OSM extract of its own: it is the walk
    graph with a public-transport network merged into it, so this builds ``walk``
    (and ``drive``, when surface rides are map-matched) first and delegates to
    :func:`segx.network.transit.build_transit_graph`. Both street graphs come
    through this same cache-aware path, so nothing is fetched twice.
    """
    cache_dir = cache_dir or settings.cache_dir

    if mode == "transit":
        return _build_transit_mode(boundary, settings, cache_dir, cancel=cancel)

    source = osm_source(settings)
    key = cache_key(boundary.bbox_wgs84, settings.network.osm_snapshot_date, mode, source=source)
    entry = cache_entry(cache_dir, key)

    if entry.exists:
        raw = read_graph(entry)
    else:
        # A local extract is a fixed snapshot: reproducible without a pinned date.
        if settings.network.osm_snapshot_date is None and source is None:
            warnings.warn(
                "network.osm_snapshot_date is not pinned — this run will fetch the current "
                "live OSM state and the result will not be reproducible. Set "
                "network.osm_snapshot_date in segx.toml.",
                UserWarning,
                stacklevel=2,
            )
        raw = fetch_raw_graph(boundary.bbox_wgs84, mode, settings, cancel=cancel)
        write_graph(
            raw,
            entry,
            bbox=boundary.bbox_wgs84,
            snapshot_date=settings.network.osm_snapshot_date,
            network_type=mode,
            source=source,
        )

    return build_mode_graph(raw, mode, settings, working_crs=boundary.crs)


def _build_transit_mode(
    boundary: LoadedBoundary,
    settings: Settings,
    cache_dir: Path,
    *,
    cancel: CancelCheck | None = None,
) -> nx.MultiDiGraph:
    """Build the merged walk ⊕ transit graph for the study area (P7).

    Refuses to build when ``transit.enabled`` is off: silently producing a
    transit-shaped graph with no transit in it would make every transit metric
    read as a plain walking result, which is exactly the kind of plausible-looking
    wrong answer invariant #2 exists to prevent.
    """
    from segx.network.transit import build_transit_graph, load_transit_feed

    if not settings.transit.enabled:
        raise ModeNotAvailableError(
            "mode 'transit' was requested but transit.enabled is false; "
            "set transit.enabled=true (and supply transit.gtfs_path for a measured feed)",
            details={"mode": "transit", "transit_enabled": False},
        )

    walk_graph = load_or_fetch_graph(boundary, "walk", settings, cache_dir, cancel=cancel)
    drive_graph = (
        load_or_fetch_graph(boundary, "drive", settings, cache_dir, cancel=cancel)
        if settings.transit.map_match_surface_rides
        else None
    )
    feed = load_transit_feed(boundary, settings, cache_dir=cache_dir)
    return build_transit_graph(feed, walk_graph, drive_graph, settings, working_crs=boundary.crs)


__all__ = [
    "attach_travel_times",
    "build_mode_graph",
    "fetch_raw_graph",
    "load_or_fetch_graph",
    "osm_source",
    "parse_maxspeed_kmh",
    "project_graph",
    "to_undirected_graph",
]
