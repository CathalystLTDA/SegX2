"""Canonical **street space** — one corridor id shared by every mode that uses
the same physical stretch of street. P7.2.

Why this exists
---------------

``cross_mode_exposure`` (P7.1) can only mix two modes where they deposit on the
*same* segment id. In a real city they never do, for two independent reasons:

1. ``walk`` and ``drive`` are separate OSM extracts, simplified independently, so
   a node that survives as an intersection in one is a removed degree-2 node in
   the other;
2. — the dominant one, and the one an id-based diagnosis misses —
   where sidewalks are **mapped as their own ways**, the pedestrian and the car
   are on different geometries entirely. OSMnx's walk filter *excludes* roads
   tagged ``sidewalk=separate``, so it is not a divergence to be reconciled: they
   are different OSM ways, with different nodes and different ``osmid``.

Measured on the Philadelphia study area (``demo/out/cache/osm``): only **201 of
533** ``drive`` ways appear in the ``walk`` graph at all, and 5316 of the walk
graph's 8316 edges are ``highway=footway``. That is why an ``osmid``-translation
fix has a structural ceiling there, and why the fix has
to be geometric. The same measurement says geometry works: a footway sits a
**median of 4.3 m** from its roadway centreline, and 91 % are within 15 m.

What it does
------------

Every edge of every registered mode graph is matched to a **corridor** — the
roadway block it belongs to. Anchors come from the first mode in
``network.street_space.anchor_priority`` present in the run (``drive`` by
default, because a corridor *is* a roadway block); every other edge is matched by
midpoint proximity plus a bearing test, which is what keeps a perpendicular
crossing out of the corridor of the street it crosses.

An edge that matches nothing gets a corridor **of its own**. A pedestrian mall or
a park path is a real street space, not a failure — invariant #2: never a silent
zero. Non-``street`` layers (all of the rail side, P7 D4), indoor corridors and
tunnels are excluded from matching by construction: a passenger in a concourse is
not co-present with the avenue above.

The corridor is a property of the **edge**, exactly like ``segx_layer``, and
rides through aggregation as a passthrough column. It must never enter the
``(mode, category, binding)`` partition key (invariant #6).

Pure module: no IO, no network, no OSMnx. Any NetworkX graph whose nodes carry
``x``/``y`` in the working CRS works, which includes the synthetic test city.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from segx.aggregation.edge_accumulator import canonical_edge

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

    from segx.config import Settings, StreetSpaceConfig
    from segx.network.registry import GraphRegistry
    from segx.routing.trajectories import EdgeId

__all__ = [
    "CorridorInfo",
    "StreetSpace",
    "edges_as_gdf",
    "own_corridor_id",
    "resolve_street_space",
]


@dataclass(frozen=True, slots=True)
class CorridorInfo:
    """Which street space an edge belongs to, and how long the edge is.

    ``length_m`` is what makes the cross-mode pool granularity-invariant: a
    corridor's representative mass per mode is the **length-weighted mean** over
    that mode's edges in it, so a pedestrian who walks one block split into three
    sidewalk edges counts once, not three times.

    ``matched is False`` means the edge is its own corridor — either nothing was
    near enough, or it was excluded from matching (rail, tunnel, indoor). It is a
    real result, not an error, and :attr:`StreetSpace.stats` reports how often it
    happens so the reader can tell "the modes don't meet here" from "the matching
    is mis-tuned".
    """

    corridor_id: str
    length_m: float
    matched: bool


@dataclass(frozen=True, slots=True)
class StreetSpace:
    """The resolved corridor map for one run, plus its audit and provenance."""

    corridors: dict[EdgeId, CorridorInfo] = field(default_factory=dict)
    #: coverage audit — see :func:`resolve_street_space`
    stats: dict[str, Any] = field(default_factory=dict)
    #: the matching parameters, stamped into output metadata (invariant #7)
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, edge: EdgeId) -> CorridorInfo | None:
        return self.corridors.get(canonical_edge(edge))


# --------------------------------------------------------------------------- #
# Geometry helpers                                                             #
# --------------------------------------------------------------------------- #


def _edge_geometry(graph: nx.Graph, u: int, v: int, data: dict) -> LineString | None:
    """The edge's geometry, falling back to the straight chord between its nodes.

    OSMnx sets ``geometry`` only on edges simplified from a multi-node way; a
    straight block has none and is exactly its chord.
    """
    geom = data.get("geometry")
    if geom is not None and getattr(geom, "geom_type", None) == "LineString" and not geom.is_empty:
        return geom
    try:
        a = graph.nodes[u]
        b = graph.nodes[v]
        return LineString([(float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"]))])
    except (KeyError, TypeError, ValueError):
        return None


def _bearings_deg(geoms) -> list[float]:
    """Orientation of each geometry's chord in degrees, folded into ``[0, 180)``.

    Folded because a street has no direction for this purpose: a sidewalk mapped
    the other way round is still parallel to its roadway.

    The end coordinates are pulled out in one vectorised call — reaching into
    ``geom.coords`` once per edge is most of what a per-geometry ``map`` costs on
    a city-scale network.
    """
    import numpy as np
    import shapely

    coords, index = shapely.get_coordinates(np.asarray(geoms, dtype=object), return_index=True)
    firsts = np.flatnonzero(np.diff(index, prepend=-1))
    lasts = np.append(firsts[1:], len(index)) - 1
    return [
        math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        for (x0, y0), (x1, y1) in zip(coords[firsts].tolist(), coords[lasts].tolist(), strict=True)
    ]


def _bearing_delta(a: float, b: float) -> float:
    """Smallest angle between two folded bearings, in ``[0, 90]``."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _first_tag(value: object) -> str | None:
    """OSM tags are scalar *or* list-valued once ways are merged by simplify."""
    if isinstance(value, list):
        return str(value[0]) if value else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _is_truthy_tag(value: object) -> bool:
    tag = _first_tag(value)
    return tag is not None and tag.lower() not in {"", "no", "false", "0", "nan", "none"}


def own_corridor_id(mode: str, edge: EdgeId) -> str:
    """Corridor id for an edge that belongs to no shared street space."""
    u, v, key = edge
    return f"own:{mode}:{u}:{v}:{key}"


def _anchor_corridor_id(edge: EdgeId) -> str:
    u, v, key = edge
    return f"e:{u}:{v}:{key}"


# --------------------------------------------------------------------------- #
# Materialisation                                                              #
# --------------------------------------------------------------------------- #


def edges_as_gdf(graph: nx.Graph, mode: str, working_crs: str) -> gpd.GeoDataFrame:
    """Every edge of *graph* as one row, keyed by :func:`canonical_edge`.

    Columns: ``u``, ``v``, ``key``, ``mode``, ``length_m``, ``highway``,
    ``layer``, ``tunnel``, ``geometry``. Endpoint order is canonicalised so the
    key matches the accumulator's segment key regardless of which way a
    trajectory traversed the edge.
    """
    rows: list[dict] = []
    seen: set[EdgeId] = set()
    for u, v, key, data in graph.edges(keys=True, data=True):
        edge = canonical_edge((int(u), int(v), int(key)))
        if edge in seen:  # a reciprocal pair folds to one segment
            continue
        geom = _edge_geometry(graph, u, v, data)
        if geom is None:
            continue
        seen.add(edge)
        length = data.get("length")
        rows.append(
            {
                "u": edge[0],
                "v": edge[1],
                "key": edge[2],
                "mode": mode,
                "length_m": float(length) if length is not None else float(geom.length),
                "highway": _first_tag(data.get("highway")),
                "layer": str(data.get("segx_layer", "street")),
                "tunnel": _is_truthy_tag(data.get("tunnel")),
                "geometry": geom,
            }
        )
    return gpd.GeoDataFrame(
        rows,
        columns=["u", "v", "key", "mode", "length_m", "highway", "layer", "tunnel", "geometry"],
        geometry="geometry",
        crs=working_crs,
    )


def _collect_edges(graphs: GraphRegistry, working_crs: str) -> gpd.GeoDataFrame:
    """One row per distinct ``(u, v, key)`` across every registered mode graph.

    A triple present in two graphs is the *same* physical edge — ``u``/``v`` are
    OSM node ids, so the namespace is shared — and is kept once. Which mode's row
    survives follows :func:`resolve_street_space`'s anchor priority, so the
    anchor layer's own geometry and tags win.
    """
    frames = [edges_as_gdf(graphs.get(mode), mode, working_crs) for mode in graphs.modes()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gpd.GeoDataFrame(
            columns=["u", "v", "key", "mode", "length_m", "highway", "layer", "tunnel", "geometry"],
            geometry="geometry",
            crs=working_crs,
        )
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=working_crs)


def _excluded(row, cfg: StreetSpaceConfig) -> bool:
    """Edges that never join a shared corridor.

    Rail and every other non-``street`` layer (P7 D4 — a passenger in a tunnel is
    co-present with nobody on the avenue above), indoor corridors/elevators/steps,
    and anything tagged as a tunnel.
    """
    if row.layer != "street":
        return True
    if cfg.exclude_tunnel and bool(row.tunnel):
        return True
    return row.highway is not None and row.highway in set(cfg.exclude_highway)


# --------------------------------------------------------------------------- #
# Resolution                                                                   #
# --------------------------------------------------------------------------- #


def resolve_street_space(
    graphs: GraphRegistry,
    settings: Settings,
    *,
    working_crs: str,
) -> StreetSpace:
    """Assign every edge of every registered mode graph to a corridor.

    Anchors are the (non-excluded) edges of the first mode in
    ``network.street_space.anchor_priority`` that this run actually holds; each
    anchor *is* a corridor. Every other edge is matched by ``sjoin_nearest`` on
    its midpoint within ``tolerance_m``, then confirmed by a bearing test against
    the anchor — the same audited-nearest-join idiom as
    :func:`segx.network.snapping.snap_points`. Anything unmatched becomes its own
    corridor (invariant #2).

    :attr:`StreetSpace.stats` reports ``corridor_match_rate`` and, crucially, a
    **per mode-pair** count of shared corridors. The scalar rate is what let the
    Philadelphia run report 64 % while ``drive ∩ walk`` was exactly zero.

    Measured on that same study area (walk 8316 edges, drive 695): 616 anchors,
    ``corridor_match_rate`` 0.72, and **581 corridors carrying both modes** where
    the raw ``(u, v, key)`` join found 6. Read the residual before calling it a
    miss — 97 % of the walk graph's *roadway* edges and 80 % of its ``footway``
    edges match, while ``highway=service`` (parking aisles and interior alleys,
    which the drive filter excludes too) matches 20 %. An alley inside a block is
    not a corridor of the street; it is its own street space.
    """
    cfg = settings.network.street_space
    params = {
        "tolerance_m": cfg.tolerance_m,
        "max_bearing_deg": cfg.max_bearing_deg,
        "anchor_priority": list(cfg.anchor_priority),
        "exclude_highway": list(cfg.exclude_highway),
        "exclude_tunnel": cfg.exclude_tunnel,
    }

    edges = _collect_edges(graphs, working_crs)
    if edges.empty:
        return StreetSpace(
            corridors={}, stats={"n_edges": 0, "n_anchors": 0, "corridor_match_rate": 0.0}, params=params
        )

    present = [m for m in cfg.anchor_priority if m in graphs] or graphs.modes()
    anchor_mode = present[0]
    rank = {mode: i for i, mode in enumerate(present)}
    edges["_rank"] = edges["mode"].map(lambda m: rank.get(m, len(rank)))
    edges = (
        edges.sort_values(["u", "v", "key", "_rank"], kind="stable")
        .drop_duplicates(subset=["u", "v", "key"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    edges["excluded"] = [_excluded(row, cfg) for row in edges.itertuples()]
    edges["bearing"] = _bearings_deg(edges.geometry)
    edges["edge"] = list(zip(edges["u"], edges["v"], edges["key"], strict=True))

    lengths = dict(zip(edges["edge"], edges["length_m"].astype(float), strict=True))
    modes_of = dict(zip(edges["edge"], edges["mode"], strict=True))

    is_anchor = edges["mode"].eq(anchor_mode) & ~edges["excluded"]
    anchors = edges.loc[is_anchor]
    corridors: dict[EdgeId, CorridorInfo] = {}
    for row in anchors.itertuples():
        corridors[row.edge] = CorridorInfo(_anchor_corridor_id(row.edge), lengths[row.edge], True)

    matchable = edges.loc[~is_anchor & ~edges["excluded"]]
    n_matched = 0
    if not anchors.empty and not matchable.empty:
        for edge, anchor_edge in _match_to_anchors(matchable, anchors, cfg, working_crs):
            info = corridors.get(anchor_edge)
            if info is None:  # pragma: no cover — anchors are all registered above
                continue
            corridors[edge] = CorridorInfo(info.corridor_id, lengths[edge], True)
            n_matched += 1

    # Everything still unassigned — excluded, out of tolerance, or bearing-rejected.
    for edge in edges["edge"]:
        if edge not in corridors:
            corridors[edge] = CorridorInfo(own_corridor_id(modes_of[edge], edge), lengths[edge], False)

    stats = _corridor_stats(graphs, corridors, working_crs)
    stats.update(
        {
            "anchor_mode": anchor_mode,
            "n_edges": int(len(edges)),
            "n_anchors": int(len(anchors)),
            "n_excluded": int(edges["excluded"].sum()),
            "n_matched_to_anchor": n_matched,
            "corridor_match_rate": (n_matched / len(matchable)) if len(matchable) else 0.0,
        }
    )
    return StreetSpace(corridors=corridors, stats=stats, params=params)


def _midpoint(geom: LineString) -> Point:
    return geom.interpolate(0.5, normalized=True)


def _match_to_anchors(
    matchable: gpd.GeoDataFrame,
    anchors: gpd.GeoDataFrame,
    cfg: StreetSpaceConfig,
    working_crs: str,
):
    """Yield ``(edge, anchor_edge)`` for every edge that joins a corridor.

    Deliberately **not** ``sjoin_nearest``: the nearest anchor to a sidewalk's
    midpoint is very often the *cross* street at the corner, which fails the
    bearing test — and a nearest-join would then give up rather than fall through
    to the parallel street a metre further away. Measured on Philadelphia, that
    single difference moves the overall match rate from 48 % to 72 %, and
    ``footway`` edges specifically from roughly half to 80 %.

    So: every anchor within ``tolerance_m`` is a candidate (a buffered-point
    ``sjoin``), candidates whose orientation disagrees by more than
    ``max_bearing_deg`` are dropped, and the nearest survivor wins.
    """
    import numpy as np
    import shapely

    probe_points = np.array([_midpoint(g) for g in matchable.geometry], dtype=object)
    probes = gpd.GeoDataFrame(
        {
            "edge": matchable["edge"].to_numpy(),
            "bearing": matchable["bearing"].to_numpy(),
            "probe": probe_points,
        },
        geometry=shapely.buffer(probe_points, cfg.tolerance_m),
        crs=working_crs,
    )
    anchor_ref = gpd.GeoDataFrame(
        {
            "anchor_edge": anchors["edge"].to_numpy(),
            "anchor_bearing": anchors["bearing"].to_numpy(),
            "anchor_geom": anchors.geometry.to_numpy(),
        },
        geometry=anchors.geometry.to_numpy(),
        crs=working_crs,
    )

    pairs = gpd.sjoin(probes, anchor_ref, how="inner", predicate="intersects")
    if pairs.empty:
        return
    delta = np.abs(pairs["bearing"].to_numpy() - pairs["anchor_bearing"].to_numpy()) % 180.0
    pairs = pairs.loc[np.minimum(delta, 180.0 - delta) <= cfg.max_bearing_deg]
    if pairs.empty:
        return
    pairs = pairs.assign(dist=shapely.distance(pairs["probe"].to_numpy(), pairs["anchor_geom"].to_numpy()))
    best = pairs.sort_values("dist", kind="stable").drop_duplicates(subset="edge", keep="first")
    yield from zip(best["edge"], best["anchor_edge"], strict=True)


def _corridor_stats(
    graphs: GraphRegistry,
    corridors: dict[EdgeId, CorridorInfo],
    working_crs: str,
) -> dict[str, Any]:
    """Which corridors are shared by which modes — the per-pair audit.

    Built from *graph membership*, not from deposited mass: it answers "could
    these two modes ever meet here", which is the question the id seam broke. The
    mass-side question is answered by
    :func:`segx.metrics.cross_mode_exposure.cross_mode_combination_counts`.
    """
    per_corridor: dict[str, set[str]] = {}
    for mode in graphs.modes():
        graph = graphs.get(mode)
        for u, v, key in graph.edges(keys=True):
            info = corridors.get(canonical_edge((int(u), int(v), int(key))))
            if info is not None:
                per_corridor.setdefault(info.corridor_id, set()).add(mode)

    pair_counts: dict[str, int] = {}
    modes = graphs.modes()
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            key = "|".join(sorted((a, b)))
            pair_counts[key] = sum(1 for present in per_corridor.values() if a in present and b in present)
    return {
        "n_corridors": len(per_corridor),
        "n_corridors_multimode": sum(1 for present in per_corridor.values() if len(present) > 1),
        "n_corridors_own": sum(1 for info in corridors.values() if not info.matched),
        "corridors_per_mode_pair": pair_counts,
    }
