r"""The merged walk ⊕ public-transport graph — the heart of P7.

One static ``MultiDiGraph`` weighted in ``travel_time_min``, containing the walk
street network *and* the transit network wired into it::

    street node --connector(walk time)--------------------> platform
    platform    --board(min(headway/2, max_wait) + board_penalty)--> route-stop
    route-stop  --ride(mean inter-stop time)--------------> route-stop
    route-stop  --alight(0)-------------------------------> platform
    platform    --transfer(walk time + transfer_penalty)---> platform
    platform    --connector(walk time)--------------------> street node

The decisive property: **a plain
Dijkstra over this graph produces walk → ride → transfer → ride → walk by
itself.** Nobody writes transfer logic; the shortest path takes a bus only when
the bus genuinely beats walking *including the expected wait*, and transfers only
when transferring beats staying on. Everything downstream — the ``RoutingEngine``
protocol and both its implementations, ``routing/_paths.py`` edge decoding, all
three strategies, ``routing/route_set.py``, ``EdgeAccumulator`` and every metric
plugin — therefore works unchanged.

Two structural details that are load-bearing:

* **Route-stop nodes, not bare stop nodes.** Nodes are per ``(pattern, seq)``.
  Staying on a line traverses only ``ride`` edges; changing lines forces
  ``alight → (transfer) → board`` and so picks up the penalty. With bare stop
  nodes a line change would be free and every result optimistic.
* **Synthetic nodes get negative integer ids.** OSM node ids are large positives,
  so a collision is impossible without knowing the graph's id range. (Verified
  against pandana, which the design flagged as a risk in D6: its contraction
  hierarchies index and route through negative ids correctly.)

**Where a rider's mass lands** (D4) — every edge carries ``segx_layer``:

======================  =====================================================
``segx_layer``          edges
======================  =====================================================
``street``              walk edges, and surface (bus/tram) rides *after*
                        :func:`expand_transit_rides` splices them onto the
                        street segments the vehicle actually drives down
``rail_line``           metro / rail / ferry / monorail rides — their own
                        output layer, because a passenger in a tunnel is
                        co-present with nobody on the avenue above
``access``              connector / board / alight / transfer — weight-bearing
                        only, never carries demographic mass
======================  =====================================================

Surface rides are map-matched at **build** time: each consecutive stop→stop hop
is routed once on the ``drive`` graph and the resulting street ``EdgeId`` sequence
is stored on the ride edge as ``street_path``. The ride stays **one** edge for
routing speed, and :func:`expand_transit_rides` splices the street path in before
aggregation. Note the seam: ``street_path`` edge ids come from the *drive* graph
while the merged graph's own street edges come from the *walk* graph. Both are
OSM-derived and share node ids, and the whole transit result lives in its own
``mode="transit"`` partition, so the two are self-consistent — but a writer
joining geometry must look a transit street edge up in both graphs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import networkx as nx

from segx.errors import GraphError, TransitDataMissingError
from segx.routing._paths import as_multidigraph

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    import geopandas as gpd

    from segx.config import Settings
    from segx.io.boundary import LoadedBoundary
    from segx.io.gtfs import LoadedGTFS
    from segx.routing.trajectories import EdgeId

#: Which output layer an edge deposits demographic mass on.
EdgeLayer = Literal["street", "rail_line", "access"]

#: The edge families of the merged graph (``segx_edge`` attribute). Street edges
#: inherited from the walk graph carry ``"street"``.
TransitEdgeKind = Literal["street", "connector", "board", "ride", "alight", "transfer"]

#: Edge families that are pure impedance — they move a rider between
#: representations of the same place and must never receive demographic mass.
ACCESS_EDGE_KINDS: frozenset[str] = frozenset({"connector", "board", "alight", "transfer"})


# --------------------------------------------------------------------------- #
# Synthetic node ids                                                           #
# --------------------------------------------------------------------------- #


class _NodeIds:
    """Hands out negative integer ids for the synthetic transit nodes."""

    def __init__(self) -> None:
        self._next = -1

    def take(self) -> int:
        nid = self._next
        self._next -= 1
        return nid


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #


def board_cost_min(headway_min: float, settings: Settings) -> float:
    """Expected wait + boarding penalty, in minutes.

    Half the headway is the expected wait of a passenger arriving at a random
    time — the standard accessibility assumption, and the *right* one here
    precisely because the OD data carries no departure times (D3). It is capped at
    ``transit.max_wait_min`` so a once-an-hour line does not model a 30-minute
    wait as if it were certain.
    """
    transit = settings.transit
    return min(headway_min / 2.0, transit.max_wait_min) + transit.board_penalty_min


def _walk_minutes(metres: float, settings: Settings) -> float:
    return metres * 60.0 / (settings.network.walk_speed_kmh * 1000.0)


def _snap_stops(
    stops: gpd.GeoDataFrame,
    graph: nx.Graph,
    settings: Settings,
    *,
    working_crs: str,
    max_distance_m: float,
) -> tuple[dict[str, tuple[int, float]], list[str]]:
    """``stop_id -> (graph_node, distance_m)`` plus the ids that snapped too far.

    Reuses :func:`segx.network.snapping.snap_points`, so platforms inherit the
    same snap-distance audit as origins and POIs instead of
    growing a parallel nearest-node implementation.
    """
    from segx.network.snapping import snap_points

    snapped = snap_points(stops[["stop_id", "geometry"]], graph, settings, working_crs=working_crs)
    resolved: dict[str, tuple[int, float]] = {}
    dropped: list[str] = []
    for row in snapped.gdf.itertuples(index=False):
        distance = float(row.snap_distance_m)
        if distance > max_distance_m:
            dropped.append(str(row.stop_id))
            continue
        resolved[str(row.stop_id)] = (int(row.node_id), distance)
    return resolved, dropped


def _add_edge(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    *,
    kind: TransitEdgeKind,
    layer: EdgeLayer,
    minutes: float,
    length_m: float = 0.0,
    **attrs: object,
) -> EdgeId:
    key = graph.add_edge(
        u,
        v,
        travel_time_min=float(minutes),
        length=float(length_m),
        segx_edge=kind,
        segx_layer=layer,
        **attrs,
    )
    return (int(u), int(v), int(key))


# --------------------------------------------------------------------------- #
# Surface-ride map matching                                                    #
# --------------------------------------------------------------------------- #


def map_match_hops(
    hops: list[tuple[str, str]],
    stops: gpd.GeoDataFrame,
    drive_graph: nx.Graph,
    settings: Settings,
    *,
    working_crs: str,
) -> dict[tuple[str, str], list[EdgeId]]:
    """Route every distinct surface stop→stop hop once over the ``drive`` graph.

    A bus rides *along a street*, so its passengers are genuinely co-present with
    everyone else on that avenue — the exact "experienced segregation" the engine
    measures. Routing consecutive stops on the drive network approximates the
    vehicle's path well and reuses the existing engine verbatim; matching
    ``shapes.txt`` geometry would be more faithful but is a substantially harder
    problem (a possible refinement).

    Hops whose endpoints are unroutable are simply absent from the result; the
    caller then leaves that ride on its own line layer rather than inventing a
    street path.
    """
    if not hops:
        return {}

    stop_nodes, _ = _snap_stops(
        stops,
        drive_graph,
        settings,
        working_crs=working_crs,
        max_distance_m=settings.transit.max_access_distance_m,
    )
    if settings.routing.engine == "networkx":
        from segx.routing.networkx_engine import NetworkXEngine

        engine = NetworkXEngine({"drive": drive_graph}, settings)
    else:
        from segx.routing.pandana_engine import PandanaEngine

        engine = PandanaEngine({"drive": drive_graph}, settings)

    out: dict[tuple[str, str], list[EdgeId]] = {}
    for from_stop, to_stop in hops:
        src, dst = stop_nodes.get(from_stop), stop_nodes.get(to_stop)
        if src is None or dst is None:
            continue
        if src[0] == dst[0]:
            continue  # both stops snap to the same street node — nothing to traverse
        result = engine.shortest_path(src[0], dst[0], "drive", weight="travel_time_min")
        if result is not None and result.path:
            out[(from_stop, to_stop)] = list(result.path)
    return out


# --------------------------------------------------------------------------- #
# Feed source resolution                                                       #
# --------------------------------------------------------------------------- #


def load_transit_feed(
    boundary: LoadedBoundary,
    settings: Settings,
    *,
    cache_dir: Path | None = None,
) -> LoadedGTFS:
    """Resolve ``transit.source`` and return the feed the graph will be built from.

    ``"auto"`` (the default) uses GTFS when a feed path/URL is configured and
    falls back to OSM otherwise, because there is **no global GTFS download API**
    — feeds are per-agency and many cities publish none (D1). The two branches
    return the same :class:`~segx.io.gtfs.LoadedGTFS` shape, differing only in the
    ``source`` stamp that marks an OSM run's headways as assumed.
    """
    resolved = settings.transit.resolved_source()
    if resolved == "gtfs":
        from segx.io.gtfs import load_gtfs

        return load_gtfs(boundary, settings, cache_dir=cache_dir)
    from segx.network.transit_osm import fetch_transit_osm

    return fetch_transit_osm(boundary, settings, cache_dir=cache_dir)


# --------------------------------------------------------------------------- #
# The builder                                                                  #
# --------------------------------------------------------------------------- #


def build_transit_graph(
    feed: LoadedGTFS,
    walk_graph: nx.Graph,
    drive_graph: nx.Graph | None,
    settings: Settings,
    *,
    working_crs: str,
) -> nx.MultiDiGraph:
    """Merge *feed* into a copy of *walk_graph* and return the routable result.

    The walk graph is copied, never mutated — the registry keeps the plain
    ``walk`` graph alive alongside this one, and a run may legitimately route both.

    Build provenance (feed source, node/edge counts, dropped stops) is recorded
    under ``graph.graph["transit"]`` so it can propagate into output metadata:
    an ``osm_assumed`` run must be *labelled*, because a headway we invented and a
    headway we measured look identical in a GeoPackage (D1).
    """
    transit = settings.transit
    if feed.n_patterns == 0:
        raise TransitDataMissingError(
            "cannot build a transit graph: the feed has no patterns in the study area",
            details={"source": feed.source, "n_stops": feed.n_stops},
        )
    if feed.crs != working_crs:
        raise GraphError(
            f"transit feed is in {feed.crs} but the run's working CRS is {working_crs}",
            details={"feed_crs": feed.crs, "working_crs": working_crs},
        )

    graph: nx.MultiDiGraph = as_multidigraph(walk_graph).copy()
    graph.graph["crs"] = working_crs
    for *_, data in graph.edges(data=True):
        data.setdefault("segx_edge", "street")
        data["segx_layer"] = "street"

    ids = _NodeIds()
    stop_geometry = feed.stops.set_index("stop_id").geometry

    # ---- platforms + connectors ------------------------------------------- #
    reachable, unreachable = _snap_stops(
        feed.stops,
        walk_graph,
        settings,
        working_crs=working_crs,
        max_distance_m=transit.max_access_distance_m,
    )
    platform_node: dict[str, int] = {}
    for stop_id, (street_node, distance_m) in reachable.items():
        point = stop_geometry.loc[stop_id]
        node = ids.take()
        graph.add_node(
            node,
            x=float(point.x),
            y=float(point.y),
            segx_node="platform",
            stop_id=stop_id,
        )
        platform_node[stop_id] = node
        minutes = _walk_minutes(distance_m, settings)
        _add_edge(graph, street_node, node, kind="connector", layer="access", minutes=minutes, length_m=distance_m)
        _add_edge(graph, node, street_node, kind="connector", layer="access", minutes=minutes, length_m=distance_m)

    if not platform_node:
        raise TransitDataMissingError(
            "no transit stop is within transit.max_access_distance_m of the walk network",
            details={"max_access_distance_m": transit.max_access_distance_m, "n_stops": feed.n_stops},
        )

    # ---- surface hops to map-match ---------------------------------------- #
    pattern_meta = feed.patterns.set_index("pattern_id")
    ordered = feed.pattern_stops.sort_values(["pattern_id", "seq"])
    surface_hops: set[tuple[str, str]] = set()
    for pattern_id, group in ordered.groupby("pattern_id", sort=False):
        if pattern_id not in pattern_meta.index:
            continue
        route_type = int(pattern_meta.loc[pattern_id, "route_type"])
        if transit.layer_for_route_type(route_type) != "street":
            continue
        sequence = [str(s) for s in group["stop_id"]]
        surface_hops.update(zip(sequence, sequence[1:], strict=False))

    street_paths: dict[tuple[str, str], list[EdgeId]] = {}
    if transit.map_match_surface_rides and drive_graph is not None:
        street_paths = map_match_hops(sorted(surface_hops), feed.stops, drive_graph, settings, working_crs=working_crs)

    # ---- route-stop nodes, board / ride / alight edges --------------------- #
    n_rides = n_unmatched_surface = 0
    for pattern_id, group in ordered.groupby("pattern_id", sort=False):
        if pattern_id not in pattern_meta.index:
            continue
        meta = pattern_meta.loc[pattern_id]
        route_type = int(meta["route_type"])
        nominal_layer = transit.layer_for_route_type(route_type)
        board_minutes = board_cost_min(float(meta["headway_min"]), settings)

        # Stops whose platform was dropped (too far from the walk network) are
        # skipped, but their share of the running time is carried onto the next
        # kept hop — dropping it too would make the line look faster than it is.
        rows: list[tuple[str, float]] = []
        carried = 0.0
        for row in group.itertuples(index=False):
            minutes = float(row.travel_time_min)
            carried += minutes if math.isfinite(minutes) and minutes > 0 else 0.0
            if str(row.stop_id) in platform_node:
                rows.append((str(row.stop_id), 0.0 if not rows else carried))
                carried = 0.0
        if len(rows) < 2:
            continue  # a pattern reduced to one reachable stop carries no ride

        route_stop: list[int] = []
        for stop_id, _ in rows:
            point = stop_geometry.loc[stop_id]
            node = ids.take()
            graph.add_node(
                node,
                x=float(point.x),
                y=float(point.y),
                segx_node="route_stop",
                stop_id=stop_id,
                pattern_id=str(pattern_id),
                route_id=str(meta["route_id"]),
                route_type=route_type,
            )
            route_stop.append(node)
            platform = platform_node[stop_id]
            _add_edge(
                graph,
                platform,
                node,
                kind="board",
                layer="access",
                minutes=board_minutes,
                headway_min=float(meta["headway_min"]),
            )
            _add_edge(graph, node, platform, kind="alight", layer="access", minutes=0.0)

        for index in range(len(rows) - 1):
            from_stop, (to_stop, minutes) = rows[index][0], rows[index + 1]
            metres = float(stop_geometry.loc[from_stop].distance(stop_geometry.loc[to_stop]))
            path = street_paths.get((from_stop, to_stop))
            if nominal_layer == "street" and not path:
                # A surface ride we could not put on the street keeps its own
                # identity rather than being silently dropped or, worse, credited
                # to street segments it was never matched to.
                layer: EdgeLayer = "rail_line"
                n_unmatched_surface += 1
            else:
                layer = nominal_layer
            _add_edge(
                graph,
                route_stop[index],
                route_stop[index + 1],
                kind="ride",
                layer=layer,
                minutes=minutes,
                length_m=metres,
                pattern_id=str(pattern_id),
                route_type=route_type,
                from_stop=from_stop,
                to_stop=to_stop,
                street_path=list(path) if path else None,
                surface_unmatched=(nominal_layer == "street" and not path),
            )
            n_rides += 1

    # ---- transfers --------------------------------------------------------- #
    n_transfers = _add_transfers(graph, feed, platform_node, settings)

    graph.graph["transit"] = {
        **feed.metadata(),
        "n_platforms": len(platform_node),
        "n_rides": n_rides,
        "n_transfers": n_transfers,
        "n_stops_unreachable": len(unreachable),
        "stops_unreachable": sorted(unreachable),
        "n_surface_rides_unmatched": n_unmatched_surface,
        "map_matched": bool(street_paths),
    }
    return graph


def _add_transfers(
    graph: nx.MultiDiGraph,
    feed: LoadedGTFS,
    platform_node: dict[str, int],
    settings: Settings,
) -> int:
    """Walk links between distinct platforms within ``max_transfer_distance_m``.

    A same-platform line change needs no transfer edge: it already costs
    ``alight(0) → board(wait + board_penalty)``. These edges are for stepping
    across the street or between the platforms of one station.
    """
    from shapely import STRtree

    radius = settings.transit.max_transfer_distance_m
    penalty = settings.transit.transfer_penalty_min
    served = feed.stops[feed.stops["stop_id"].isin(platform_node)].reset_index(drop=True)
    if len(served) < 2:
        return 0

    geometries = list(served.geometry)
    stop_ids = [str(s) for s in served["stop_id"]]
    tree = STRtree(geometries)
    n_added = 0
    for i, geometry in enumerate(geometries):
        for j in tree.query(geometry.buffer(radius), predicate="intersects"):
            if int(j) == i:
                continue
            metres = float(geometry.distance(geometries[int(j)]))
            if metres > radius:
                continue
            minutes = _walk_minutes(metres, settings) + penalty
            _add_edge(
                graph,
                platform_node[stop_ids[i]],
                platform_node[stop_ids[int(j)]],
                kind="transfer",
                layer="access",
                minutes=minutes,
                length_m=metres,
            )
            n_added += 1
    return n_added


# --------------------------------------------------------------------------- #
# Consumption helpers                                                          #
# --------------------------------------------------------------------------- #


def edge_layers(graph: nx.MultiDiGraph) -> dict[EdgeId, str]:
    """``EdgeId -> segx_layer`` for the **non-street** edges only.

    Street edges are the overwhelming majority and default to ``"street"`` in
    :class:`~segx.aggregation.edge_accumulator.EdgeAccumulator`, so returning only
    the minority keeps the map small — a city walk graph has ~10⁵ edges and a
    transit network ~10³ rides.
    """
    out: dict[EdgeId, str] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        layer = data.get("segx_layer", "street")
        if layer != "street":
            out[(int(u), int(v), int(key))] = str(layer)
    return out


def expand_transit_rides(path: list[EdgeId], graph: nx.MultiDiGraph) -> tuple[list[EdgeId], list[str]]:
    """Rewrite a routed path into the edges that receive mass, **and label each
    edge with the mode the traveller is actually in**.

    Two rewrites, both required before aggregation:

    * a **map-matched surface ride** is replaced by its ``street_path`` — the
      street segments the vehicle drives down — so a bus rider deposits mass on
      the real avenue (D4);
    * an **access** edge (connector / board / alight / transfer) is dropped: it
      moves the rider between representations of one place and carries no
      co-presence.

    Rail rides pass through untouched, keeping their own identity so their mass
    lands on the ``rail_line`` layer. A path with no transit edges is returned
    unchanged, so running this over walk/drive trajectories is a safe no-op.

    **The labels (P7.2).** A public-transport trip walks to the stop, rides, and
    walks again; before P7.2 all of that was filed under ``mode="transit"``, so
    "bus passenger" silently meant "bus passenger *plus* the same passenger on the
    sidewalk", and both ``transit × drive`` and ``transit × walk`` came out
    inflated by pedestrians. Here a ``ride`` leg — surface or rail — is labelled
    ``transit`` and every remaining street leg is labelled ``walk``. ``transit``
    therefore means *inside the vehicle*, which is what
    ``metrics.cross_mode.factor`` discounts.

    Returns ``(edges, leg_modes)`` with ``len(leg_modes) == len(edges)``, ready
    for :attr:`segx.routing.trajectories.Trajectory.leg_modes`.
    """
    if not path:
        return list(path), []
    out: list[EdgeId] = []
    modes: list[str] = []
    for edge in path:
        u, v, key = edge
        data = graph.get_edge_data(u, v, key)
        if data is None:
            out.append(edge)
            modes.append("walk")
            continue
        kind = data.get("segx_edge", "street")
        if kind in ACCESS_EDGE_KINDS:
            continue
        street_path = data.get("street_path")
        if kind == "ride" and street_path:
            out.extend((int(a), int(b), int(k)) for a, b, k in street_path)
            modes.extend("transit" for _ in street_path)
            continue
        out.append(edge)
        modes.append("transit" if kind == "ride" else "walk")
    return out, modes


def has_transit(graph: nx.Graph) -> bool:
    """True if *graph* was built by :func:`build_transit_graph`."""
    return "transit" in graph.graph


def street_nodes(graph: nx.Graph) -> set[int] | None:
    """The nodes a real-world point may snap to, or ``None`` for "all of them".

    On a merged transit graph the synthetic platform and route-stop nodes sit at
    genuine coordinates, so a nearby origin would happily snap onto one — and an
    origin snapped to a *route-stop* would board without ever paying the wait.
    Restricting the snap to street nodes keeps the transit layer reachable only
    through connector edges, which is what makes the frequency model honest.
    Returns ``None`` for an ordinary walk/drive graph so callers can pass the
    result straight to :func:`segx.network.snapping.snap_points` unconditionally.
    """
    if not has_transit(graph):
        return None
    return {n for n, data in graph.nodes(data=True) if "segx_node" not in data}


__all__ = [
    "ACCESS_EDGE_KINDS",
    "EdgeLayer",
    "TransitEdgeKind",
    "board_cost_min",
    "build_transit_graph",
    "edge_layers",
    "expand_transit_rides",
    "has_transit",
    "load_transit_feed",
    "map_match_hops",
    "street_nodes",
]
