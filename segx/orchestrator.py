"""Pipeline orchestration: ingest → network → choose origin source → route
(three strategies) → aggregate → metrics → export.

P1 wires the **ingest** stages end-to-end: the boundary loader resolves the
working CRS, demographics and OD pairs reproject into it, per-mode street
graphs are built (live OSMnx + GeoParquet cache), POIs are extracted (cached
GeoParquet), and origins / POIs / OD endpoints are snapped to their mode
graph with an audit trail. P2–P7 add origins, routing, aggregation and the
metric plugins; P8 adds the ``write_outputs`` stage (:func:`write`), so
:func:`run_pipeline` now runs end to end and returns a :class:`RunResult`.

The orchestrator deliberately keeps the per-stage functions importable on
their own so the integration tests can stitch a custom subset together
(e.g. unit-level "ingest with cached OSM only"). The CLI's ``segx run``
calls :func:`run_pipeline`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd

from segx.aggregation.edge_accumulator import EdgeAccumulator
from segx.aggregation.path_accumulator import PathAccumulator
from segx.io.boundary import LoadedBoundary, load_boundary
from segx.io.demographics import LoadedDemographics, class_columns_for, load_demographics
from segx.io.od_pairs import LoadedODPairs, load_od_pairs
from segx.io.pois import LoadedPOIs, load_pois
from segx.io.writers import WrittenOutputs, write_outputs
from segx.metrics.registry import available_metrics, get_metric
from segx.network.builder import load_or_fetch_graph
from segx.network.registry import GraphRegistry
from segx.network.snapping import snap_points
from segx.network.transit import street_nodes
from segx.population.attributes import attach_attributes
from segx.population.origins import build_origins
from segx.progress import report
from segx.routing.inputs import RoutableODPair, RoutableOrigin, RoutablePOI
from segx.routing.nearest_poi import nearest_poi_trajectories
from segx.routing.od_router import od_pair_trajectories
from segx.routing.poi_catchment import catchment_trajectories
from segx.validation.invariants import check_weight_conservation

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from segx.aggregation.edge_accumulator import DemoVector
    from segx.config import Settings
    from segx.progress import CancelCheck, ProgressCallback
    from segx.routing.engine import RoutingEngine
    from segx.routing.trajectories import Trajectory
    from segx.validation.invariants import WeightLedger

# Importing the metrics package registers every metric plugin (invariant #5):
# the P4 ``diversity`` + refining measures (``experienced_segregation``,
# ``intergroup_contact``), the P5 measures (``exposure``,
# ``segregated_trajectories``, ``dominant_trajectories``) and the P7.1
# ``cross_mode_exposure``. The orchestrator resolves them by name from the
# registry — never hardcoding a formula.
import segx.metrics  # noqa: E402,F401  (import for registration side-effect)

#: OD ``mode`` values normalise to the network's per-mode graph keys.
_OD_MODE_ALIAS: dict[str, str] = {
    "car": "drive",
    "driving": "drive",
    "auto": "drive",
    "foot": "walk",
    "walking": "walk",
    # P7: an OD file whose `mode` column says `bus` routes on the merged transit
    # graph with no code change — the alias table already existed for `car`.
    "bus": "transit",
    "metro": "transit",
    "subway": "transit",
    "rail": "transit",
    "train": "transit",
    "tram": "transit",
    "pt": "transit",
    "public_transport": "transit",
    "transporte_publico": "transit",
    "onibus": "transit",
}

#: Ordered pipeline stages, for documentation and progress reporting.
PIPELINE_STAGES: tuple[str, ...] = (
    "load_boundary",  # io.boundary            (P1)
    "load_demographics",  # io.demographics    (P1)
    "build_networks",  # network.builder + network.cache  (P1)
    "load_pois",  # io.pois                    (P1, item 2.3)
    "load_od_pairs",  # io.od_pairs (optional) (P1)
    "analyse_od_spread",  # analysis.od_spread: destinations per class, pre-routing  (P7.5)
    "build_origins",  # population.origins: OD origins, else polygon centroids  (P2)
    "attach_attributes",  # population.attributes  (P2)
    "snap_to_network",  # network.snapping     (P1 module; consumed in P3)
    "route_nearest",  # routing.nearest_poi    (P3, binding A)
    "route_catchment",  # routing.poi_catchment  (P3, binding B)
    "route_od_pairs",  # routing.od_router      (P3, binding C; skipped if no OD pairs)
    "expand_transit_rides",  # network.transit: splice surface rides onto streets, drop access edges (P7)
    "resolve_street_space",  # network.street_space: one corridor id shared across modes (P7.2)
    "aggregate_edges",  # aggregation.edge_accumulator  (P4)
    "compute_metrics",  # metrics.registry: entropy (P4) + segregation metrics (P5)
    "check_invariants",  # validation.invariants  (P4)
    "write_outputs",  # io.writers: segments.gpkg + metrics.xlsx  (P8)
)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Everything the P1 ingest phase produces.

    Carries the resolved working CRS (via ``boundary.crs``) and the loaded
    artifacts; downstream phases (origins, routing) consume this object as
    their starting point.
    """

    boundary: LoadedBoundary
    demographics: LoadedDemographics
    graphs: GraphRegistry
    pois: LoadedPOIs
    od_pairs: LoadedODPairs | None


def ingest(
    settings: Settings,
    *,
    boundary_path: str | Path,
    demographics_path: str | Path,
    od_pairs_path: str | Path | None = None,
    cache_dir: Path | None = None,
    poi_fetcher: str = "auto",
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> IngestResult:
    """Run the five P1 ingest stages and return the consolidated state.

    Stages are run in the documented order so that the working CRS is fixed
    by the boundary load *before* anything else touches geometry. ``progress``
    and ``cancel`` are optional (see :mod:`segx.progress`): each stage announces
    itself and honours a cancellation request before starting work, so the
    embedding app can show progress and stop mid-run.
    """
    # Stages actually run by P1 ingest, in order (a subset of PIPELINE_STAGES).
    stages = ["load_boundary", "load_demographics", "build_networks", "load_pois"]
    if od_pairs_path is not None:
        stages.append("load_od_pairs")
    total = len(stages)

    def _stage(name: str, message: str = "") -> None:
        report(progress, cancel, stage=name, index=stages.index(name), total=total, message=message)

    _stage("load_boundary", str(boundary_path))
    boundary = load_boundary(boundary_path, settings)

    _stage("load_demographics", str(demographics_path))
    demographics = load_demographics(demographics_path, settings, working_crs=boundary.crs)

    _stage("build_networks", f"modes={settings.network.modes}")
    graphs = GraphRegistry()
    for mode in settings.network.modes:
        graphs.register(mode, load_or_fetch_graph(boundary, mode, settings, cache_dir, cancel=cancel))

    _stage("load_pois", f"fetcher={poi_fetcher}")
    pois = load_pois(boundary, settings, cache_dir=cache_dir, fetcher=poi_fetcher, cancel=cancel)

    od_pairs: LoadedODPairs | None = None
    if od_pairs_path is not None:
        _stage("load_od_pairs", str(od_pairs_path))
        od_pairs = load_od_pairs(od_pairs_path, settings, working_crs=boundary.crs, demographics=demographics)

    return IngestResult(
        boundary=boundary,
        demographics=demographics,
        graphs=graphs,
        pois=pois,
        od_pairs=od_pairs,
    )


def analyse_od(
    state: IngestResult,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
):
    """Run the P7.5 OD destination-spread analysis, or return ``None``.

    Returns ``None`` — never raises — when the run has no OD pairs, when the
    analysis is switched off, or when the OD file carries no per-class columns
    (the usual case: the class vector lives on the demographics file, where every
    flow out of an origin would carry the same composition and the measure would
    be a tautology). A *diagnostic of the input table* must not be able to fail an
    otherwise valid pipeline run; the reason is reported through ``progress``
    instead, and :func:`segx.analysis.od_spread.analyse_od_spread` raises for the
    caller who asked for the analysis by name (``segx od-report``).
    """
    from segx.analysis.od_spread import ODSpreadResult, analyse_od_spread, od_spread_unavailable_reason

    reason = od_spread_unavailable_reason(state.od_pairs, settings)
    report(progress, cancel, stage="analyse_od_spread", index=0, total=1, message=reason or "od destination spread")
    if reason is not None:
        return None
    assert state.od_pairs is not None  # od_spread_unavailable_reason already checked
    result: ODSpreadResult = analyse_od_spread(state.od_pairs, settings)
    return result


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """The P2 output: the attributed origin set ready for routing.

    ``origins`` columns: ``origin_id`` (str), ``geometry`` (working CRS),
    ``weight`` (float), ``demo_vector`` (per-class mass), ``demo_resolved``
    (bool) and ``demo_reason`` (str | None). See :mod:`segx.population`.
    """

    origins: gpd.GeoDataFrame


def build_population(
    state: IngestResult,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> PopulationResult:
    """Run the two P2 stages: choose the origin source, then attach attributes."""
    stages = ["build_origins", "attach_attributes"]
    total = len(stages)

    def _stage(name: str, message: str = "") -> None:
        report(progress, cancel, stage=name, index=stages.index(name), total=total, message=message)

    _stage("build_origins", "od_origins" if state.od_pairs is not None else f"{state.demographics.kind}_centroids")
    origins = build_origins(state.demographics, state.od_pairs, settings)

    _stage("attach_attributes", f"source={settings.demographics.attribute_source}")
    origins = attach_attributes(origins, state.demographics, state.od_pairs, settings)

    return PopulationResult(origins=origins)


def _make_engine(graphs: GraphRegistry, settings: Settings) -> RoutingEngine:
    """Construct the configured routing engine over the registered mode graphs.

    Engines are imported lazily so a ``networkx``-only run need not import
    pandana (and vice versa). Default is pandana (``settings.routing.engine``).
    """
    graphs_by_mode = {mode: graphs.get(mode) for mode in settings.network.modes}
    if settings.routing.engine == "networkx":
        from segx.routing.networkx_engine import NetworkXEngine

        return NetworkXEngine(graphs_by_mode, settings)
    from segx.routing.pandana_engine import PandanaEngine

    return PandanaEngine(graphs_by_mode, settings)


def _routable_origins(
    origins: gpd.GeoDataFrame, graph, settings: Settings, *, working_crs: str
) -> tuple[list[RoutableOrigin], int]:
    """Snap origins to *graph* and return the node-level records + over-tolerance count."""
    snapped = snap_points(
        origins[["origin_id", "weight", "geometry"]],
        graph,
        settings,
        working_crs=working_crs,
        node_subset=street_nodes(graph),
    )
    out = [
        RoutableOrigin(str(row.origin_id), int(row.node_id), float(row.weight))
        for row in snapped.gdf.itertuples(index=False)
    ]
    return out, snapped.n_over_tolerance


def _routable_pois(
    pois: gpd.GeoDataFrame, graph, settings: Settings, *, working_crs: str
) -> tuple[dict[str, list[RoutablePOI]], int]:
    """Snap POIs to *graph* and group them by category."""
    snapped = snap_points(
        pois[["category", "osm_id", "geometry"]],
        graph,
        settings,
        working_crs=working_crs,
        node_subset=street_nodes(graph),
    )
    by_category: dict[str, list[RoutablePOI]] = {}
    for idx, row in snapped.gdf.iterrows():
        category = str(row["category"])
        poi_id = row["osm_id"] if row.get("osm_id") is not None else f"{category}_{idx}"
        by_category.setdefault(category, []).append(RoutablePOI(str(poi_id), int(row["node_id"])))
    return by_category, snapped.n_over_tolerance


def _routable_od_pairs(
    od_pairs: gpd.GeoDataFrame, graphs: GraphRegistry, settings: Settings, *, working_crs: str
) -> list[RoutableODPair]:
    """Snap each OD endpoint to *its pair's* mode graph; one record per pair.

    OD ``mode`` values are normalised to the network's graph keys (e.g. ``car`` →
    ``drive``) via :data:`_OD_MODE_ALIAS`; an unknown/unregistered mode raises a
    typed :class:`~segx.errors.ModeNotRegisteredError` via ``GraphRegistry.get``.
    """
    out: list[RoutableODPair] = []
    od = od_pairs.copy()
    od["_graph_mode"] = od["mode"].astype(str).map(lambda m: _OD_MODE_ALIAS.get(m, m))
    for graph_mode, sub in od.groupby("_graph_mode"):
        graph = graphs.get(str(graph_mode))
        origins_gdf = sub.set_geometry("geometry")[["geometry"]]
        dests_gdf = gpd.GeoDataFrame(geometry=sub["dest_geom"].to_numpy(), crs=working_crs)
        subset = street_nodes(graph)
        o = snap_points(origins_gdf, graph, settings, working_crs=working_crs, node_subset=subset)
        d = snap_points(dests_gdf, graph, settings, working_crs=working_crs, node_subset=subset)
        o_nodes, d_nodes = o.gdf["node_id"].to_numpy(), d.gdf["node_id"].to_numpy()
        for (idx, row), onode, dnode in zip(sub.iterrows(), o_nodes, d_nodes, strict=True):
            out.append(
                RoutableODPair(
                    origin_id=str(row["origin_id"]),
                    origin_node=int(onode),
                    dest_id=f"od{idx}",
                    dest_node=int(dnode),
                    mode=str(graph_mode),
                    category=str(row["category"]),
                    weight=float(row["weight"]),
                )
            )
    return out


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """The P3 output: the unified trajectory stream plus reachability stats.

    ``trajectories`` is the chained nearest + catchment + OD stream (each tagged
    ``(mode, category, binding)``). ``stats`` summarises per-binding reachability
    (counts of reachable vs. unreachable trajectories, by reason) so the
    embedding app / invariants can report it without rescanning the stream.
    """

    trajectories: list[Trajectory]
    stats: dict


def _summarise(trajectories: list[Trajectory]) -> dict:
    """Per-binding reachability counts."""
    stats: dict[str, dict] = {}
    for traj in trajectories:
        b = stats.setdefault(traj.binding, {"total": 0, "reachable": 0, "unreachable": 0, "reasons": {}})
        b["total"] += 1
        if traj.reachable:
            b["reachable"] += 1
        else:
            b["unreachable"] += 1
            reason = traj.reason or "unreachable"
            b["reasons"][reason] = b["reasons"].get(reason, 0) + 1
    return stats


def route(
    state: IngestResult,
    population: PopulationResult,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> RoutingResult:
    """Run the three P3 routing strategies into one unified ``Trajectory`` stream.

    Origins and POIs are snapped to the access-mode graph (``settings.routing.
    access_mode``); OD endpoints are snapped to each pair's own mode graph.
    Over-tolerance snaps are *flagged* in ``stats``, not dropped. Bindings:
    A = nearest POI, B = catchment, C = OD pairs (only when OD pairs were supplied).
    """
    stages = ["snap_to_network", "route_nearest", "route_catchment"]
    if state.od_pairs is not None:
        stages.append("route_od_pairs")
    total = len(stages)

    def _stage(name: str, message: str = "") -> None:
        report(progress, cancel, stage=name, index=stages.index(name), total=total, message=message)

    working_crs = state.boundary.crs
    access_mode = settings.routing.access_mode
    access_graph = state.graphs.get(access_mode)

    _stage("snap_to_network", f"access_mode={access_mode}, engine={settings.routing.engine}")
    routable_origins, origins_over_tol = _routable_origins(
        population.origins, access_graph, settings, working_crs=working_crs
    )
    pois_by_category, pois_over_tol = _routable_pois(state.pois.gdf, access_graph, settings, working_crs=working_crs)
    routable_pairs = (
        _routable_od_pairs(state.od_pairs.gdf, state.graphs, settings, working_crs=working_crs)
        if state.od_pairs is not None
        else []
    )

    engine = _make_engine(state.graphs, settings)
    trajectories: list[Trajectory] = []

    _stage("route_nearest")
    trajectories.extend(
        nearest_poi_trajectories(routable_origins, pois_by_category, engine, settings, mode=access_mode)
    )

    _stage("route_catchment")
    trajectories.extend(catchment_trajectories(routable_origins, pois_by_category, engine, settings, mode=access_mode))

    if state.od_pairs is not None:
        _stage("route_od_pairs", f"{len(routable_pairs)} pairs")
        # The route-set (P4.5, k>1) needs the mode graphs directly (NetworkX
        # link-penalty / simple-path enumeration); k==1 uses the fast engine.
        od_graphs = {mode: state.graphs.get(mode) for mode in settings.network.modes}
        trajectories.extend(od_pair_trajectories(routable_pairs, engine, settings, graphs=od_graphs))

    stats = _summarise(trajectories)
    stats["snapping"] = {
        "origins_over_tolerance": origins_over_tol,
        "pois_over_tolerance": pois_over_tol,
        "tolerance_m": settings.network.snap_tolerance_m,
    }
    return RoutingResult(trajectories=trajectories, stats=stats)


@dataclass(frozen=True, slots=True)
class AggregationResult:
    """The P4 output: the per-edge demographic table + per-segment metrics.

    ``edge_table`` is the tidy long frame (one row per ``(mode, category, binding,
    u, v, key, attribute, class)`` with ``mass`` + ``n_paths``); ``metrics`` maps a
    metric name to its per-segment frame and ``summaries`` to its network-wide
    indicator (``diversity``, ``experienced_segregation``, ``intergroup_contact``,
    ``exposure``, ``segregated_trajectories``, ``dominant_trajectories``);
    ``ledger`` is the weight-conservation bookkeeping; ``stats`` summarises counts,
    including ``stats["cross_mode"]`` — the coverage audit that says whether
    ``cross_mode_exposure`` actually had anything to join (P7.1) — and
    ``stats["fragmentation"]``, the equivalent for ``path_fragmentation``: how many
    paths were measurable, how many were too small to carry a fractal dimension,
    and under which method and scale ladder (P7.3).

    ``path_table`` (P7.3) is the second streamed aggregate: per-``(mode, category,
    binding, attribute, class)`` weighted sums of path shape, from
    :class:`~segx.aggregation.path_accumulator.PathAccumulator`. It is kept
    alongside ``edge_table`` rather than folded into it because its rows are
    *paths*, not segments — the two cannot share a grain.
    """

    edge_table: pd.DataFrame
    metrics: dict[str, pd.DataFrame]
    summaries: dict[str, pd.DataFrame]
    ledger: WeightLedger
    stats: dict
    path_table: pd.DataFrame | None = None


def _origin_vectors(population: PopulationResult) -> dict[str, DemoVector]:
    """``origin_id -> demo_vector`` from the attributed origin set."""
    cols = population.origins[["origin_id", "demo_vector"]]
    return {str(row.origin_id): row.demo_vector for row in cols.itertuples(index=False)}


def _od_flow_vectors(state: IngestResult, settings: Settings) -> dict[str, DemoVector] | None:
    """Per-flow ``dest_id -> demo_vector`` when class counts live on the OD file.

    In ``attribute_source == 'od'`` mode each commute flow carries its own per-class
    worker counts (LODES ``SE01/02/03``); the OD trajectory's ``destination_id`` is
    ``"od<idx>"`` for row ``idx`` of the OD frame, so we key by that. Returns
    ``None`` when the OD file has no class columns (the caller then falls back to
    the origin's vector), keeping a home block's workers from being multiplied
    across its destinations.
    """
    if state.od_pairs is None or settings.demographics.attribute_source != "od":
        return None
    df = state.od_pairs.gdf
    cols_by_attr = {a.name: class_columns_for(a) for a in settings.demographics.attributes}
    if any(c not in df.columns for cols in cols_by_attr.values() for c in cols):
        return None
    # Column-wise, not `iterrows`: one Series per row is most of the cost on a
    # city-scale OD file, and the values are the same floats either way.
    per_attr = [
        (
            attr.name,
            list(attr.class_labels),
            [df[col].astype(float).tolist() for col in cols_by_attr[attr.name]],
        )
        for attr in settings.demographics.attributes
    ]
    out: dict[str, DemoVector] = {}
    for row, idx in enumerate(df.index):
        out[f"od{idx}"] = {
            name: {label: values[row] for label, values in zip(labels, columns, strict=True)}
            for name, labels, columns in per_attr
        }
    return out


def _scale_vector(vector: DemoVector, w: float) -> DemoVector:
    """A copy of *vector* with every class mass multiplied by *w* (route-set split).

    ``NaN`` masses (unresolved origin) stay ``NaN`` (``NaN*w == NaN``), so the
    trajectory is still flagged unresolved, never zeroed (invariant #2).
    """
    return {attr: {cls: m * w for cls, m in classes.items()} for attr, classes in vector.items()}


def _demo_vector_for(
    trajectory: Trajectory,
    origin_vectors: dict[str, DemoVector],
    od_flow_vectors: dict[str, DemoVector] | None,
) -> DemoVector:
    """The per-class vector to deposit for *trajectory* (per-flow for OD, else
    origin), scaled by the route-set choice weight ``route_weight`` so a flow's
    mass splits across its *k* routes with ``Σ wᵢ = 1`` (P4.5)."""
    if od_flow_vectors is not None and trajectory.binding == "od":
        vector = od_flow_vectors.get(str(trajectory.destination_id))
        if vector is None:
            vector = origin_vectors[str(trajectory.origin_id)]
    else:
        vector = origin_vectors[str(trajectory.origin_id)]
    if trajectory.route_weight != 1.0:
        return _scale_vector(vector, trajectory.route_weight)
    return vector


def _node_coordinates(state: IngestResult) -> dict[int, tuple[float, float]]:
    """``node id -> (x, y)`` in the working CRS, pooled across every mode graph.

    What :class:`~segx.aggregation.path_accumulator.PathAccumulator` measures
    path shape on. Pooling is safe and necessary: ``expand_transit_rides`` rewrites
    a transit path onto ``drive``-graph street edges, so a single path's nodes can
    come from more than one graph, and OSM node ids are global — the same id means
    the same place in every extract. Synthetic transit nodes (negative ids) carry
    ``x``/``y`` too, so a metro ride is measurable like any other leg.

    Bounded by the node count, not the trajectory count, and built once per run.
    A node missing ``x``/``y`` is simply absent, which makes the paths through it
    unmeasurable rather than misplaced (invariant #2).
    """
    coords: dict[int, tuple[float, float]] = {}
    for mode in state.graphs.modes():
        for node, data in state.graphs.get(mode).nodes(data=True):
            x, y = data.get("x"), data.get("y")
            if x is not None and y is not None:
                coords[int(node)] = (float(x), float(y))
    return coords


def _edge_coordinates(state: IngestResult, node_xy: dict[int, tuple[float, float]]):
    """A lazy ``(u, v, key) -> vertex sequence`` resolver over every mode graph.

    Lets the fragmentation measure trace a curved street along its real geometry
    instead of the chord between its end nodes (P7.3). OSMnx sets ``geometry``
    only on edges simplified from a multi-node way, so a straight block resolves
    to ``None`` and the caller falls back to the chord. The sequence is returned
    **oriented u→v**: a stored geometry can run the other way (the reverse edge
    of a two-way street, or any edge of an ``undirected`` graph). Cached per
    traversed edge — bounded by the edge count, not the trajectory count.
    """
    graphs = [state.graphs.get(mode) for mode in state.graphs.modes()]
    cache: dict[tuple, list[tuple[float, float]] | None] = {}

    def resolve(edge):
        if edge in cache:
            return cache[edge]
        coords = None
        u, v, key = edge
        for graph in graphs:
            if not graph.has_edge(u, v, key):
                continue
            geom = graph.edges[u, v, key].get("geometry")
            if geom is None or getattr(geom, "geom_type", None) != "LineString" or geom.is_empty:
                continue
            coords = [(float(x), float(y)) for x, y, *_ in geom.coords]
            start = node_xy.get(int(u))
            if start is not None and math.dist(coords[-1], start) < math.dist(coords[0], start):
                coords.reverse()
            break
        cache[edge] = coords
        return coords

    return resolve


def _expand_transit(
    trajectories: list[Trajectory],
    state: IngestResult,
) -> tuple[list[Trajectory], dict | None]:
    """Rewrite transit paths for aggregation and return the edge→layer map (P7).

    A no-op when no transit graph is registered — the returned list is the input
    list and the map is ``None``, so walk/drive runs are untouched. Otherwise each
    trajectory routed on the transit graph has its path passed through
    :func:`segx.network.transit.expand_transit_rides`, which splices a map-matched
    surface ride onto the street segments the vehicle drives down and drops the
    access edges (connector / board / alight / transfer) that carry no
    co-presence. This runs **before** the accumulator, so the accumulator itself
    stays untouched in the hot path.

    It also fills in ``leg_modes`` (P7.2): a public-transport trip walks to the
    stop, rides, and walks again, and before P7.2 all three legs were filed under
    ``mode="transit"`` — so "bus passenger" silently meant "bus passenger plus the
    same passenger on the sidewalk", inflating both ``transit × drive`` and
    ``transit × walk``. Ride legs are now labelled ``transit`` and walking legs
    ``walk``.

    Conservation is unaffected: rewriting a path changes *where* a trajectory's
    mass lands, never *how much* of it there is (the ledger books mass per
    trajectory, not per edge — and per trajectory it stays booked once even when
    its legs land in two partitions).
    """
    if "transit" not in state.graphs:
        return trajectories, None
    from segx.network.transit import edge_layers, expand_transit_rides

    graph = state.graphs.get("transit")
    out: list[Trajectory] = []
    for trajectory in trajectories:
        if trajectory.mode != "transit" or trajectory.path is None:
            out.append(trajectory)
            continue
        trajectory.path, trajectory.leg_modes = expand_transit_rides(trajectory.path, graph)
        out.append(trajectory)
    return out, edge_layers(graph)


def _resolve_street_space(state: IngestResult, settings: Settings):
    """Resolve the canonical corridor map for this run, or ``None`` (P7.2).

    Skipped when switched off or when the run holds a single mode graph — with one
    mode there is nothing to pool across, and every segment would become its own
    corridor anyway.
    """
    if not settings.network.street_space.enabled or len(state.graphs.modes()) < 2:
        return None
    from segx.network.street_space import resolve_street_space

    return resolve_street_space(state.graphs, settings, working_crs=state.boundary.crs)


def _cross_mode_audit(edge_table: pd.DataFrame, state: IngestResult, street_space=None) -> dict:
    """Whether the ``cross_mode_exposure`` join found anything to join (P7.1).

    Three parts of one coverage story, computed here rather than inside the metric
    because :class:`~segx.metrics.base.Metric` returns only a frame:

    * how many street pools actually carry more than one mode, **and the same
      broken out per mode pair** — the scalar alone is what let the Philadelphia
      run report 64 % while ``walk ∩ drive`` was exactly zero;
    * the corridor resolution (P7.2): how many edges matched a shared street
      space, and under which parameters. The matched corridor is an *inference*,
      so the parameters are stamped here and travel into the output metadata —
      the same discipline as ``transit_source = "osm_assumed"`` (invariant #7);
    * ``n_surface_rides_unmatched`` from the transit graph's build provenance — a
      surface ride that failed map-matching stays on ``rail_line`` and drops out
      of the street pool too.

    None of these shows up as a zero in the output; they all show up as
    ``cross_mode_exposure == exposure``, which reads like a finding. Hence the
    numbers.
    """
    from segx.metrics.cross_mode_exposure import cross_mode_stats

    audit = cross_mode_stats(edge_table)
    unmatched = None
    if "transit" in state.graphs:
        unmatched = state.graphs.get("transit").graph.get("transit", {}).get("n_surface_rides_unmatched")
    audit["n_surface_rides_unmatched"] = unmatched
    audit["street_space"] = {**street_space.stats, "params": street_space.params} if street_space is not None else None
    return audit


def aggregate(
    state: IngestResult,
    population: PopulationResult,
    routing: RoutingResult,
    settings: Settings,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> AggregationResult:
    """Run the three P4 stages: accumulate per-edge mass, compute metrics, check
    conservation. Streams the trajectory list through :class:`EdgeAccumulator`.
    """
    stages = [
        "expand_transit_rides",
        "resolve_street_space",
        "aggregate_edges",
        "compute_metrics",
        "check_invariants",
    ]
    total = len(stages)

    def _stage(name: str, message: str = "") -> None:
        report(progress, cancel, stage=name, index=stages.index(name), total=total, message=message)

    _stage("expand_transit_rides", "transit" if "transit" in state.graphs else "no transit graph")
    trajectories, layers = _expand_transit(routing.trajectories, state)

    _stage("resolve_street_space", f"modes={state.graphs.modes()}")
    street_space = _resolve_street_space(state, settings)

    _stage("aggregate_edges", f"{len(trajectories)} trajectories")
    accumulator = EdgeAccumulator(
        settings,
        edge_layers=layers,
        edge_corridors=street_space.corridors if street_space is not None else None,
    )
    # Resolved before the stream so a switched-off metric's supporting work is
    # skipped too, not just its `compute` call.
    metric_names = settings.metrics.active(available_metrics())
    needs_path_table = any("path_table" in get_metric(name).requires for name in metric_names)
    # P7.3: path shape is measured in the *same* pass. Two accumulators, one
    # stream — the alternative (a second pass over the trajectory list) would be
    # the only place in the pipeline that reads a trajectory twice. It runs only
    # when an active metric declares `path_table`: rasterising every path is the
    # most expensive thing in this loop.
    path_accumulator: PathAccumulator | None = None
    if needs_path_table:
        node_xy = _node_coordinates(state)
        edge_xy = _edge_coordinates(state, node_xy) if settings.metrics.fragmentation.use_edge_geometry else None
        path_accumulator = PathAccumulator(settings, node_xy, edge_xy)
    origin_vectors = _origin_vectors(population)
    od_flow_vectors = _od_flow_vectors(state, settings)
    for trajectory in trajectories:
        demo_vector = _demo_vector_for(trajectory, origin_vectors, od_flow_vectors)
        accumulator.add(trajectory, demo_vector)
        if path_accumulator is not None:
            path_accumulator.add(trajectory, demo_vector)
    edge_table = accumulator.to_frame()
    path_table = path_accumulator.to_frame() if path_accumulator is not None else None

    _stage("compute_metrics", ", ".join(metric_names))
    # Frames a metric may declare via `Metric.requires`, beyond the edge table.
    # A metric gets only what it names, so nothing can quietly grow a dependency
    # on an aggregate it never asked for.
    extra_frames: dict[str, pd.DataFrame | None] = {"path_table": path_table}
    metrics: dict[str, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    for name in metric_names:
        metric = get_metric(name)()
        extras = {key: extra_frames[key] for key in metric.requires}
        per_segment = metric.compute(edge_table, settings, **extras)
        metrics[name] = per_segment
        summaries[name] = metric.summarize(per_segment, settings)

    _stage("check_invariants", "weight conservation")
    ledger = accumulator.ledger()
    check_weight_conservation(ledger)

    if edge_table.empty:
        n_edges = 0
    else:
        n_edges = int(edge_table[["mode", "category", "binding", "u", "v", "key"]].drop_duplicates().shape[0])
    stats = {
        "n_reachable": accumulator.n_reachable,
        "n_unreachable": accumulator.n_unreachable,
        "n_unresolved": accumulator.n_unresolved,
        "n_segments": n_edges,
        "edge_mass_total": accumulator.edge_mass_total,
        "partitions": [list(p) for p in accumulator.partitions()],
        "cross_mode": _cross_mode_audit(edge_table, state, street_space),
        # P7.3: the same discipline as `cross_mode` — the failure mode of a
        # fragmentation run is not a zero, it is a column of NaN because every
        # path was too small for the scale ladder. These counts say which it was.
        # `None` when no active metric asked for path shape — not measured, as
        # against measured-and-empty.
        "fragmentation": path_accumulator.stats() if path_accumulator is not None else None,
        "metrics_enabled": list(metric_names),
    }
    return AggregationResult(
        edge_table=edge_table,
        metrics=metrics,
        summaries=summaries,
        ledger=ledger,
        stats=stats,
        path_table=path_table,
    )


def write(
    state: IngestResult,
    routing: RoutingResult,
    aggregation: AggregationResult,
    settings: Settings,
    output_dir: str | Path,
    *,
    od_spread=None,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> WrittenOutputs:
    """The ``write_outputs`` stage (P8): ``segments.gpkg`` + ``metrics.xlsx``.

    Thin wrapper over :func:`segx.io.writers.write_outputs` that pulls the
    working CRS, the graphs (for segment geometry) and the routing audit out of
    the stage results.
    """
    report(progress, cancel, stage="write_outputs", index=0, total=1, message=str(output_dir))
    return write_outputs(
        aggregation,
        settings,
        output_dir,
        graphs=state.graphs,
        crs=state.boundary.crs if state.boundary is not None else None,
        routing_stats=routing.stats,
        od_spread=od_spread,
    )


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything a full :func:`run_pipeline` produced.

    ``outputs`` says what was written and where; the stage results are kept so a
    caller (the RPC shim, the demo) can summarise or render without re-running.
    """

    state: IngestResult
    population: PopulationResult
    routing: RoutingResult
    aggregation: AggregationResult
    od_spread: object | None
    outputs: WrittenOutputs

    @property
    def stats(self) -> dict:
        return self.aggregation.stats


def run_pipeline(
    settings: Settings,
    *,
    boundary_path: str | Path,
    demographics_path: str | Path,
    od_pairs_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    cache_dir: Path | None = None,
    poi_fetcher: str = "auto",
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> RunResult:
    """Run the full pipeline and write the deliverables.

    ingest → OD spread → population → routing → aggregation/metrics → writers.
    ``output_dir`` defaults to ``settings.output_dir``. ``progress`` and
    ``cancel`` thread through to every stage (see :mod:`segx.progress`).
    """
    state = ingest(
        settings,
        boundary_path=boundary_path,
        demographics_path=demographics_path,
        od_pairs_path=od_pairs_path,
        cache_dir=cache_dir,
        poi_fetcher=poi_fetcher,
        progress=progress,
        cancel=cancel,
    )
    od_spread = analyse_od(state, settings, progress=progress, cancel=cancel)
    population = build_population(state, settings, progress=progress, cancel=cancel)
    routing = route(state, population, settings, progress=progress, cancel=cancel)
    aggregation = aggregate(state, population, routing, settings, progress=progress, cancel=cancel)
    outputs = write(
        state,
        routing,
        aggregation,
        settings,
        output_dir if output_dir is not None else settings.output_dir,
        od_spread=od_spread,
        progress=progress,
        cancel=cancel,
    )
    return RunResult(
        state=state,
        population=population,
        routing=routing,
        aggregation=aggregation,
        od_spread=od_spread,
        outputs=outputs,
    )


__all__ = [
    "PIPELINE_STAGES",
    "AggregationResult",
    "IngestResult",
    "PopulationResult",
    "RoutingResult",
    "RunResult",
    "aggregate",
    "analyse_od",
    "build_population",
    "ingest",
    "route",
    "run_pipeline",
    "write",
]
