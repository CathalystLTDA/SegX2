"""End-to-end / cross-engine checks on the synthetic city — the gating test.

P3 (routing) and P4 (aggregation + diversity) are switched on: the binding
ground-truth, the NetworkX-oracle-vs-Pandana agreement, the per-edge demographic
sums, the paper's diversity metric and the weight-conservation invariant all run
here against :mod:`segx.validation.synthetic_city`. The fast, geopandas-free
version of the binding assertions also lives in ``tests/unit/test_routing.py``;
these integration copies additionally gate the **default** engine (Pandana) via
the cross-engine agreement test.
"""

from __future__ import annotations

import math

import pytest

from segx.aggregation.edge_accumulator import EdgeAccumulator
from segx.config import (
    CatchmentBindingConfig,
    DemographicAttribute,
    DemographicSchema,
    NearestBindingConfig,
    NetworkConfig,
    POIBindingConfig,
    RoutingConfig,
    Settings,
    TransitConfig,
)
from segx.metrics.cross_mode_exposure import CrossModeExposureMetric, cross_mode_stats
from segx.metrics.entropy import ShannonEntropyMetric, segment_diversity
from segx.metrics.exposure import ExposureMetric
from segx.network.transit import build_transit_graph, edge_layers, expand_transit_rides
from segx.routing.inputs import RoutableODPair, RoutableOrigin, RoutablePOI
from segx.routing.nearest_poi import nearest_poi_trajectories
from segx.routing.networkx_engine import NetworkXEngine
from segx.routing.od_router import od_pair_trajectories
from segx.routing.poi_catchment import catchment_trajectories
from segx.validation.invariants import check_weight_conservation
from segx.validation.synthetic_city import (
    CATCHMENT_BUDGET_STEPS,
    CATCHMENT_GROUND_TRUTH,
    DEMOGRAPHIC_CLASSES,
    EDGE_M,
    NEAREST_POI_GROUND_TRUTH,
    ORIGIN_DEMOGRAPHICS,
    ORIGIN_NODES,
    POI_CATEGORIES,
    POIS,
    SYNTHETIC_CRS,
    TRANSIT_BOARD_PENALTY_MIN,
    TRANSIT_FAST_HEADWAY_MIN,
    TRANSIT_SLOW_HEADWAY_MIN,
    WALK_0_TO_3_MIN,
    build_grid_graph,
    build_transit_feed,
)

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    cats = list(POI_CATEGORIES)
    return Settings(
        poi=POIBindingConfig(
            presets={c: {} for c in cats},
            nearest=NearestBindingConfig(categories=cats),
            catchment=CatchmentBindingConfig(
                categories=cats, budget_minutes=None, distance_budget_m=CATCHMENT_BUDGET_STEPS * EDGE_M
            ),
        ),
        routing=RoutingConfig(engine="networkx", access_mode="walk"),
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=list(DEMOGRAPHIC_CLASSES))]
        ),
    )


def _demo_vectors() -> dict[str, dict[str, dict[str, float]]]:
    """``origin_id -> {income: {class: mass}}`` from the synthetic ground truth."""
    return {oid: {"income": dict(classes)} for oid, classes in ORIGIN_DEMOGRAPHICS.items()}


def _origins() -> list[RoutableOrigin]:
    return [RoutableOrigin(oid, node, 1.0) for oid, node in ORIGIN_NODES.items()]


def _pois() -> dict[str, list[RoutablePOI]]:
    out: dict[str, list[RoutablePOI]] = {}
    for poi_id, (category, node) in POIS.items():
        out.setdefault(category, []).append(RoutablePOI(poi_id, node))
    return out


def _catchment_sets(trajs) -> dict[tuple[str, str], set[str]]:
    got: dict[tuple[str, str], set[str]] = {}
    for t in trajs:
        got.setdefault((t.origin_id, t.category), set()).add(t.destination_id)
    return got


def test_nearest_binding_matches_ground_truth():
    engine = NetworkXEngine({"walk": build_grid_graph()})
    trajs = {
        (t.origin_id, t.category): t
        for t in nearest_poi_trajectories(_origins(), _pois(), engine, _settings(), mode="walk")
    }
    for oid in ORIGIN_NODES:
        for category in POI_CATEGORIES:
            gt_poi, gt_steps = NEAREST_POI_GROUND_TRUTH[oid][category]
            assert trajs[(oid, category)].destination_id == gt_poi
            assert trajs[(oid, category)].cost == pytest.approx(gt_steps * EDGE_M)


def test_catchment_binding_matches_ground_truth():
    engine = NetworkXEngine({"walk": build_grid_graph()})
    got = _catchment_sets(catchment_trajectories(_origins(), _pois(), engine, _settings(), mode="walk"))
    for oid in ORIGIN_NODES:
        for category in POI_CATEGORIES:
            assert got.get((oid, category), set()) == set(CATCHMENT_GROUND_TRUTH[oid][category])


def test_networkx_oracle_agrees_with_pandana():
    pytest.importorskip("pandana")
    from segx.routing.pandana_engine import PandanaEngine

    grid = build_grid_graph()
    nx_engine = NetworkXEngine({"walk": grid})
    pd_engine = PandanaEngine({"walk": grid})
    settings, origins, pois = _settings(), _origins(), _pois()

    nx_near = {
        (t.origin_id, t.category): t.cost
        for t in nearest_poi_trajectories(origins, pois, nx_engine, settings, mode="walk")
    }
    pd_near = {
        (t.origin_id, t.category): t.cost
        for t in nearest_poi_trajectories(origins, pois, pd_engine, settings, mode="walk")
    }
    assert pd_near.keys() == nx_near.keys()
    for key, nx_cost in nx_near.items():
        assert pd_near[key] == pytest.approx(nx_cost)

    nx_catch = _catchment_sets(catchment_trajectories(origins, pois, nx_engine, settings, mode="walk"))
    pd_catch = _catchment_sets(catchment_trajectories(origins, pois, pd_engine, settings, mode="walk"))
    assert nx_catch == pd_catch


def test_per_edge_demographic_sums_and_entropy():
    """Route nearest for every origin, accumulate, and check the per-class edge
    sums and the paper's diversity against independent recomputations."""
    engine = NetworkXEngine({"walk": build_grid_graph()})
    settings, vectors = _settings(), _demo_vectors()
    trajs = list(nearest_poi_trajectories(_origins(), _pois(), engine, settings, mode="walk"))

    acc = EdgeAccumulator(settings)
    for t in trajs:
        acc.add(t, vectors[t.origin_id])
    frame = acc.to_frame()

    # Independent expectation: each reachable trajectory deposits its origin's
    # class mass onto every edge of its path, so the grand total per class is
    # Σ_traj (class mass × path length).
    expected = {cls: 0.0 for cls in DEMOGRAPHIC_CLASSES}
    for t in trajs:
        if t.path:
            for cls in DEMOGRAPHIC_CLASSES:
                expected[cls] += vectors[t.origin_id]["income"][cls] * len(t.path)
    got = frame.groupby("class")["mass"].sum().to_dict()
    for cls in DEMOGRAPHIC_CLASSES:
        assert got.get(cls, 0.0) == pytest.approx(expected[cls])

    # Diversity: finite and non-negative everywhere, and equal to a direct
    # recomputation of segment_diversity on each segment's class masses.
    diversity = ShannonEntropyMetric().compute(frame, settings)
    assert len(diversity) > 0
    assert diversity["diversity"].notna().all()
    assert (diversity["diversity"] >= 0).all()
    row = diversity.iloc[0]
    seg = frame[
        (frame["binding"] == row.binding)
        & (frame["category"] == row.category)
        & (frame["u"] == row.u)
        & (frame["v"] == row.v)
        & (frame["key"] == row.key)
    ]
    assert row.diversity == pytest.approx(segment_diversity(seg["mass"].to_numpy()))


def test_weight_is_conserved():
    """Nearest + catchment mass is fully accounted for (emitted = deposited +
    unreachable), with emitted equal to the summed origin mass per trajectory."""
    engine = NetworkXEngine({"walk": build_grid_graph()})
    settings, vectors = _settings(), _demo_vectors()

    trajs = list(nearest_poi_trajectories(_origins(), _pois(), engine, settings, mode="walk"))
    trajs += list(catchment_trajectories(_origins(), _pois(), engine, settings, mode="walk"))

    acc = EdgeAccumulator(settings)
    for t in trajs:
        acc.add(t, vectors[t.origin_id])

    ledger = acc.ledger()
    check_weight_conservation(ledger)  # raises InvariantViolation on any leak

    def _mass(t) -> float:
        return math.fsum(vectors[t.origin_id]["income"].values())

    assert ledger.emitted == pytest.approx(math.fsum(_mass(t) for t in trajs))
    assert ledger.deposited == pytest.approx(math.fsum(_mass(t) for t in trajs if t.path is not None))
    # per-edge integrity: the whole edge table (one income attribute) sums to the
    # depositable mass Σ_traj (mass × path length).
    assert float(acc.to_frame()["mass"].sum()) == pytest.approx(acc.edge_mass_total)


# --------------------------------------------------------------------------- #
# P7 — public transport on the merged graph                                    #
# --------------------------------------------------------------------------- #


def _transit_settings() -> Settings:
    """The P4/P5 settings, plus transit enabled on the same demographic schema."""
    return _settings().model_copy(
        update={
            "network": NetworkConfig(modes=["walk", "drive", "transit"]),
            "transit": TransitConfig(enabled=True, board_penalty_min=TRANSIT_BOARD_PENALTY_MIN),
            "routing": RoutingConfig(engine="networkx", access_mode="transit"),
        }
    )


def _transit_graph(headway: float):
    settings = _transit_settings()
    grid = build_grid_graph()
    graph = build_transit_graph(build_transit_feed(headway), grid, grid, settings, working_crs=SYNTHETIC_CRS)
    return graph, settings


def _transit_pairs() -> list[RoutableODPair]:
    """Two commutes: along the bus line (0->3) and along the metro line (12->15)."""
    return [
        RoutableODPair("o1", 0, "d_bus", 3, "transit", "commute", 1.0),
        RoutableODPair("o2", 12, "d_metro", 15, "transit", "commute", 1.0),
    ]


def _accumulate_transit(headway: float):
    """Route, expand and accumulate — the orchestrator's P7 path, in miniature."""
    graph, settings = _transit_graph(headway)
    engine = NetworkXEngine({"transit": graph}, settings)
    trajectories = list(od_pair_trajectories(_transit_pairs(), engine, settings))
    for trajectory in trajectories:
        trajectory.path, trajectory.leg_modes = expand_transit_rides(trajectory.path, graph)
    acc = EdgeAccumulator(settings, edge_layers=edge_layers(graph))
    vectors = _demo_vectors()
    for trajectory in trajectories:
        acc.add(trajectory, vectors[trajectory.origin_id])
    return graph, settings, trajectories, acc


def test_transit_trajectory_has_the_walk_ride_walk_shape():
    """Verification criterion #2: the shape is *decoded*, not asserted."""
    graph, settings = _transit_graph(TRANSIT_FAST_HEADWAY_MIN)
    engine = NetworkXEngine({"transit": graph}, settings)
    trajectory = next(iter(od_pair_trajectories(_transit_pairs()[:1], engine, settings)))
    kinds = [graph.get_edge_data(*e)["segx_edge"] for e in trajectory.path]
    assert kinds == ["connector", "board", "ride", "ride", "ride", "alight", "connector"]
    layers = [graph.get_edge_data(*e)["segx_layer"] for e in trajectory.path]
    assert layers == ["access", "access", "street", "street", "street", "access", "access"]


def test_transit_weight_is_conserved():
    """Verification criterion #1 — the invariant that fails loudly if ride
    expansion double-counts or drops mass."""
    _, _, trajectories, acc = _accumulate_transit(TRANSIT_FAST_HEADWAY_MIN)
    ledger = acc.ledger()
    check_weight_conservation(ledger)  # raises InvariantViolation on any leak
    vectors = _demo_vectors()

    def _mass(t) -> float:
        return math.fsum(vectors[t.origin_id]["income"].values())

    assert ledger.emitted == pytest.approx(math.fsum(_mass(t) for t in trajectories))
    assert ledger.deposited == pytest.approx(ledger.emitted)  # both pairs are reachable
    assert float(acc.to_frame()["mass"].sum()) == pytest.approx(acc.edge_mass_total)


def test_bus_mass_lands_on_streets_and_metro_mass_on_its_own_layer():
    """Verification criterion #3, and the whole point of D4: a bus rider is
    co-present with the avenue; a metro rider in a tunnel is not."""
    _, _, _, acc = _accumulate_transit(TRANSIT_FAST_HEADWAY_MIN)
    frame = acc.to_frame()
    by_layer = frame.groupby("layer")["mass"].sum()
    assert by_layer.get("street", 0.0) > 0
    assert by_layer.get("rail_line", 0.0) > 0
    assert "access" not in by_layer  # access edges never receive demographic mass

    # the bus commute deposits on the three real grid edges of the bottom row
    street_edges = {(int(r.u), int(r.v), int(r.key)) for r in frame[frame["layer"] == "street"].itertuples(index=False)}
    assert street_edges == {(0, 1, 0), (1, 2, 0), (2, 3, 0)}
    # the metro deposits only on synthetic rail edges, never on the streets above
    rail_edges = frame[frame["layer"] == "rail_line"]
    assert (rail_edges["u"] < 0).all() and (rail_edges["v"] < 0).all()


def test_metrics_run_unchanged_over_a_transit_edge_table():
    """Invariant #5 in practice: `layer` is a passthrough column, so no metric
    plugin needed a single line changed for P7."""
    _, settings, _, acc = _accumulate_transit(TRANSIT_FAST_HEADWAY_MIN)
    diversity = ShannonEntropyMetric().compute(acc.to_frame(), settings)
    assert len(diversity) > 0
    assert diversity["diversity"].notna().all()


def test_pandana_and_networkx_agree_on_the_merged_graph():
    """Verification criterion #4, and the D6 risk closed: pandana indexes and
    routes through the synthetic *negative* node ids correctly.

    Two claims, because the merged graph is full of ties (riding the bus east then
    walking north costs exactly what walking north then riding the metro east
    costs, so ``0 -> 15`` has four optima). The verification plan permits
    engines to "differ only by ties", so:

    * on every pair the two engines must report the **same cost**, and each
      returned path must actually *sum* to that cost — a decoded path that does
      not add up would mean the edge decoding is wrong, tie or no tie;
    * on the two unambiguous corridors the paths must be **byte-identical**.
    """
    pytest.importorskip("pandana")
    from segx.routing.pandana_engine import PandanaEngine

    graph, settings = _transit_graph(TRANSIT_FAST_HEADWAY_MIN)
    nx_engine = NetworkXEngine({"transit": graph}, settings)
    pd_engine = PandanaEngine({"transit": graph}, settings)

    def _path_cost(path) -> float:
        return math.fsum(graph.get_edge_data(*edge)["travel_time_min"] for edge in path)

    for src, dst in ((0, 3), (12, 15), (0, 15), (4, 11)):
        nx_result = nx_engine.shortest_path(src, dst, "transit", weight="travel_time_min")
        pd_result = pd_engine.shortest_path(src, dst, "transit", weight="travel_time_min")
        assert (nx_result is None) == (pd_result is None)
        if nx_result is None:
            continue
        assert pd_result.cost == pytest.approx(nx_result.cost, rel=1e-3)
        assert _path_cost(nx_result.path) == pytest.approx(nx_result.cost)
        assert _path_cost(pd_result.path) == pytest.approx(nx_result.cost, rel=1e-3)

    for src, dst in ((0, 3), (12, 15)):
        assert (
            pd_engine.shortest_path(src, dst, "transit", weight="travel_time_min").path
            == nx_engine.shortest_path(src, dst, "transit", weight="travel_time_min").path
        )


# --------------------------------------------------------------------------- #
# P7.1 — cross-mode exposure needs a genuinely multi-mode edge table            #
# --------------------------------------------------------------------------- #


def _accumulate_multimode():
    """A walk commute alongside the two transit ones, in **one** accumulator.

    Every P7 fixture pins ``access_mode="transit"``, so until now no edge table in
    the repo ever held more than one ``mode`` — and ``cross_mode_exposure`` is
    identically ``exposure`` on a single-mode table. The walking pair rides the
    same bottom row as the bus (0 -> 3 is unique on this grid), so the two land on
    the same ``(u, v, key)`` and actually meet.
    """
    graph, settings = _transit_graph(TRANSIT_FAST_HEADWAY_MIN)
    grid = build_grid_graph()
    engine = NetworkXEngine({"transit": graph, "walk": grid}, settings)
    pairs = [*_transit_pairs(), RoutableODPair("o3", 0, "d_walk", 3, "walk", "commute", 1.0)]

    trajectories = list(od_pair_trajectories(pairs, engine, settings))
    for trajectory in trajectories:
        if trajectory.mode == "transit":
            trajectory.path, trajectory.leg_modes = expand_transit_rides(trajectory.path, graph)
    acc = EdgeAccumulator(settings, edge_layers=edge_layers(graph))
    vectors = _demo_vectors()
    for trajectory in trajectories:
        acc.add(trajectory, vectors[trajectory.origin_id])
    return settings, acc.to_frame()


def test_a_walk_and_a_bus_commute_share_a_street_segment():
    settings, frame = _accumulate_multimode()
    assert set(frame["mode"]) == {"walk", "transit"}

    streets = frame[frame["layer"] == "street"]
    per_segment = streets.groupby(["u", "v", "key"])["mode"].nunique()
    assert (per_segment > 1).any(), "no segment carries both modes — the join found nothing"

    stats = cross_mode_stats(frame)
    assert stats["n_modes_present"] == 2
    assert stats["cross_mode_join_rate"] > 0.0


def test_cross_mode_exposure_differs_from_exposure_on_a_multimode_table():
    """The whole point: with `mode` as a partition key the pedestrian and the bus
    passenger on one avenue meet in no P5 metric. Here they do."""
    settings, frame = _accumulate_multimode()
    cross = CrossModeExposureMetric().compute(frame, settings)
    plain = ExposureMetric().compute(frame, settings)

    key = ["mode", "category", "binding", "attribute", "class"]
    merged = cross.merge(plain, on=key, how="inner")
    walk_low = merged[(merged["mode"] == "walk") & (merged["class"] == "low")].iloc[0]

    # o3 walks (low 3, high 1) -> exposure H(3, 1); o1 rides the bus (low 4,
    # high 0) down the same row, so at f = 0.5 the walker's effective vector is
    # (3 + 2, 1) = (5, 1) -- *less* mixed than what walking alone reported.
    assert float(walk_low["exposure"]) == pytest.approx(0.8112781244591328)
    assert float(walk_low["cross_mode_exposure"]) == pytest.approx(0.6500224216483541)
    assert float(walk_low["cross_mode_exposure"]) < float(walk_low["exposure"])


def test_an_infrequent_line_leaves_riders_on_foot_end_to_end():
    """The frequency model has to be able to say *no*: at a 10-minute headway the
    bus (7.5 min) loses to walking (3.6 min), and all the mass stays on streets."""
    _, _, trajectories, acc = _accumulate_transit(TRANSIT_SLOW_HEADWAY_MIN)
    bus_commute = next(t for t in trajectories if t.destination_id == "d_bus")
    assert bus_commute.cost == pytest.approx(WALK_0_TO_3_MIN)
    assert set(acc.to_frame()["layer"]) == {"street"}


# --------------------------------------------------------------------------- #
# Canonical street space — car x pedestrian x bus (P7.2)                       #
# --------------------------------------------------------------------------- #


def _multimode_registry():
    """A registry shaped like the real defect: three modes on three id spaces.

    * ``drive`` — the grid, i.e. the roadway centrelines;
    * ``walk`` — the **separately-mapped** sidewalk layer, sharing not one node
      with the grid (this is what OSMnx produces for a city tagged
      ``sidewalk=separate``, and it is why no id remap can join the two);
    * ``transit`` — the merged walk (grid) + bus graph, whose surface rides are
      map-matched onto grid ids.
    """
    from segx.network.registry import GraphRegistry
    from segx.routing._paths import as_multidigraph
    from segx.validation.synthetic_city import build_sidewalk_graph

    transit_graph, settings = _transit_graph(TRANSIT_FAST_HEADWAY_MIN)
    graphs = GraphRegistry()
    graphs.register("drive", as_multidigraph(build_grid_graph()))
    graphs.register("walk", build_sidewalk_graph())
    graphs.register("transit", transit_graph)
    return graphs, transit_graph, settings


def _street_space(graphs, settings):
    from segx.network.street_space import resolve_street_space
    from segx.validation.synthetic_city import SYNTHETIC_CRS

    return resolve_street_space(graphs, settings, working_crs=SYNTHETIC_CRS)


def test_a_separately_mapped_sidewalk_shares_the_roadway_corridor():
    """The synthetic restatement of the Philadelphia finding. The sidewalk shares
    no node with the roadway, so ``(u, v, key)`` can never join them; the corridor
    does. The crossing and the park path stay out — one on bearing, one on
    distance."""
    from segx.validation.synthetic_city import CROSSING_EDGE, PARK_PATH_EDGE, SIDEWALK_EDGE

    graphs, _, settings = _multimode_registry()
    space = _street_space(graphs, settings)

    roadway = space.get((0, 1, 0))  # grid row r=0, the street the sidewalk runs along
    assert space.get(SIDEWALK_EDGE).corridor_id == roadway.corridor_id
    assert space.get(CROSSING_EDGE).corridor_id != roadway.corridor_id
    assert space.get(PARK_PATH_EDGE).corridor_id != roadway.corridor_id
    assert space.stats["corridors_per_mode_pair"]["drive|walk"] >= 1


def test_every_mode_pair_meets_once_the_corridor_is_resolved():
    """The acceptance criterion for this phase: ``drive+walk``
    must stop being zero — and ``drive+transit`` and
    ``transit+walk`` must be non-zero too.

    Bus riders are map-matched onto grid (``drive``) ids, pedestrians are on the
    sidewalk layer, and cars are on the grid. Before P7.2 that is three disjoint
    id spaces and every cross-mode number is a fiction.
    """
    from segx.metrics.cross_mode_exposure import cross_mode_combination_counts
    from segx.routing.trajectories import Trajectory
    from segx.validation.synthetic_city import SIDEWALK_EDGE

    graphs, transit_graph, settings = _multimode_registry()
    space = _street_space(graphs, settings)

    ride = NetworkXEngine({"transit": transit_graph}, settings).shortest_path(0, 3, "transit", weight="travel_time_min")
    ride_path, ride_legs = expand_transit_rides(ride.path, transit_graph)

    trajectories = [
        Trajectory("o0", "d0", "drive", "commute", "od", path=[(0, 1, 0)]),
        Trajectory("o1", "d1", "walk", "commute", "od", path=[SIDEWALK_EDGE]),
        Trajectory("o2", "d2", "transit", "commute", "od", path=ride_path, leg_modes=ride_legs),
    ]
    acc = EdgeAccumulator(settings, edge_layers=edge_layers(transit_graph), edge_corridors=space.corridors)
    for trajectory in trajectories:
        acc.add(trajectory, {"income": {"low": 1.0, "high": 1.0}})
    frame = acc.to_frame()

    # Here one corridor carries all three modes, so it is reported as the single
    # exact combination `drive+transit+walk` — the pair-wise question is "does any
    # combination contain both", which is what the audit table is read for.
    combos = cross_mode_combination_counts(frame)
    shared_in = {
        (a, b)
        for row in combos.itertuples()
        for a in row.modes.split("+")
        for b in row.modes.split("+")
        if a < b and int(row.n_segments) > 0
    }
    assert ("drive", "walk") in shared_in, "the acceptance criterion: drive+walk must not be zero"
    assert ("drive", "transit") in shared_in
    assert ("transit", "walk") in shared_in

    pairs = cross_mode_stats(frame)["pair_join_rates"]
    assert all(pairs[key]["n_shared"] > 0 for key in ("drive|walk", "drive|transit", "transit|walk"))


def test_the_corridor_moves_the_drivers_number_off_exposure():
    """What the fix is *for*. The driver's ``cross_mode_exposure`` must stop being
    identical to ``exposure`` — in Philadelphia it moved 1.20 -> 1.21, which was
    the engine failing to see pedestrians, not a fact about the city."""
    from segx.routing.trajectories import Trajectory
    from segx.validation.synthetic_city import SIDEWALK_EDGE

    graphs, transit_graph, settings = _multimode_registry()
    space = _street_space(graphs, settings)

    # deliberately unlike composition, so pooling has to change the entropy
    driver = {"income": {"low": 0.0, "high": 4.0}}
    pedestrian = {"income": {"low": 4.0, "high": 0.0}}

    acc = EdgeAccumulator(settings, edge_corridors=space.corridors)
    acc.add(Trajectory("o0", "d0", "drive", "commute", "od", path=[(0, 1, 0)]), driver)
    acc.add(Trajectory("o1", "d1", "walk", "commute", "od", path=[SIDEWALK_EDGE]), pedestrian)
    frame = acc.to_frame()

    cross = CrossModeExposureMetric().compute(frame, settings)
    plain = ExposureMetric().compute(frame, settings)
    merged = cross.merge(plain, on=["mode", "category", "binding", "attribute", "class"])
    drive_high = merged[(merged["mode"] == "drive") & (merged["class"] == "high")].iloc[0]
    assert float(drive_high["exposure"]) == pytest.approx(0.0)  # alone, the driver sees one class
    assert float(drive_high["cross_mode_exposure"]) > 0.0  # with the pedestrian, real diversity


def test_leg_modes_split_a_transit_trip_without_breaking_conservation():
    """A bus trip walks, rides, and walks again. Its legs land in two partitions
    now — but the ledger books mass **once per trajectory**, so conservation is
    untouched. This is the guard on that."""
    from segx.routing.trajectories import Trajectory

    _, transit_graph, settings = _multimode_registry()
    engine = NetworkXEngine({"transit": transit_graph}, settings)
    result = engine.shortest_path(4, 3, "transit", weight="travel_time_min")
    path, legs = expand_transit_rides(result.path, transit_graph)
    assert set(legs) == {"walk", "transit"}, "fixture must exercise both leg kinds"

    vector = {"income": {"low": 2.0, "high": 0.0}}
    acc = EdgeAccumulator(settings, edge_layers=edge_layers(transit_graph))
    acc.add(Trajectory("o0", "d0", "transit", "commute", "od", path=path, leg_modes=legs), vector)

    frame = acc.to_frame()
    assert set(frame["mode"]) == {"walk", "transit"}
    check_weight_conservation(acc.ledger())
    ledger = acc.ledger()
    assert ledger.emitted == pytest.approx(2.0)  # one trajectory, booked once
    assert acc.n_reachable == 1


# --------------------------------------------------------------------------- #
# P7.3 — path fragmentation over the same routed stream
# --------------------------------------------------------------------------- #
def _node_xy(graph) -> dict[int, tuple[float, float]]:
    return {n: (float(d["x"]), float(d["y"])) for n, d in graph.nodes(data=True)}


def test_box_counting_measures_every_path_of_the_synthetic_grid():
    """The default method (the paper's pixels) has no segment floor: a one-block
    walk is 100 m, which spans four resolutions. A grid geodesic is straight or
    L-shaped, so ``D`` sits near 1."""
    from segx.aggregation.path_accumulator import PathAccumulator
    from segx.metrics.fragmentation import PathFragmentationMetric

    graph = build_grid_graph()
    settings, vectors = _settings(), _demo_vectors()
    trajs = list(nearest_poi_trajectories(_origins(), _pois(), NetworkXEngine({"walk": graph}), settings, mode="walk"))

    acc = PathAccumulator(settings, _node_xy(graph))
    for t in trajs:
        acc.add(t, vectors[t.origin_id])

    stats = acc.stats()
    assert stats["n_measured"] > 0
    assert stats["n_saturated"] == stats["n_too_short"] == 0

    frame = PathFragmentationMetric().compute(None, settings, path_table=acc.to_frame())
    measured = frame[frame["n_measured"] > 0]
    assert measured["fragmentation"].between(0.9, 1.1).all()


def test_the_synthetic_grid_is_below_the_divider_resolution_floor():
    """Pinned deliberately, because the NaN column it produces is *correct*.

    Under ``method = "divider"`` the synthetic city is too small: a 4x4 grid's
    longest trajectory is five segments, and the default ruler ladder needs eight
    to fit a slope. So every path is booked ``n_too_short`` and every value is
    ``NaN`` — "these trips are too short to have a fractal dimension", not "the
    metric is broken" (invariant #2).
    """
    from segx.aggregation.path_accumulator import PathAccumulator
    from segx.config import FragmentationConfig, MetricsConfig

    graph = build_grid_graph()
    settings = _settings().model_copy(
        update={"metrics": MetricsConfig(fragmentation=FragmentationConfig(method="divider"))}
    )
    vectors = _demo_vectors()
    trajs = list(nearest_poi_trajectories(_origins(), _pois(), NetworkXEngine({"walk": graph}), settings, mode="walk"))
    assert max(len(t.path) for t in trajs if t.path) < 8, "fixture premise: no path reaches the floor"

    acc = PathAccumulator(settings, _node_xy(graph))
    for t in trajs:
        acc.add(t, vectors[t.origin_id])

    stats = acc.stats()
    assert stats["n_measured"] == 0
    assert stats["n_too_short"] == len(trajs)
    assert acc.to_frame()["n_measured"].sum() == 0


def test_fragmentation_runs_end_to_end_once_the_ladder_fits_the_city():
    """The same stream, with a divider ladder scaled to a small city, produces real
    numbers — and a grid's shortest paths are near-straight, so ``D`` sits close
    to 1 and the plain ``detour_ratio`` reading comes out at the Manhattan value.
    """
    from segx.aggregation.path_accumulator import PathAccumulator
    from segx.config import FragmentationConfig, MetricsConfig
    from segx.metrics.fragmentation import PathFragmentationMetric

    graph = build_grid_graph()
    settings = _settings().model_copy(
        update={
            "metrics": MetricsConfig(fragmentation=FragmentationConfig(method="divider", min_segments=4, min_scales=2))
        }
    )
    vectors = _demo_vectors()
    trajs = list(nearest_poi_trajectories(_origins(), _pois(), NetworkXEngine({"walk": graph}), settings, mode="walk"))

    acc = PathAccumulator(settings, _node_xy(graph))
    for t in trajs:
        acc.add(t, vectors[t.origin_id])
    assert acc.stats()["n_measured"] > 0, "the lowered ladder must actually measure something"

    frame = PathFragmentationMetric().compute(None, settings, path_table=acc.to_frame())
    measured = frame[frame["n_measured"] > 0]
    assert not measured.empty
    # A grid geodesic is an L-shaped staircase: crooked, but equally crooked at
    # every scale, so D stays near 1 while detour_ratio carries the real detour.
    assert (measured["fragmentation"] >= 0.9).all()
    assert (measured["fragmentation"] <= 1.3).all()
    assert (measured["detour_ratio"] >= 1.0).all()
    # every reported mean is finite where a path was measured (never a silent 0)
    assert measured["fragmentation"].notna().all()
    assert measured["mean_segments"].notna().all()
