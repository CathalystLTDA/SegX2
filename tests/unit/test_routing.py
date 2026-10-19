"""P3 routing core — engine + the three trajectory strategies, asserted against
the synthetic city's hand-computed ground truth.

These run the NetworkX oracle on :mod:`segx.validation.synthetic_city`; they need
only NetworkX (no geopandas), so they live as unit tests. The Pandana cross-check
(``test_pandana_matches_networkx*``) imports pandana lazily via ``importorskip``.
"""

from __future__ import annotations

import networkx as nx
import pytest

from segx.config import (
    CatchmentBindingConfig,
    NearestBindingConfig,
    POIBindingConfig,
    RoutingConfig,
    Settings,
)
from segx.routing.inputs import RoutableODPair, RoutableOrigin, RoutablePOI
from segx.routing.nearest_poi import nearest_poi_trajectories
from segx.routing.networkx_engine import NetworkXEngine
from segx.routing.od_router import od_pair_trajectories
from segx.routing.poi_catchment import catchment_trajectories
from segx.validation.synthetic_city import (
    CATCHMENT_BUDGET_STEPS,
    CATCHMENT_GROUND_TRUTH,
    EDGE_M,
    EDGE_WALK_MIN,
    NEAREST_POI_GROUND_TRUTH,
    ORIGIN_NODES,
    POI_CATEGORIES,
    POIS,
    build_grid_graph,
    manhattan_steps,
    node_id,
)

# --------------------------------------------------------------------------- #
# Fixtures / helpers                                                           #
# --------------------------------------------------------------------------- #


def _settings(
    *,
    categories: tuple[str, ...] = POI_CATEGORIES,
    distance_budget_m: float | None = CATCHMENT_BUDGET_STEPS * EDGE_M,
    budget_minutes: float | None = None,
    weight_mode: str = "unit",
    max_distance_m: float | None = None,
) -> Settings:
    """A Settings whose POI bindings target the synthetic city's categories."""
    return Settings(
        poi=POIBindingConfig(
            presets={c: {} for c in categories},
            nearest=NearestBindingConfig(categories=list(categories), max_distance_m=max_distance_m),
            catchment=CatchmentBindingConfig(
                categories=list(categories),
                budget_minutes=budget_minutes,
                distance_budget_m=distance_budget_m,
                weight_mode=weight_mode,
            ),
        ),
        routing=RoutingConfig(engine="networkx", access_mode="walk"),
    )


@pytest.fixture
def engine() -> NetworkXEngine:
    return NetworkXEngine({"walk": build_grid_graph()})


@pytest.fixture
def origins() -> list[RoutableOrigin]:
    return [RoutableOrigin(oid, node, 1.0) for oid, node in ORIGIN_NODES.items()]


@pytest.fixture
def pois_by_category() -> dict[str, list[RoutablePOI]]:
    out: dict[str, list[RoutablePOI]] = {}
    for poi_id, (category, node) in POIS.items():
        out.setdefault(category, []).append(RoutablePOI(poi_id, node))
    return out


# --------------------------------------------------------------------------- #
# Strategy A — nearest                                                         #
# --------------------------------------------------------------------------- #


def test_nearest_matches_ground_truth(engine, origins, pois_by_category):
    trajs = list(nearest_poi_trajectories(origins, pois_by_category, engine, _settings(), mode="walk"))
    by_key = {(t.origin_id, t.category): t for t in trajs}
    for oid in ORIGIN_NODES:
        for category in POI_CATEGORIES:
            gt_poi, gt_steps = NEAREST_POI_GROUND_TRUTH[oid][category]
            traj = by_key[(oid, category)]
            assert traj.binding == "nearest"
            assert traj.reachable
            assert traj.destination_id == gt_poi
            assert traj.cost == pytest.approx(gt_steps * EDGE_M)
            # path has one edge per grid step, each a (u, v, key) triple
            assert len(traj.path) == gt_steps
            assert all(isinstance(e, tuple) and len(e) == 3 for e in traj.path)


def test_nearest_none_in_boundary_when_category_absent(engine, origins, pois_by_category):
    settings = _settings(categories=(*POI_CATEGORIES, "library"))  # no library POIs exist
    trajs = list(nearest_poi_trajectories(origins, pois_by_category, engine, settings, mode="walk"))
    libs = [t for t in trajs if t.category == "library"]
    assert len(libs) == len(ORIGIN_NODES)
    assert all(not t.reachable and t.reason == "none_in_boundary" for t in libs)


def test_nearest_over_max_distance(engine, origins, pois_by_category):
    # o1's nearest park (park_e) is 2 steps = 200 m; a 150 m cap rejects it.
    settings = _settings(max_distance_m=150.0)
    trajs = list(nearest_poi_trajectories(origins, pois_by_category, engine, settings, mode="walk"))
    o1_park = next(t for t in trajs if t.origin_id == "o1" and t.category == "park")
    assert not o1_park.reachable
    assert o1_park.reason == "over_max_distance"
    assert o1_park.destination_id == "park_e"  # still names which POI was too far


def test_nearest_disabled_yields_nothing(engine, origins, pois_by_category):
    settings = _settings()
    settings.poi.nearest.enabled = False
    assert list(nearest_poi_trajectories(origins, pois_by_category, engine, settings, mode="walk")) == []


# --------------------------------------------------------------------------- #
# Strategy B — catchment                                                       #
# --------------------------------------------------------------------------- #


def _catchment_sets(trajs) -> dict[tuple[str, str], set[str]]:
    got: dict[tuple[str, str], set[str]] = {}
    for t in trajs:
        got.setdefault((t.origin_id, t.category), set()).add(t.destination_id)
    return got


def test_catchment_matches_ground_truth(engine, origins, pois_by_category):
    trajs = list(catchment_trajectories(origins, pois_by_category, engine, _settings(), mode="walk"))
    assert all(t.binding == "catchment" and t.reachable for t in trajs)
    got = _catchment_sets(trajs)
    for oid in ORIGIN_NODES:
        for category in POI_CATEGORIES:
            expected = set(CATCHMENT_GROUND_TRUTH[oid][category])
            assert got.get((oid, category), set()) == expected


def test_catchment_shrinks_with_tighter_budget(engine, origins, pois_by_category):
    wide = _catchment_sets(list(catchment_trajectories(origins, pois_by_category, engine, _settings(), mode="walk")))
    tight_settings = _settings(distance_budget_m=(CATCHMENT_BUDGET_STEPS - 1) * EDGE_M)
    tight = _catchment_sets(
        list(catchment_trajectories(origins, pois_by_category, engine, tight_settings, mode="walk"))
    )
    strictly_smaller = False
    for key, wide_set in wide.items():
        assert tight.get(key, set()) <= wide_set  # subset
        if tight.get(key, set()) < wide_set:
            strictly_smaller = True
    assert strictly_smaller, "tightening the budget should drop at least one reachable POI"


def test_catchment_weight_modes(engine, pois_by_category):
    # o6 reaches exactly park_w and clinic within 200 m (k=1 each category) → both
    # weighting modes agree there; use o2 which reaches park_e + clinic (k=1 each).
    origins = [RoutableOrigin("o2", ORIGIN_NODES["o2"], 4.0)]  # non-unit base weight
    unit = list(catchment_trajectories(origins, pois_by_category, engine, _settings(weight_mode="unit"), mode="walk"))
    inv = list(
        catchment_trajectories(origins, pois_by_category, engine, _settings(weight_mode="inverse_count"), mode="walk")
    )
    # unit: each reachable POI carries the full origin weight
    assert all(t.weight == pytest.approx(4.0) for t in unit)
    # inverse_count: per (origin, category) the weights sum to the origin weight
    sums: dict[tuple[str, str], float] = {}
    for t in inv:
        sums[(t.origin_id, t.category)] = sums.get((t.origin_id, t.category), 0.0) + t.weight
    assert all(s == pytest.approx(4.0) for s in sums.values())


# --------------------------------------------------------------------------- #
# Strategy C — OD pairs                                                        #
# --------------------------------------------------------------------------- #


def test_od_pair_routing(engine):
    src, dst = node_id(1, 1), node_id(3, 3)
    pairs = [RoutableODPair("o1", src, "work", dst, "walk", "commute", weight=2.0)]
    trajs = list(od_pair_trajectories(pairs, engine, _settings()))
    assert len(trajs) == 1
    traj = trajs[0]
    assert traj.binding == "od" and traj.reachable
    assert traj.weight == 2.0
    # OD routes on travel_time_min
    assert traj.cost == pytest.approx(manhattan_steps(src, dst) * EDGE_WALK_MIN)
    assert len(traj.path) == manhattan_steps(src, dst)


def test_od_pair_unreachable_to_isolated_node():
    g = build_grid_graph()
    g.add_node(99, x=9999.0, y=9999.0)  # disconnected island
    engine = NetworkXEngine({"walk": g})
    pairs = [RoutableODPair("o1", node_id(0, 0), "iso", 99, "walk", "commute", weight=1.0)]
    (traj,) = list(od_pair_trajectories(pairs, engine, _settings()))
    assert not traj.reachable
    assert traj.reason == "unreachable"


# --------------------------------------------------------------------------- #
# Engine primitives                                                            #
# --------------------------------------------------------------------------- #


def test_engine_shortest_path_self_is_zero_cost(engine):
    res = engine.shortest_path(node_id(1, 1), node_id(1, 1), "walk", weight="length")
    assert res is not None
    assert res.cost == 0.0
    assert res.path == []


def test_engine_unknown_mode_raises(engine):
    from segx.errors import ModeNotRegisteredError

    with pytest.raises(ModeNotRegisteredError):
        engine.shortest_path(0, 1, "drive", weight="length")


def test_multidigraph_view_preserves_path_lengths():
    # converting the undirected synthetic graph to a MultiDiGraph view must not
    # change shortest-path lengths (the EdgeId-decoding decision).
    from segx.routing._paths import as_multidigraph

    g = build_grid_graph()
    mg = as_multidigraph(g)
    assert isinstance(mg, nx.MultiDiGraph)
    for a, b in [(0, 15), (5, 10), (12, 3)]:
        assert nx.shortest_path_length(mg, a, b, weight="length") == nx.shortest_path_length(g, a, b, weight="length")


# --------------------------------------------------------------------------- #
# Cross-engine check (Pandana vs NetworkX) — imports pandana lazily            #
# --------------------------------------------------------------------------- #


def test_pandana_matches_networkx_nearest_and_catchment(origins, pois_by_category):
    pytest.importorskip("pandana")
    from segx.routing.pandana_engine import PandanaEngine

    grid = build_grid_graph()
    nx_engine = NetworkXEngine({"walk": grid})
    pd_engine = PandanaEngine({"walk": grid})
    settings = _settings()

    nx_near = {
        (t.origin_id, t.category): t
        for t in nearest_poi_trajectories(origins, pois_by_category, nx_engine, settings, mode="walk")
    }
    pd_near = {
        (t.origin_id, t.category): t
        for t in nearest_poi_trajectories(origins, pois_by_category, pd_engine, settings, mode="walk")
    }
    for key, nx_t in nx_near.items():
        pd_t = pd_near[key]
        assert pd_t.destination_id == nx_t.destination_id or pd_t.cost == pytest.approx(nx_t.cost)  # ties allowed
        assert pd_t.cost == pytest.approx(nx_t.cost)

    nx_catch = _catchment_sets(
        list(catchment_trajectories(origins, pois_by_category, nx_engine, settings, mode="walk"))
    )
    pd_catch = _catchment_sets(
        list(catchment_trajectories(origins, pois_by_category, pd_engine, settings, mode="walk"))
    )
    assert nx_catch == pd_catch


def test_pandana_batched_shortest_paths_match_the_per_pair_call():
    """Binding C routes through the batched call; it must be the per-pair answer,
    pair for pair and in input order — duplicates, a self-pair and a disconnected
    pair included. The disconnected one is the case that matters: pandana reports
    it as ``4294967.295`` with an empty node path, which read as a *reachable*
    zero-edge route until the sentinel was corrected (invariant #2)."""
    pytest.importorskip("pandana")
    from segx.routing.pandana_engine import PandanaEngine

    grid = build_grid_graph()
    grid.add_node(99, x=9999.0, y=9999.0)  # disconnected island
    engine = PandanaEngine({"walk": grid})
    oracle = NetworkXEngine({"walk": grid})
    a, b, c = node_id(0, 0), node_id(3, 3), node_id(1, 2)
    pairs = [(a, b), (c, a), (a, 99), (a, a), (a, b), (b, c)]

    batched = engine.shortest_paths(pairs, "walk", weight="travel_time_min")
    single = [engine.shortest_path(s, d, "walk", weight="travel_time_min") for s, d in pairs]

    assert len(batched) == len(pairs)
    for (src, dst), got, want in zip(pairs, batched, single, strict=True):
        if dst == 99:
            assert got is None and want is None
            continue
        assert (got.target_node, got.cost, got.path) == (want.target_node, want.cost, want.path)
        assert got.cost == pytest.approx(oracle.shortest_path(src, dst, "walk", weight="travel_time_min").cost)
    assert batched[3].path == [] and batched[3].cost == 0.0
    # duplicates are searched once but never share a list — a transit path is rewritten downstream
    assert batched[0].path == batched[4].path and batched[0].path is not batched[4].path


def test_od_pairs_are_routed_per_mode_and_emitted_in_input_order():
    """Batching groups the pairs by mode; the trajectory stream must not."""
    g = build_grid_graph()
    g.add_node(99, x=9999.0, y=9999.0)
    engine = NetworkXEngine({"walk": g, "drive": g})
    pairs = [
        RoutableODPair("o1", node_id(0, 0), "d1", node_id(2, 2), "walk", "commute", 1.0),
        RoutableODPair("o2", node_id(0, 0), "d2", node_id(3, 0), "drive", "commute", 2.0),
        RoutableODPair("o3", node_id(0, 0), "d3", 99, "walk", "commute", 3.0),
        RoutableODPair("o4", node_id(1, 1), "d4", node_id(0, 1), "drive", "commute", 4.0),
    ]
    trajs = list(od_pair_trajectories(pairs, engine, _settings()))
    assert [(t.origin_id, t.destination_id, t.mode, t.weight) for t in trajs] == [
        (p.origin_id, p.dest_id, p.mode, p.weight) for p in pairs
    ]
    assert [t.reachable for t in trajs] == [True, True, False, True]
    assert trajs[2].reason == "unreachable"


# --------------------------------------------------------------------------- #
# Batched access bindings (nearest / catchment over many origins at once)       #
# --------------------------------------------------------------------------- #


def _island_grid() -> nx.Graph:
    g = build_grid_graph()
    g.add_node(99, x=9999.0, y=9999.0)  # disconnected island
    return g


def test_pandana_batched_nearest_and_catchment_match_the_oracle(monkeypatch):
    """The batched forms measure an origins x POIs cost matrix and pick winners with
    NumPy; they must give the oracle's answers for every origin, in order — with a
    repeated origin node, an unreachable origin, an unreachable POI, two POIs on
    one node (the tie-break is the smaller ``poi_id``, whatever order they come in)
    and the matrix cut into several blocks."""
    pytest.importorskip("pandana")
    from segx.routing import pandana_engine
    from segx.routing.pandana_engine import PandanaEngine

    grid = _island_grid()
    pd_engine = PandanaEngine({"walk": grid})
    nx_engine = NetworkXEngine({"walk": grid})
    monkeypatch.setattr(pandana_engine, "_MATRIX_CELLS", 7)  # 2 origins per block with 3 POI nodes

    pois = [
        RoutablePOI("z_late", node_id(3, 3)),
        RoutablePOI("a_early", node_id(3, 3)),  # same node, smaller id: must win the tie
        RoutablePOI("mid", node_id(0, 2)),
        RoutablePOI("island", 99),
    ]
    nodes = [node_id(0, 0), node_id(3, 2), 99, node_id(0, 0), node_id(1, 1), node_id(2, 3)]

    got = pd_engine.nearest_many(nodes, pois, "walk", weight="length")
    want = [nx_engine.nearest(n, pois, "walk", weight="length") for n in nodes]
    assert [(r.target_id, r.target_node) for r in got] == [(r.target_id, r.target_node) for r in want]
    for g, w in zip(got, want, strict=True):
        assert g.cost == pytest.approx(w.cost) and len(g.path) == len(w.path)
    assert got[1].target_id == "a_early"
    assert got[2].target_id == "island" and got[2].cost == 0.0 and got[2].path == []
    assert got[0].path == got[3].path and got[0].path is not got[3].path  # repeated node, own list each

    budget = 3 * EDGE_M
    got_r = pd_engine.reachable_within_many(nodes, pois, "walk", weight="length", budget=budget)
    want_r = [nx_engine.reachable_within(n, pois, "walk", weight="length", budget=budget) for n in nodes]
    assert [[r.target_id for r in rs] for rs in got_r] == [[r.target_id for r in rs] for rs in want_r]
    for gs, ws in zip(got_r, want_r, strict=True):
        for g, w in zip(gs, ws, strict=True):
            assert g.cost == pytest.approx(w.cost) and len(g.path) == len(w.path)

    ids_only = pd_engine.reachable_within_many(nodes, pois, "walk", weight="length", budget=budget, with_paths=False)
    assert [[r.target_id for r in rs] for rs in ids_only] == [[r.target_id for r in rs] for rs in got_r]
    assert all(r.path == [] for rs in ids_only for r in rs)

    # one origin is the same code path as many
    assert pd_engine.nearest(node_id(3, 2), pois, "walk", weight="length").target_id == "a_early"
    assert pd_engine.nearest_many([], pois, "walk", weight="length") == []
    assert pd_engine.nearest_many(nodes[:2], [], "walk", weight="length") == [None, None]


def test_pandana_nearest_is_none_when_no_poi_is_reachable():
    pytest.importorskip("pandana")
    from segx.routing.pandana_engine import PandanaEngine

    engine = PandanaEngine({"walk": _island_grid()})
    island_only = [RoutablePOI("island", 99)]
    assert engine.nearest_many([node_id(0, 0), 99], island_only, "walk", weight="length")[0] is None
    assert engine.reachable_within(node_id(0, 0), island_only, "walk", weight="length", budget=1e9) == []


@pytest.mark.parametrize("block", [1, 2, 1000])
def test_access_strategies_emit_the_same_stream_whatever_the_block_size(
    monkeypatch, engine, origins, pois_by_category, block
):
    """Origins are routed a block per engine call; the block size is a memory knob
    and must not show in the output — same trajectories, same order."""
    from segx.routing import nearest_poi, poi_catchment

    def key(t):
        return (t.origin_id, t.destination_id, t.category, t.binding, t.weight, t.path, t.cost, t.reason, t.meta)

    settings = _settings()
    want_near = [key(t) for t in nearest_poi_trajectories(origins, pois_by_category, engine, settings, mode="walk")]
    want_catch = [key(t) for t in catchment_trajectories(origins, pois_by_category, engine, settings, mode="walk")]

    monkeypatch.setattr(nearest_poi, "ORIGIN_BLOCK", block)
    monkeypatch.setattr(poi_catchment, "ORIGIN_BLOCK", block)
    assert [
        key(t) for t in nearest_poi_trajectories(origins, pois_by_category, engine, settings, mode="walk")
    ] == want_near
    assert [
        key(t) for t in catchment_trajectories(origins, pois_by_category, engine, settings, mode="walk")
    ] == want_catch
    # origin by origin, category by category — the order the unbatched loop produced
    assert [k[0] for k in want_near] == [o.origin_id for o in origins for _ in settings.poi.categories_for("nearest")]


def test_catchment_with_two_budgets_keeps_only_pois_passing_both(engine, origins, pois_by_category):
    """Time and distance budgets together: the secondary one only filters (it is
    asked without paths), and the reported cost stays the primary's."""
    loose = _settings(distance_budget_m=CATCHMENT_BUDGET_STEPS * EDGE_M, budget_minutes=1e6)
    tight = _settings(distance_budget_m=CATCHMENT_BUDGET_STEPS * EDGE_M, budget_minutes=1 * EDGE_WALK_MIN)
    only_distance = _settings(distance_budget_m=CATCHMENT_BUDGET_STEPS * EDGE_M)

    def sets(settings):
        return _catchment_sets(list(catchment_trajectories(origins, pois_by_category, engine, settings, mode="walk")))

    assert sets(loose) == sets(only_distance)  # a budget nothing exceeds changes nothing
    one_step = _settings(distance_budget_m=1 * EDGE_M)
    assert sets(tight) == sets(one_step)  # the tighter of the two wins
    for t in catchment_trajectories(origins, pois_by_category, engine, tight, mode="walk"):
        assert t.cost == pytest.approx(len(t.path) * EDGE_WALK_MIN)  # primary = time
