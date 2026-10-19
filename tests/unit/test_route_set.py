"""Route-set per OD pair (P4.5): link-penalty vs. exact Yen, choice weights, and
weight conservation when a flow splits across k routes. Runs on the synthetic
grid (pure NetworkX — no pandas), so it stays in the offline suite."""

from __future__ import annotations

import math

import pytest

from segx.aggregation.edge_accumulator import EdgeAccumulator
from segx.config import DemographicAttribute, DemographicSchema, RouteSetConfig, RoutingConfig, Settings
from segx.routing.inputs import RoutableODPair
from segx.routing.networkx_engine import NetworkXEngine
from segx.routing.od_router import od_pair_trajectories
from segx.routing.route_set import link_penalty_routes, route_weights, yen_routes
from segx.validation.invariants import check_weight_conservation
from segx.validation.synthetic_city import build_grid_graph, node_id

WEIGHT = "travel_time_min"


def _corner_pair():
    return node_id(0, 0), node_id(3, 3)


def _edge_set(route):
    return {(min(u, v), max(u, v), k) for (u, v, k) in route.edges}


def _mean_pairwise_overlap(routes) -> float:
    """Mean Jaccard overlap of edge sets across a route-set (0 = all disjoint)."""
    sets = [_edge_set(r) for r in routes]
    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    if not pairs:
        return 0.0
    vals = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in pairs]
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
# route generation                                                            #
# --------------------------------------------------------------------------- #


def test_link_penalty_returns_k_distinct_routes():
    g = build_grid_graph()
    src, dst = _corner_pair()
    routes = link_penalty_routes(g, src, dst, WEIGHT, k=3, penalty=1.5)
    assert len(routes) == 3
    node_paths = {tuple(r.nodes) for r in routes}
    assert len(node_paths) == 3  # all distinct


def test_yen_returns_routes_in_nondecreasing_cost():
    g = build_grid_graph()
    src, dst = _corner_pair()
    routes = yen_routes(g, src, dst, WEIGHT, k=4)
    assert len(routes) == 4
    costs = [r.cost for r in routes]
    assert costs == sorted(costs)
    assert costs[0] == pytest.approx(min(costs))  # route 1 is a true shortest path


def test_link_penalty_more_distinct_than_yen():
    # The headline comparison: link-penalty spreads; Yen returns near-duplicates.
    g = build_grid_graph()
    src, dst = _corner_pair()
    lp = link_penalty_routes(g, src, dst, WEIGHT, k=3, penalty=1.6)
    yen = yen_routes(g, src, dst, WEIGHT, k=3)
    assert _mean_pairwise_overlap(lp) < _mean_pairwise_overlap(yen)


def test_same_node_is_single_zero_cost_route():
    g = build_grid_graph()
    n = node_id(1, 1)
    for routes in (link_penalty_routes(g, n, n, WEIGHT, 3, 1.5), yen_routes(g, n, n, WEIGHT, 3)):
        assert len(routes) == 1
        assert routes[0].cost == 0.0
        assert routes[0].edges == []


# --------------------------------------------------------------------------- #
# choice weights                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", ["uniform", "inverse_cost", "path_size_logit"])
def test_weights_sum_to_one(scheme):
    g = build_grid_graph()
    src, dst = _corner_pair()
    routes = link_penalty_routes(g, src, dst, WEIGHT, k=3, penalty=1.5)
    w = route_weights(routes, scheme)
    assert math.fsum(w) == pytest.approx(1.0)
    assert all(x >= 0 for x in w)
    assert [r.weight for r in routes] == w  # written back onto the routes


def test_uniform_is_equal_split():
    g = build_grid_graph()
    src, dst = _corner_pair()
    routes = yen_routes(g, src, dst, WEIGHT, k=4)
    assert route_weights(routes, "uniform") == pytest.approx([0.25] * 4)


def test_single_route_takes_all_mass():
    g = build_grid_graph()
    routes = link_penalty_routes(g, node_id(0, 0), node_id(1, 0), WEIGHT, k=3, penalty=1.5)
    # adjacent nodes: only one sensible route survives dedup
    w = route_weights(routes, "path_size_logit")
    assert math.fsum(w) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# integration: od_router emits k trajectories, conservation holds             #
# --------------------------------------------------------------------------- #


def _settings(k: int, method: str = "link_penalty") -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        ),
        routing=RoutingConfig(route_set=RouteSetConfig(k=k, method=method, weight_scheme="uniform")),
    )


def _pair():
    src, dst = _corner_pair()
    return RoutableODPair("o1", src, "od0", dst, mode="walk", category="commute", weight=1.0)


def test_route_set_off_by_default_yields_single_trajectory():
    g = build_grid_graph()
    engine = NetworkXEngine({"walk": g})
    trajs = list(od_pair_trajectories([_pair()], engine, _settings(k=1), graphs={"walk": g}))
    assert len(trajs) == 1
    assert trajs[0].route_rank == 0
    assert trajs[0].route_weight == 1.0


@pytest.mark.parametrize("method", ["link_penalty", "yen"])
def test_route_set_emits_k_trajectories_summing_to_one(method):
    g = build_grid_graph()
    trajs = list(od_pair_trajectories([_pair()], None, _settings(k=3, method=method), graphs={"walk": g}))
    assert len(trajs) == 3
    assert {t.route_rank for t in trajs} == {0, 1, 2}
    assert math.fsum(t.route_weight for t in trajs) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["link_penalty", "yen"])
def test_split_flow_conserves_mass(method):
    """k routes each deposit route_weight × vector; the pair's total is unchanged."""
    g = build_grid_graph()
    settings = _settings(k=3, method=method)
    trajs = list(od_pair_trajectories([_pair()], None, settings, graphs={"walk": g}))
    vector = {"income": {"low": 4.0, "high": 2.0}}  # total 6 workers on the flow

    acc = EdgeAccumulator(settings)
    for t in trajs:
        scaled = {a: {c: m * t.route_weight for c, m in cls.items()} for a, cls in vector.items()}
        acc.add(t, scaled)

    ledger = acc.ledger()
    assert ledger.emitted == pytest.approx(6.0)  # 6·Σwᵢ = 6, not 6·k
    assert ledger.deposited == pytest.approx(6.0)
    check_weight_conservation(ledger)
