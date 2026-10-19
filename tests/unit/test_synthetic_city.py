"""P0: the synthetic-city fixture — graph structure plus consistency of the
hand-computed ground truth with the graph itself.

End-to-end assertions (per-edge demographic sums, segment entropy via the full
routing+aggregation pipeline) live in ``tests/integration/`` and switch on in
P3/P4.
"""

from __future__ import annotations

import networkx as nx
import pytest

from segx.metrics.entropy import shannon_entropy
from segx.validation.synthetic_city import (
    CATCHMENT_BUDGET_STEPS,
    CATCHMENT_GROUND_TRUTH,
    EDGE_M,
    EXPECTED_ORIGIN_ENTROPY_BITS,
    GRID_N,
    NEAREST_POI_GROUND_TRUTH,
    ORIGIN_DEMOGRAPHICS,
    ORIGIN_NODES,
    POI_CATEGORIES,
    POIS,
    build_grid_graph,
    manhattan_steps,
    node_colrow,
    node_id,
)


@pytest.fixture(scope="module")
def grid() -> nx.Graph:
    return build_grid_graph()


def test_grid_structure(grid: nx.Graph):
    assert grid.number_of_nodes() == GRID_N * GRID_N == 16
    # horizontal edges: (GRID_N - 1) per row * GRID_N rows; vertical: symmetric
    assert grid.number_of_edges() == 2 * (GRID_N - 1) * GRID_N == 24
    for nid, data in grid.nodes(data=True):
        c, r = node_colrow(nid)
        assert (data["x"], data["y"]) == (c * EDGE_M, r * EDGE_M)
    for _, _, data in grid.edges(data=True):
        assert data["length"] == EDGE_M
        assert data["travel_time_min"] > 0


def test_node_id_roundtrip():
    for r in range(GRID_N):
        for c in range(GRID_N):
            assert node_colrow(node_id(c, r)) == (c, r)
    with pytest.raises(ValueError):
        node_id(GRID_N, 0)


def test_manhattan_equals_graph_distance(grid: nx.Graph):
    # On a complete grid, hop-count and length-weighted shortest paths both
    # equal the Manhattan distance.
    sample = [(0, 15), (5, 10), (4, 7), (12, 3), (1, 14)]
    for a, b in sample:
        assert nx.shortest_path_length(grid, a, b) == manhattan_steps(a, b)
        assert nx.shortest_path_length(grid, a, b, weight="length") == manhattan_steps(a, b) * EDGE_M


def test_origin_entropies_match_demographics():
    for oid, demo in ORIGIN_DEMOGRAPHICS.items():
        assert shannon_entropy(demo.values()) == pytest.approx(EXPECTED_ORIGIN_ENTROPY_BITS[oid], abs=1e-12)


def _nearest_by_hand(origin_node: int) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    for category in POI_CATEGORIES:
        candidates = [(pid, manhattan_steps(origin_node, node)) for pid, (cat, node) in POIS.items() if cat == category]
        out[category] = min(candidates, key=lambda t: t[1])
    return out


def test_nearest_poi_ground_truth_is_consistent():
    for oid, origin_node in ORIGIN_NODES.items():
        assert _nearest_by_hand(origin_node) == NEAREST_POI_GROUND_TRUTH[oid]


def _catchment_by_hand(origin_node: int, budget_steps: int) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for category in POI_CATEGORIES:
        out[category] = frozenset(
            pid
            for pid, (cat, node) in POIS.items()
            if cat == category and manhattan_steps(origin_node, node) <= budget_steps
        )
    return out


def test_catchment_ground_truth_is_consistent():
    for oid, origin_node in ORIGIN_NODES.items():
        assert _catchment_by_hand(origin_node, CATCHMENT_BUDGET_STEPS) == CATCHMENT_GROUND_TRUTH[oid]


def test_catchment_shrinks_with_tighter_budget():
    any_strictly_smaller = False
    for origin_node in ORIGIN_NODES.values():
        wide = _catchment_by_hand(origin_node, CATCHMENT_BUDGET_STEPS)
        tight = _catchment_by_hand(origin_node, CATCHMENT_BUDGET_STEPS - 1)
        for category in POI_CATEGORIES:
            assert tight[category] <= wide[category]  # subset
            if tight[category] < wide[category]:
                any_strictly_smaller = True
    assert any_strictly_smaller, "expected at least one catchment to shrink when the budget tightens"


def test_two_parks_so_dual_binding_is_exercised():
    parks = [pid for pid, (cat, _) in POIS.items() if cat == "park"]
    assert len(parks) >= 2
    # and at least one origin's nearest park differs from another's
    nearest_parks = {NEAREST_POI_GROUND_TRUTH[oid]["park"][0] for oid in ORIGIN_NODES}
    assert len(nearest_parks) >= 2
