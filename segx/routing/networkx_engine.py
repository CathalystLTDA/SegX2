"""Validation / fallback routing engine: Dijkstra on the per-mode NetworkX
graphs. Too slow for city scale; used as the oracle that cross-checks Pandana on
the synthetic city and on sampled real-city subsets (runs in CI). P3.

Each mode graph is held as a ``MultiDiGraph`` view (see
:func:`segx.routing._paths.as_multidigraph`) so paths decode to the same
``(u, v, key)`` edge ids the default engine and the aggregator use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from segx.errors import ModeNotRegisteredError
from segx.routing._paths import as_multidigraph, node_path_to_edges
from segx.routing.engine import RouteResult

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from segx.config import Settings
    from segx.routing.inputs import RoutablePOI

#: distances within this slack of the budget still count as reachable (float guard)
_BUDGET_EPS = 1e-9


class NetworkXEngine:
    """See :class:`segx.routing.engine.RoutingEngine`. Pure-Python Dijkstra oracle."""

    def __init__(self, graphs_by_mode: Mapping[str, nx.Graph], settings: Settings | None = None):  # noqa: ARG002
        self._graphs: dict[str, nx.MultiDiGraph] = {m: as_multidigraph(g) for m, g in graphs_by_mode.items()}

    def _graph(self, mode: str) -> nx.MultiDiGraph:
        try:
            return self._graphs[mode]
        except KeyError:
            raise ModeNotRegisteredError(
                f"NetworkXEngine has no graph for mode {mode!r}; available: {sorted(self._graphs)}",
                details={"mode": mode, "available": sorted(self._graphs)},
            ) from None

    def nearest(self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str) -> RouteResult | None:
        if not pois:
            return None
        graph = self._graph(mode)
        dist, paths = nx.single_source_dijkstra(graph, origin_node, weight=weight)
        # min over reachable POIs by (cost, poi_id) for a deterministic tie-break.
        best: tuple[float, str, RoutablePOI] | None = None
        for poi in pois:
            if poi.node in dist:
                cand = (dist[poi.node], poi.poi_id, poi)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best is None:
            return None
        cost, _, poi = best
        return RouteResult(poi.poi_id, poi.node, float(cost), node_path_to_edges(graph, paths[poi.node], weight))

    def reachable_within(
        self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str, budget: float
    ) -> list[RouteResult]:
        if not pois:
            return []
        graph = self._graph(mode)
        dist, paths = nx.single_source_dijkstra(graph, origin_node, cutoff=budget, weight=weight)
        out: list[RouteResult] = []
        for poi in pois:
            cost = dist.get(poi.node)
            if cost is not None and cost <= budget + _BUDGET_EPS:
                out.append(
                    RouteResult(poi.poi_id, poi.node, float(cost), node_path_to_edges(graph, paths[poi.node], weight))
                )
        return out

    def shortest_path(self, src_node: int, dst_node: int, mode: str, *, weight: str) -> RouteResult | None:
        graph = self._graph(mode)
        try:
            cost, path = nx.single_source_dijkstra(graph, src_node, target=dst_node, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return RouteResult(str(dst_node), dst_node, float(cost), node_path_to_edges(graph, path, weight))

    def nearest_many(self, origin_nodes, pois: list[RoutablePOI], mode: str, *, weight: str):
        """One :meth:`nearest` per origin node, in order (a repeated node is searched once)."""
        found = {n: self.nearest(n, pois, mode, weight=weight) for n in dict.fromkeys(int(n) for n in origin_nodes)}
        return [_own_path(found[int(n)]) for n in origin_nodes]

    def reachable_within_many(
        self, origin_nodes, pois: list[RoutablePOI], mode: str, *, weight: str, budget: float, with_paths: bool = True
    ):
        """One :meth:`reachable_within` per origin node, in order. ``with_paths`` is
        accepted for interface parity; Dijkstra yields the paths either way."""
        found = {
            n: self.reachable_within(n, pois, mode, weight=weight, budget=budget)
            for n in dict.fromkeys(int(n) for n in origin_nodes)
        }
        return [[_own_path(r) for r in found[int(n)]] for n in origin_nodes]

    def shortest_paths(self, pairs, mode: str, *, weight: str) -> list[RouteResult | None]:
        """One :meth:`shortest_path` per pair, in order — the oracle has nothing to batch."""
        return [self.shortest_path(int(src), int(dst), mode, weight=weight) for src, dst in pairs]


def _own_path(result: RouteResult | None) -> RouteResult | None:
    """A copy of *result* with its own ``path`` list — two origins on one node must
    not share a list that a later stage may rewrite."""
    if result is None:
        return None
    return RouteResult(result.target_id, result.target_node, result.cost, list(result.path))


__all__ = ["NetworkXEngine"]
