"""Default routing engine: a Pandana network per mode (contraction-hierarchy
backend) — the only engine fast enough for city-scale 10^5–10^6 origin queries.
Core dependency from P3. P3.

Each mode keeps two parallel structures, built once at construction:

* a Pandana ``Network`` carrying both impedance columns (``length`` in metres and
  ``travel_time_min``), used for the actual shortest-path / nearest queries;
* the ``MultiDiGraph`` view it was built from, used only to decode a returned node
  sequence into ``(u, v, key)`` edge ids — the *same* decoder the oracle uses
  (:func:`segx.routing._paths.node_path_to_edges`), so both engines emit
  byte-identical ``Trajectory.path`` values and the cross-engine check is exact.

Every binding goes through pandana's **vectorised** point-to-point calls
(``shortest_path_lengths`` + ``shortest_paths``), which run on the contraction
hierarchy and are correct and version-robust. A scalar pandana call translates
its two node ids with a ``pandas.merge`` — about a millisecond of fixed overhead
per call, which at city scale is most of the routing time and none of the
routing. The vectorised calls pay it once per batch.

* Binding C (OD pairs): :meth:`PandanaEngine.shortest_paths`. Identical
  ``(src, dst)`` node pairs (many OD rows snap to the same two nodes) are searched
  and decoded once.
* Bindings A and B (nearest / catchment): :meth:`PandanaEngine.nearest_many` and
  :meth:`PandanaEngine.reachable_within_many` measure the whole *origins × POIs*
  cost matrix in bounded chunks, pick the winners with NumPy, and only then ask
  for paths — for the pairs that are actually emitted. The single-origin
  ``nearest`` / ``reachable_within`` are the same code with one origin.

Pandana's own ``nearest_pois`` is intentionally not used: it answers "the k
nearest within a radius" on one impedance, and bindings A/B need the engine's
exact tie-break and budget rules, shared with the NetworkX oracle.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandana
import pandas as pd

from segx.errors import ModeNotRegisteredError
from segx.routing._paths import as_multidigraph, node_path_to_edges
from segx.routing.engine import RouteResult

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    import networkx as nx

    from segx.config import Settings
    from segx.routing.inputs import RoutablePOI

#: Pandana reports a disconnected pair as ``4294967.295`` (the uint32 maximum in
#: its internal milli-units), for both the scalar and the vectorised calls. Any
#: cost at or above this is "unreachable" — a real cost that large would be
#: 4 295 km or 8 years of travel time, far outside any study area.
_UNREACHABLE = 4294967.0
_BUDGET_EPS = 1e-9
#: OD pairs handed to pandana per vectorised call — bounds the transient arrays.
_BATCH_SIZE = 50_000
#: origin × POI cost-matrix cells measured per vectorised call (bindings A/B).
_MATRIX_CELLS = 1_000_000


class PandanaEngine:
    """See :class:`segx.routing.engine.RoutingEngine`. Default city-scale engine."""

    def __init__(self, graphs_by_mode: Mapping[str, nx.Graph], settings: Settings | None = None):  # noqa: ARG002
        self._graphs: dict[str, nx.MultiDiGraph] = {}
        self._nets: dict[str, pandana.Network] = {}
        for mode, graph in graphs_by_mode.items():
            mg = as_multidigraph(graph)
            self._graphs[mode] = mg
            self._nets[mode] = self._build_network(mg)

    @staticmethod
    def _build_network(graph: nx.MultiDiGraph) -> pandana.Network:
        node_ids = list(graph.nodes)
        xs = pd.Series([float(graph.nodes[n]["x"]) for n in node_ids], index=node_ids, dtype=float)
        ys = pd.Series([float(graph.nodes[n]["y"]) for n in node_ids], index=node_ids, dtype=float)
        froms: list[int] = []
        tos: list[int] = []
        lengths: list[float] = []
        times: list[float] = []
        for u, v, data in graph.edges(data=True):
            length = float(data.get("length", 1.0))
            time = data.get("travel_time_min")
            froms.append(u)
            tos.append(v)
            lengths.append(length)
            times.append(float(time) if time is not None else length)
        edge_weights = pd.DataFrame({"length": lengths, "travel_time_min": times})
        # Edges are already directed (MultiDiGraph view) → twoway=False.
        return pandana.Network(xs, ys, pd.Series(froms), pd.Series(tos), edge_weights, twoway=False)

    def _net(self, mode: str) -> pandana.Network:
        try:
            return self._nets[mode]
        except KeyError:
            raise ModeNotRegisteredError(
                f"PandanaEngine has no graph for mode {mode!r}; available: {sorted(self._nets)}",
                details={"mode": mode, "available": sorted(self._nets)},
            ) from None

    def _route(self, mode: str, src: int, dst: int, weight: str) -> tuple[float, list] | None:
        net = self._net(mode)
        cost = float(net.shortest_path_length(src, dst, imp_name=weight))
        if cost >= _UNREACHABLE:
            return None
        nodes = [int(n) for n in net.shortest_path(src, dst, imp_name=weight)]
        return cost, node_path_to_edges(self._graphs[mode], nodes, weight)

    def _cost_matrices(self, origin_nodes: list[int], poi_nodes: np.ndarray, mode: str, weight: str):
        """Yield ``(first origin index, cost matrix)`` blocks of *origins × POI nodes*.

        One vectorised pandana call per block, sized so a block never holds more
        than ``_MATRIX_CELLS`` costs — the full matrix of a city-scale run would
        not fit, and nothing downstream needs more than a row at a time.
        """
        net = self._net(mode)
        n_pois = len(poi_nodes)
        rows = max(1, _MATRIX_CELLS // n_pois)
        for start in range(0, len(origin_nodes), rows):
            block = np.asarray(origin_nodes[start : start + rows], dtype=np.int64)
            with warnings.catch_warnings():
                # an unconnected pair is a handled result here, not a fault
                warnings.filterwarnings("ignore", message="Unsigned integer: shortest path")
                costs = net.shortest_path_lengths(
                    np.repeat(block, n_pois), np.tile(poi_nodes, len(block)), imp_name=weight
                )
            yield start, np.asarray(costs, dtype=float).reshape(len(block), n_pois)

    def _paths(self, pairs: list[tuple[int, int]], mode: str, weight: str) -> dict[tuple[int, int], list]:
        """Decoded edge path per distinct ``(src, dst)`` pair (``[]`` if unroutable)."""
        net = self._net(mode)
        graph = self._graphs[mode]
        out: dict[tuple[int, int], list] = {}
        unique = list(dict.fromkeys(pairs))
        for start in range(0, len(unique), _BATCH_SIZE):
            chunk = unique[start : start + _BATCH_SIZE]
            node_paths = net.shortest_paths([s for s, _ in chunk], [d for _, d in chunk], imp_name=weight)
            for key, nodes in zip(chunk, node_paths, strict=True):
                out[key] = node_path_to_edges(graph, [int(n) for n in nodes], weight)
        return out

    def nearest_many(
        self, origin_nodes: Sequence[int], pois: list[RoutablePOI], mode: str, *, weight: str
    ) -> list[RouteResult | None]:
        """:meth:`nearest` for many origins: one result per origin node, in order.

        The winner is the reachable POI with the smallest ``(cost, poi_id)`` — the
        same deterministic tie-break as the per-origin call and the oracle.
        """
        origin_nodes = [int(n) for n in origin_nodes]
        if not pois or not origin_nodes:
            return [None] * len(origin_nodes)
        unique_origins = list(dict.fromkeys(origin_nodes))
        # POIs in ``poi_id`` order (stable): the first minimum of a row is then the
        # smallest id among equal costs, which is what ``argmin`` returns.
        ranked = sorted(pois, key=lambda poi: poi.poi_id)
        poi_nodes, columns = np.unique(np.asarray([poi.node for poi in ranked], dtype=np.int64), return_inverse=True)

        chosen: dict[int, tuple[RoutablePOI, float] | None] = {}
        for start, costs in self._cost_matrices(unique_origins, poi_nodes, mode, weight):
            per_poi = costs[:, columns]
            winners = per_poi.argmin(axis=1)
            best = per_poi[np.arange(len(winners)), winners]
            for offset, (winner, cost) in enumerate(zip(winners.tolist(), best.tolist(), strict=True)):
                chosen[unique_origins[start + offset]] = None if cost >= _UNREACHABLE else (ranked[winner], cost)

        paths = self._paths([(o, hit[0].node) for o, hit in chosen.items() if hit is not None], mode, weight)
        out: list[RouteResult | None] = []
        for origin in origin_nodes:
            hit = chosen[origin]
            if hit is None:
                out.append(None)
            else:
                poi, cost = hit
                out.append(RouteResult(poi.poi_id, poi.node, cost, list(paths[(origin, poi.node)])))
        return out

    def reachable_within_many(
        self,
        origin_nodes: Sequence[int],
        pois: list[RoutablePOI],
        mode: str,
        *,
        weight: str,
        budget: float,
        with_paths: bool = True,
    ) -> list[list[RouteResult]]:
        """:meth:`reachable_within` for many origins: one list per origin node, in order.

        Each list holds the POIs within *budget*, in the order *pois* gave them.
        ``with_paths=False`` skips path decoding (every ``path`` is empty) — for a
        secondary budget constraint, which only needs to know *which* POIs pass.
        """
        origin_nodes = [int(n) for n in origin_nodes]
        if not pois or not origin_nodes:
            return [[] for _ in origin_nodes]
        unique_origins = list(dict.fromkeys(origin_nodes))
        poi_nodes, columns = np.unique(np.asarray([poi.node for poi in pois], dtype=np.int64), return_inverse=True)

        hits: dict[int, list[tuple[RoutablePOI, float]]] = {}
        for start, costs in self._cost_matrices(unique_origins, poi_nodes, mode, weight):
            per_poi = costs[:, columns]
            within = (per_poi < _UNREACHABLE) & (per_poi <= budget + _BUDGET_EPS)
            for offset in range(len(per_poi)):
                indexes = np.flatnonzero(within[offset]).tolist()
                row = per_poi[offset]
                hits[unique_origins[start + offset]] = [(pois[i], float(row[i])) for i in indexes]

        paths: dict[tuple[int, int], list] = {}
        if with_paths:
            paths = self._paths([(o, poi.node) for o, found in hits.items() for poi, _ in found], mode, weight)
        return [
            [
                RouteResult(poi.poi_id, poi.node, cost, list(paths[(origin, poi.node)]) if with_paths else [])
                for poi, cost in hits[origin]
            ]
            for origin in origin_nodes
        ]

    def nearest(self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str) -> RouteResult | None:
        return self.nearest_many([origin_node], pois, mode, weight=weight)[0]

    def reachable_within(
        self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str, budget: float
    ) -> list[RouteResult]:
        return self.reachable_within_many([origin_node], pois, mode, weight=weight, budget=budget)[0]

    def shortest_path(self, src_node: int, dst_node: int, mode: str, *, weight: str) -> RouteResult | None:
        routed = self._route(mode, src_node, dst_node, weight)
        if routed is None:
            return None
        cost, path = routed
        return RouteResult(str(dst_node), dst_node, cost, path)

    def shortest_paths(self, pairs: Sequence[tuple[int, int]], mode: str, *, weight: str) -> list[RouteResult | None]:
        """Batched :meth:`shortest_path`: one result per ``(src, dst)`` pair, in order.

        Same answers as calling :meth:`shortest_path` per pair (``None`` for a
        disconnected one), without the per-call id translation. Each distinct
        pair is searched and decoded once; every result still gets its **own**
        ``path`` list, because a transit path is rewritten downstream.
        """
        net = self._net(mode)
        graph = self._graphs[mode]
        unique = list(dict.fromkeys((int(src), int(dst)) for src, dst in pairs))
        routed: dict[tuple[int, int], tuple[float, list] | None] = {}
        for start in range(0, len(unique), _BATCH_SIZE):
            chunk = unique[start : start + _BATCH_SIZE]
            srcs = [src for src, _ in chunk]
            dsts = [dst for _, dst in chunk]
            with warnings.catch_warnings():
                # Pandana warns once per batch listing every unconnected pair; they
                # are a handled result here (``None``), not a fault.
                warnings.filterwarnings("ignore", message="Unsigned integer: shortest path")
                costs = net.shortest_path_lengths(srcs, dsts, imp_name=weight)
            node_paths = net.shortest_paths(srcs, dsts, imp_name=weight)
            for key, cost, nodes in zip(chunk, costs, node_paths, strict=True):
                if cost >= _UNREACHABLE:
                    routed[key] = None
                else:
                    routed[key] = (float(cost), node_path_to_edges(graph, [int(n) for n in nodes], weight))
        out: list[RouteResult | None] = []
        for src, dst in pairs:
            hit = routed[(int(src), int(dst))]
            out.append(None if hit is None else RouteResult(str(dst), int(dst), hit[0], list(hit[1])))
        return out


__all__ = ["PandanaEngine"]
