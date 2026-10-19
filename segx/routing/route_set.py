"""Route-set generation per OD pair (P4.5): *k* alternative routes instead of the
single geodesic, so a commute flow's demographic mass spreads over plausible
route choices.

Two methods, both selectable at runtime (`routing.route_set.method`) and compared
in the demo:

* **link-penalty** (``link_penalty_routes``) — behavioural: take the shortest
  path, multiply its edges' weights by a penalty factor, recompute; repeat. Every
  route is a shortest path over positive weights, hence simple (loopless) by
  construction. Alternatives are spatially *distinct* (the penalty pushes them off
  the used corridor).
* **exact k-shortest** (``yen_routes``) — NetworkX ``shortest_simple_paths``
  (Yen/Lawler): the provably next-shortest *simple* paths. Alternatives are
  optimal but tend to be near-duplicates of route 1 (differ by a block).

Both run on a **simple min-weight ``DiGraph`` view** of the mode graph (so
``shortest_simple_paths``, which rejects multigraphs, works and both methods are
node-path comparable), then decode node paths back to ``(u, v, key)`` ``EdgeId``s
on the original ``MultiDiGraph`` via :func:`segx.routing._paths.node_path_to_edges`
— the same edge identity the aggregator uses.

The flow's mass is split across the routes by a **choice weight** ``wᵢ`` with
``Σ wᵢ = 1`` (:func:`route_weights`); because the shares sum to 1 the
weight-conservation ledger is unaffected — only the spatial distribution changes.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import networkx as nx

from segx.routing._paths import as_multidigraph, node_path_to_edges

if TYPE_CHECKING:  # pragma: no cover
    from segx.routing.trajectories import EdgeId

WeightScheme = Literal["uniform", "inverse_cost", "path_size_logit"]
Method = Literal["link_penalty", "yen"]


@dataclass(slots=True)
class Route:
    """One route of a route-set: its node sequence, decoded edges, and true cost.

    ``cost`` is always the *unpenalised* cost on the mode graph (for link-penalty
    the penalty only steers which path is found, never its reported cost).
    """

    nodes: list[int]
    edges: list[EdgeId]
    cost: float
    weight: float = field(default=1.0)  # choice weight wᵢ, filled by route_weights()


def simple_digraph_view(graph: nx.Graph, weight: str) -> nx.DiGraph:
    """A plain ``DiGraph`` keeping, for each ``(u, v)``, the min-``weight`` parallel
    edge. This is the routing view both methods search over; edges are decoded back
    to real ``(u, v, key)`` ids on the original graph afterwards.
    """
    mg = as_multidigraph(graph)
    d = nx.DiGraph()
    d.add_nodes_from(mg.nodes(data=True))
    for u, v, data in mg.edges(data=True):
        w = data.get(weight)
        w = float(w) if w is not None else float(data.get("length", 1.0))
        if not d.has_edge(u, v) or w < d[u][v][weight]:
            d.add_edge(u, v, **{weight: w})
    return d


def _path_cost(view: nx.DiGraph, nodes: list[int], weight: str) -> float:
    return math.fsum(view[u][v][weight] for u, v in zip(nodes, nodes[1:], strict=False))


def _make_route(view: nx.DiGraph, decode_graph: nx.MultiDiGraph, nodes: list[int], weight: str) -> Route:
    edges = node_path_to_edges(decode_graph, nodes, weight)
    return Route(nodes=nodes, edges=edges, cost=_path_cost(view, nodes, weight))


def link_penalty_routes(graph: nx.Graph, src: int, dst: int, weight: str, k: int, penalty: float) -> list[Route]:
    """Up to *k* routes by iterated link-penalty. Fewer than *k* if the penalty
    stops producing new paths (returns each distinct path once)."""
    decode_graph = as_multidigraph(graph)
    view = simple_digraph_view(graph, weight)
    if src == dst:
        return [Route(nodes=[src], edges=[], cost=0.0)]
    work = view.copy()  # mutable weights; copy() is independent (verified)
    routes: list[Route] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(k):
        try:
            _, nodes = nx.single_source_dijkstra(work, src, target=dst, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        key = tuple(nodes)
        if key in seen:  # penalty too low to shift the route — no new alternative
            break
        seen.add(key)
        routes.append(_make_route(view, decode_graph, nodes, weight))
        for u, v in zip(nodes, nodes[1:], strict=False):  # penalise the used edges on the working copy
            work[u][v][weight] *= penalty
    return routes


def yen_routes(graph: nx.Graph, src: int, dst: int, weight: str, k: int) -> list[Route]:
    """Up to *k* exact shortest *simple* paths (NetworkX ``shortest_simple_paths``,
    i.e. Yen/Lawler), in increasing cost."""
    decode_graph = as_multidigraph(graph)
    view = simple_digraph_view(graph, weight)
    if src == dst:
        return [Route(nodes=[src], edges=[], cost=0.0)]
    try:
        gen = nx.shortest_simple_paths(view, src, dst, weight=weight)
        node_paths = list(itertools.islice(gen, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return [_make_route(view, decode_graph, nodes, weight) for nodes in node_paths]


def route_weights(routes: list[Route], scheme: WeightScheme, *, logit_beta: float = 0.1) -> list[float]:
    """Choice weights ``wᵢ`` for the routes, with ``Σ wᵢ = 1``.

    * ``uniform`` — ``1/k``.
    * ``inverse_cost`` — ``∝ 1/costᵢ`` (a zero-cost degenerate route gets all mass).
    * ``path_size_logit`` — ``∝ PSᵢ · exp(-β·costᵢ)`` where the path-size factor
      ``PSᵢ = Σ_{a∈i} (lₐ/Lᵢ)/N_a`` down-weights routes whose edges are shared by
      others (``N_a`` = number of routes using edge ``a``, ``lₐ`` = edge weight,
      ``Lᵢ`` = route cost). Distinct routes score near 1; a route fully overlapped
      by others is discounted.

    The weights are written back onto ``route.weight`` and also returned.
    """
    n = len(routes)
    if n == 0:
        return []
    if n == 1:
        routes[0].weight = 1.0
        return [1.0]

    if scheme == "uniform":
        w = [1.0 / n] * n
    elif scheme == "inverse_cost":
        inv = [(1.0 / r.cost if r.cost > 0 else math.inf) for r in routes]
        total = math.fsum(x for x in inv if math.isfinite(x)) or 0.0
        if any(math.isinf(x) for x in inv):  # a zero-cost route swallows the mass
            w = [1.0 if math.isinf(x) else 0.0 for x in inv]
            s = math.fsum(w)
            w = [x / s for x in w]
        else:
            w = [x / total for x in inv]
    else:  # path_size_logit
        w = _path_size_logit_weights(routes, logit_beta)

    for r, wi in zip(routes, w, strict=True):
        r.weight = wi
    return w


def _path_size_logit_weights(routes: list[Route], beta: float) -> list[float]:
    # N_a: how many routes traverse each (undirected) edge; lₐ approximated by 1
    # per edge share is by count — we use route cost as Lᵢ and unit edge length so
    # PSᵢ = (1/Lᵢ)·Σ_a 1/N_a is dominated by overlap, which is what we want.
    edge_use: dict[tuple[int, int, int], int] = {}
    edge_sets: list[set[tuple[int, int, int]]] = []
    for r in routes:
        es = {(min(u, v), max(u, v), key) for (u, v, key) in r.edges}
        edge_sets.append(es)
        for e in es:
            edge_use[e] = edge_use.get(e, 0) + 1
    utilities: list[float] = []
    for r, es in zip(routes, edge_sets, strict=True):
        n_edges = len(es) or 1
        ps = math.fsum(1.0 / edge_use[e] for e in es) / n_edges  # ∈ (0, 1]
        ps = ps if ps > 0 else 1e-9
        utilities.append(math.log(ps) - beta * r.cost)
    m = max(utilities)
    exps = [math.exp(u - m) for u in utilities]  # softmax, overflow-safe
    total = math.fsum(exps)
    return [e / total for e in exps]


def build_route_set(
    graph: nx.Graph,
    src: int,
    dst: int,
    *,
    weight: str,
    method: Method,
    k: int,
    penalty: float,
    weight_scheme: WeightScheme,
    logit_beta: float = 0.1,
) -> list[Route]:
    """Generate a route-set for one OD pair and attach the choice weights.

    Returns an empty list if the pair is unreachable. A single-route result (k=1
    or only one path exists) gets ``weight == 1.0``.
    """
    if method == "yen":
        routes = yen_routes(graph, src, dst, weight, k)
    else:
        routes = link_penalty_routes(graph, src, dst, weight, k, penalty)
    route_weights(routes, weight_scheme, logit_beta=logit_beta)
    return routes


__all__ = [
    "Method",
    "Route",
    "WeightScheme",
    "build_route_set",
    "link_penalty_routes",
    "route_weights",
    "simple_digraph_view",
    "yen_routes",
]
