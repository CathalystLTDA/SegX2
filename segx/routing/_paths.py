"""Node-sequence → ``EdgeId`` decoding, shared by both engines (P3).

Routing always operates on a **MultiDiGraph view** of the mode graph (see
:func:`as_multidigraph`), so there is exactly one edge-identity convention —
the one production uses. An :data:`~segx.routing.trajectories.EdgeId` is the
``(u, v, key)`` triple of a traversed directed edge; for a node hop with several
parallel edges, the **min-weight** one is chosen. The undirected synthetic city
is converted to a MultiDiGraph at engine setup (both directions, equal weights),
which leaves every shortest-path length — and therefore the hand-computed ground
truth — unchanged, while giving the aggregator stable directed edge keys.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover
    from segx.routing.trajectories import EdgeId


def as_multidigraph(graph: nx.Graph) -> nx.MultiDiGraph:
    """Return *graph* as a ``MultiDiGraph`` (the routing-internal representation).

    An OSMnx graph already is one and is returned unchanged. An undirected /
    simple graph (e.g. the synthetic city) is converted, which materialises both
    directions of every edge with the same attributes — shortest-path lengths are
    preserved. Node and edge attributes (``x``/``y``, ``length``,
    ``travel_time_min``) carry over.
    """
    if isinstance(graph, nx.MultiDiGraph):
        return graph
    return nx.MultiDiGraph(graph)


def _min_weight_key(graph: nx.MultiDiGraph, u: int, v: int, weight: str) -> int:
    """The parallel-edge key from *u* to *v* with the smallest *weight*."""
    parallel = graph.get_edge_data(u, v)
    if not parallel:  # pragma: no cover - guarded by caller (edge came from a path)
        raise KeyError(f"no edge {u!r}->{v!r} in graph")
    best_key, best_w = None, math.inf
    for key, data in parallel.items():
        w = data.get(weight)
        if w is None:
            w = data.get("length", 1.0)
        if w < best_w:
            best_w, best_key = float(w), key
    return int(best_key)  # type: ignore[arg-type]


def node_path_to_edges(graph: nx.MultiDiGraph, nodes: list[int], weight: str) -> list[EdgeId]:
    """Turn an ordered node path into the list of traversed ``(u, v, key)`` edges.

    A zero- or one-node path (origin == destination) yields an empty edge list —
    a real, reachable trajectory that simply traverses no edge.
    """
    return [(int(u), int(v), _min_weight_key(graph, u, v, weight)) for u, v in zip(nodes, nodes[1:], strict=False)]


__all__ = ["as_multidigraph", "node_path_to_edges"]
