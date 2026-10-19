"""Strategy C — route the user-supplied origin→destination pairs in the per-pair
``mode`` (default ``'car'``→``drive`` for ``commute``, ``'walk'`` otherwise — see
:class:`segx.config.ODColumnMapping`). Emits :class:`~segx.routing.trajectories.Trajectory`
objects with ``binding='od'`` and ``category`` taken from the pair. An unreachable
destination yields a ``path=None`` trajectory with ``reason='unreachable'`` (never
silently dropped). Demographic mass is attached later by the
aggregator. P3; route-set (P4.5) added.

**Single route (default, ``routing.route_set.k == 1``):** one trajectory per pair
via the fast engine (Pandana) shortest path. Pairs are routed **in batches, one
per mode** (``engine.shortest_paths``) and emitted in their input order.

**Route-set (``k > 1``):** the pair emits *k* alternative routes
(:mod:`segx.routing.route_set`, link-penalty or exact Yen), each tagged with its
``route_rank`` and choice weight ``route_weight`` (``Σ = 1`` per pair). Requires the
mode graph (passed as ``graphs``); route-set search runs on NetworkX (mutable
weights / simple-path enumeration), so it is opt-in and used for the commute binding.

Routes on ``"travel_time_min"`` — commute cost is naturally a travel time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from segx.routing.route_set import build_route_set
from segx.routing.trajectories import Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Iterator, Mapping

    import networkx as nx

    from segx.config import Settings
    from segx.routing.engine import RoutingEngine
    from segx.routing.inputs import RoutableODPair

#: Edge weight OD/commute trajectories are routed on (minutes).
OD_WEIGHT = "travel_time_min"


def od_pair_trajectories(
    od_pairs: Iterable[RoutableODPair],
    engine: RoutingEngine,
    settings: Settings,
    *,
    graphs: Mapping[str, nx.Graph] | None = None,
) -> Iterator[Trajectory]:
    """Yield ``Trajectory`` objects (``binding='od'``) for each OD pair — one per
    pair by default, or *k* per pair when a route-set is enabled and ``graphs`` is
    provided."""
    route_set = settings.routing.route_set
    use_route_set = route_set.k > 1 and graphs is not None

    if not use_route_set:
        yield from _single_routes(list(od_pairs), engine)
        return

    for pair in od_pairs:
        graph = graphs[pair.mode]
        routes = build_route_set(
            graph,
            pair.origin_node,
            pair.dest_node,
            weight=OD_WEIGHT,
            method=route_set.method,
            k=route_set.k,
            penalty=route_set.penalty,
            weight_scheme=route_set.weight_scheme,
            logit_beta=route_set.logit_beta,
        )
        if not routes:
            yield Trajectory(
                pair.origin_id, pair.dest_id, pair.mode, pair.category, "od", weight=pair.weight, reason="unreachable"
            )
            continue
        for rank, route in enumerate(routes):
            yield Trajectory(
                pair.origin_id,
                pair.dest_id,
                pair.mode,
                pair.category,
                "od",
                weight=pair.weight,
                path=route.edges,
                cost=route.cost,
                route_rank=rank,
                route_weight=route.weight,
            )


def _single_routes(pairs: list[RoutableODPair], engine: RoutingEngine) -> Iterator[Trajectory]:
    """The default single shortest path for every pair (fast engine), in input order.

    Routed one batch per mode — a batch is a query against one mode graph — then
    re-emitted in the order the pairs came in, so the trajectory stream is the
    same as routing them one by one.
    """
    by_mode: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        by_mode.setdefault(pair.mode, []).append(index)
    results: list = [None] * len(pairs)
    for mode, indexes in by_mode.items():
        nodes = [(pairs[i].origin_node, pairs[i].dest_node) for i in indexes]
        for i, result in zip(indexes, engine.shortest_paths(nodes, mode, weight=OD_WEIGHT), strict=True):
            results[i] = result

    for pair, result in zip(pairs, results, strict=True):
        if result is None:
            yield Trajectory(
                pair.origin_id, pair.dest_id, pair.mode, pair.category, "od", weight=pair.weight, reason="unreachable"
            )
            continue
        yield Trajectory(
            pair.origin_id,
            pair.dest_id,
            pair.mode,
            pair.category,
            "od",
            weight=pair.weight,
            path=result.path,
            cost=result.cost,
        )


__all__ = ["OD_WEIGHT", "od_pair_trajectories"]
