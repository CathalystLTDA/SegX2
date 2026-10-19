"""Routing-engine interface and the result record both engines return.

Implementations: :mod:`segx.routing.pandana_engine` (default, fast, contraction
hierarchies) and :mod:`segx.routing.networkx_engine` (validation oracle). The
strategy modules (``nearest_poi``, ``poi_catchment``, ``od_router``) are written
against this interface, not a concrete engine.

An engine holds one routable graph per mode and answers three primitives, each
with a batched form the strategies actually call (``nearest_many``,
``reachable_within_many``, ``shortest_paths``) — a batch must agree with the
per-item call exactly, it only shares the work. Each primitive is
in the units of a named edge ``weight`` (``"length"`` in metres, or
``"travel_time_min"``). All three return :class:`RouteResult` objects carrying
the decoded ``(u, v, key)`` edge path, so the aggregator never needs the engine
again. "Unreachable" is a ``None`` (nearest / shortest_path) or an absence from
the returned list (reachable_within) — never a fabricated zero; the *reason* is classified by the calling strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from segx.routing.inputs import RoutablePOI
    from segx.routing.trajectories import EdgeId


@dataclass(slots=True)
class RouteResult:
    """A reachable route from an origin to a target node.

    ``target_id`` is the POI id (bindings A/B) or destination id (binding C);
    ``cost`` is in the routing ``weight``'s units; ``path`` is the ordered list
    of traversed ``(u, v, key)`` edges (empty when origin == target).
    """

    target_id: str
    target_node: int
    cost: float
    path: list[EdgeId] = field(default_factory=list)


@runtime_checkable
class RoutingEngine(Protocol):
    """Per-mode routing primitives. See :class:`segx.routing.networkx_engine.NetworkXEngine`."""

    def nearest(self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str) -> RouteResult | None:
        """The single closest POI to *origin_node* by *weight*, or ``None`` if none reachable."""
        ...

    def reachable_within(
        self, origin_node: int, pois: list[RoutablePOI], mode: str, *, weight: str, budget: float
    ) -> list[RouteResult]:
        """All POIs whose *weight*-cost from *origin_node* is ``<= budget`` (possibly empty)."""

    def nearest_many(
        self, origin_nodes: Sequence[int], pois: list[RoutablePOI], mode: str, *, weight: str
    ) -> list[RouteResult | None]:
        """:meth:`nearest` for many origin nodes at once, one result per node in input order."""

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
        """:meth:`reachable_within` for many origin nodes, one list per node in input
        order. ``with_paths=False`` may leave every ``path`` empty (the caller only
        needs to know which POIs pass)."""

    def shortest_path(self, src_node: int, dst_node: int, mode: str, *, weight: str) -> RouteResult | None:
        """Route between two explicit nodes (binding C), or ``None`` if disconnected."""

    def shortest_paths(self, pairs: Sequence[tuple[int, int]], mode: str, *, weight: str) -> list[RouteResult | None]:
        """:meth:`shortest_path` for many ``(src, dst)`` pairs at once, one result
        per pair in input order. Must agree with the per-pair call exactly."""


__all__ = ["RouteResult", "RoutingEngine"]
