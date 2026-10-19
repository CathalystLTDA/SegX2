"""Graph-level inputs to the routing strategies (P3).

The strategies (``nearest_poi``, ``poi_catchment``, ``od_router``) and the
engines deliberately work on **plain node-level records**, not GeoDataFrames:
the orchestrator snaps origins / POIs / OD endpoints to their mode graph (via
:mod:`segx.network.snapping`) and hands routing these light dataclasses. That
keeps the whole routing layer free of geopandas/pandas — it only needs NetworkX
(and pandana) — so it imports and unit-tests against the synthetic city without
the heavy geospatial stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class RoutableOrigin:
    """An origin reduced to its snapped graph node and routing weight."""

    origin_id: str
    node: int
    weight: float = 1.0


@dataclass(slots=True, frozen=True)
class RoutablePOI:
    """A POI of some category reduced to its snapped graph node."""

    poi_id: str
    node: int


@dataclass(slots=True, frozen=True)
class RoutableODPair:
    """A user-supplied origin→destination pair, both endpoints snapped."""

    origin_id: str
    origin_node: int
    dest_id: str
    dest_node: int
    mode: str
    category: str
    weight: float = 1.0


def origin_blocks(origins: Iterable[RoutableOrigin], size: int) -> Iterator[list[RoutableOrigin]]:
    """Consecutive runs of at most *size* origins, in input order.

    The access strategies route a block per engine call: large enough to share
    the work, small enough that the routes held before they are yielded stay
    bounded.
    """
    block: list[RoutableOrigin] = []
    for origin in origins:
        block.append(origin)
        if len(block) >= size:
            yield block
            block = []
    if block:
        yield block


__all__ = ["RoutableODPair", "RoutableOrigin", "RoutablePOI", "origin_blocks"]
