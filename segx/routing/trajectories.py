"""The unified trajectory record consumed by the aggregator.

Every routing strategy emits :class:`Trajectory` objects; the aggregator does not
care which strategy produced one, only its ``(mode, category, binding)`` tags and
``weight``. ``path`` is the ordered list of graph edge ids the trajectory
traverses in the ``mode`` graph; it may be ``None`` for an unreachable
destination (``reason`` then says why), in which case the trajectory carries no
demographic mass onto any edge but is still recorded for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: Which routing strategy produced a trajectory — also the aggregator partition
#: dimension that keeps the dual POI setting (nearest vs catchment) separable.
Binding = Literal["nearest", "catchment", "od"]

#: ``transit`` (P7) is the merged walk ⊕ public-transport graph — see
#: :mod:`segx.network.transit`. A transit trajectory is an ordinary trajectory:
#: its ``path`` is still a list of ``(u, v, key)`` edges, some of which are
#: synthetic transit edges until :func:`segx.network.transit.expand_transit_rides`
#: rewrites them into the street/rail edges that actually receive mass.
Mode = Literal["walk", "drive", "transit"]

#: Why a trajectory has no path. ``None`` = reachable.
UnreachableReason = Literal["unreachable", "none_in_boundary", "over_max_distance", "no_demographics"]

#: Edge identifier in a mode graph: ``(u, v, key)`` for a MultiDiGraph edge.
EdgeId = tuple[int, int, int]


@dataclass(slots=True)
class Trajectory:
    origin_id: str
    destination_id: str
    mode: Mode
    category: str
    binding: Binding
    weight: float = 1.0
    path: list[EdgeId] | None = None
    cost: float | None = None  # travel time (min) or distance (m), per the engine
    reason: UnreachableReason | None = None
    #: Route-set provenance (P4.5). For a single-route OD pair (or any access
    #: trajectory) ``route_rank == 0`` and ``route_weight == 1.0``. When the ``od``
    #: binding emits *k* alternatives, they carry ``route_rank`` ``0..k-1`` and a
    #: route-choice weight ``route_weight`` (``wᵢ``) with ``Σ wᵢ = 1`` per pair;
    #: the aggregator deposits ``route_weight × demo_vector`` so a flow's mass is
    #: split across its routes, keeping weight conservation.
    route_rank: int = 0
    route_weight: float = 1.0
    #: Per-leg travel mode (P7.2), one entry per edge of ``path``. ``None`` — the
    #: normal case — means every leg travels in :attr:`mode`, which is true of any
    #: single-mode trajectory. A **public-transport** trip is not single-mode: it
    #: walks to the stop, rides, and walks again, and before P7.2 all three legs
    #: were filed under ``transit``, so "bus passenger" silently meant "bus
    #: passenger *plus* the same passenger on the sidewalk". Now
    #: :func:`segx.network.transit.expand_transit_rides` labels the ride legs
    #: ``transit`` and the walking legs ``walk``.
    #:
    #: This is **not** a fourth partition dimension (invariant #6): it is the
    #: ``mode`` dimension that already exists, resolved per edge rather than per
    #: trajectory — which is what ``mode`` always meant, "how is this person
    #: occupying this edge".
    leg_modes: list[str] | None = None
    #: free-form provenance (e.g. snap distances, which POI of k was chosen)
    meta: dict = field(default_factory=dict)

    @property
    def reachable(self) -> bool:
        return self.path is not None

    @property
    def partition(self) -> tuple[str, str, str]:
        """The aggregator key dimension this trajectory contributes to."""
        return (self.mode, self.category, self.binding)

    def leg_partitions(self) -> list[tuple[str, str, str]]:
        """One partition key per edge of :attr:`path`.

        Falls back to :attr:`partition` repeated when :attr:`leg_modes` is unset
        or the wrong length — a mismatched label list is a caller bug that must
        not silently misfile mass.
        """
        if self.path is None:
            return []
        if self.leg_modes is None or len(self.leg_modes) != len(self.path):
            return [self.partition] * len(self.path)
        return [(mode, self.category, self.binding) for mode in self.leg_modes]
