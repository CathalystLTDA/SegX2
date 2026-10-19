"""Streaming per-edge accumulation of demographic mass, partitioned by
``(mode, category, binding)``. P4.

As each trajectory is streamed in, every edge on its decoded path receives the
trajectory's per-class agent counts (its ``demo_vector``), added into the bucket
for that edge under the trajectory's ``(mode, category, binding)`` partition.
This is the operation behind Netto et al. (2018): *"we counted … trajectories
classified by income level for each street segment … accumulated for her/his
income group."*

Design choices:

* **Deposit = the ``demo_vector`` itself** (absolute per-class counts), *not*
  ``trajectory.weight × vector``. ``origin.weight`` is a reserved routing hook
  (currently ``1.0`` everywhere); the demographic counts already live in the
  vector, and the caller (the orchestrator) supplies the vector appropriate to
  the trajectory — the **origin's** vector for access bindings (the household
  reaches amenities from home) and the **per-flow** vector for OD commutes (so a
  home block's workers are split across its destinations, not multiplied by
  them). Whether a future weighting policy multiplies this is an open item.
* **Streaming, bounded state.** The accumulator holds one bucket per
  ``(partition, edge, attribute)`` — bounded by edges × partitions, never by the
  number of trajectories. No per-trajectory storage in the hot path.
* **The partition is resolved per *leg*, not per trajectory** (P7.2). A
  public-transport trip walks to the stop, rides, and walks again; its ride legs
  belong in ``mode="transit"`` and its walking legs in ``mode="walk"``, and
  :meth:`Trajectory.leg_partitions` says which is which. This is not a new
  dimension — it is the ``mode`` dimension resolved where it actually varies. The
  conservation ledger is unaffected: it is still booked **once per trajectory**.
* **Unresolved demographics are flagged, never zeroed**: a
  trajectory whose origin has a ``NaN`` vector deposits nothing and is counted in
  ``n_unresolved`` — it is *not* folded into the conservation ledger as zero.

The weight-conservation ledger (:class:`~segx.validation.invariants.WeightLedger`)
is kept over **resolved** trajectories at the *trajectory* level: each resolved
trajectory's mass is booked once as either ``deposited`` (it had a path) or
``unreachable`` (it did not), so ``emitted == deposited + unreachable``. The
separate ``edge_mass_total`` (mass × path length, summed) lets a test confirm the
per-edge sums didn't lose or duplicate deposits.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from segx.validation.invariants import WeightLedger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from segx.config import Settings
    from segx.network.street_space import CorridorInfo
    from segx.routing.trajectories import EdgeId, Trajectory

#: ``(mode, category, binding)``
Partition = tuple[str, str, str]

#: per-class demographic vector: ``{attribute_name: {class_label: mass}}``
DemoVector = dict[str, dict[str, float]]


def _is_unresolved(demo_vector: DemoVector) -> bool:
    """True if any class mass is ``NaN`` (the flag for an unresolved origin)."""
    return any(math.isnan(v) for classes in demo_vector.values() for v in classes.values())


def canonical_edge(edge: EdgeId) -> EdgeId:
    """Undirected segment key ``(min(u, v), max(u, v), key)``.

    Normalises an edge's endpoint order so a segment can't split by which way a
    trajectory traversed it. Cheap and applied unconditionally
    — a no-op when the graph is already undirected (endpoints already ordered by
    :func:`segx.network.builder.to_undirected_graph`), and a safety net when a
    directed graph is used so both directions still fold into one map line.
    """
    u, v, key = edge
    return (u, v, key) if u <= v else (v, u, key)


class EdgeAccumulator:
    """Per-edge, per-partition demographic-mass accumulator. P4.

    Use :meth:`add` once per trajectory (streaming), then :meth:`to_frame` for the
    tidy per-edge table and :meth:`ledger` for the conservation bookkeeping.
    """

    def __init__(
        self,
        settings: Settings,
        edge_layers: dict[EdgeId, str] | None = None,
        edge_corridors: dict[EdgeId, CorridorInfo] | None = None,
    ):
        # P7: ``edge -> segx_layer`` for the non-street minority (rail rides). The
        # layer is a property of the **edge**, not of the trajectory, so it must not
        # enter the ``(mode, category, binding)`` partition key — a fourth dimension
        # there would multiply the accumulator's state and break the bounded-memory
        # guarantee of invariant #6. It rides along as a passthrough column instead,
        # which is why no metric needs changing. Keys are canonicalised so a map
        # built from a directed graph still matches the accumulator's segment keys.
        self._layers: dict[EdgeId, str] = (
            {canonical_edge(edge): layer for edge, layer in edge_layers.items()} if edge_layers else {}
        )
        # P7.2: ``edge -> CorridorInfo`` — the canonical street space the edge sits
        # in, from :mod:`segx.network.street_space`. Exactly the same contract as
        # ``_layers``: a property of the edge, a passthrough column, and *never* a
        # partition key. It is what lets ``cross_mode_exposure`` pool a car on the
        # roadway with a pedestrian on the separately-mapped sidewalk beside it.
        self._corridors: dict[EdgeId, CorridorInfo] = (
            {canonical_edge(edge): info for edge, info in edge_corridors.items()} if edge_corridors else {}
        )
        # attribute -> ordered class labels, fixed from the schema
        self._attrs: list[tuple[str, list[str]]] = [
            (a.name, list(a.class_labels)) for a in settings.demographics.attributes
        ]
        # Every class of every attribute, flattened in schema order: one running
        # sum per slot. The hot loop adds a trajectory's values slot by slot
        # instead of walking attribute and class names once per edge.
        self._slots: list[tuple[str, str]] = [(name, label) for name, labels in self._attrs for label in labels]
        # partition -> edge -> per-slot running sums
        self._acc: dict[Partition, dict[EdgeId, list[float]]] = {}
        # partition -> edge -> number of trajectories through it (line-thickness
        # signal). Counted independent of demographic mass.
        self._paths: dict[Partition, dict[EdgeId, int]] = {}
        self._emitted = 0.0
        self._deposited = 0.0
        self._unreachable = 0.0
        self._edge_mass_total = 0.0
        self.n_unresolved = 0
        self.n_reachable = 0
        self.n_unreachable = 0

    def _representative_mass(self, demo_vector: DemoVector) -> float:
        """Total agents on the trajectory = sum of the first attribute's classes.

        Every attribute is a different partitioning of the *same* people, so all
        attributes share this total; the first is a representative choice.
        """
        name, labels = self._attrs[0]
        return math.fsum(demo_vector[name][label] for label in labels)

    def add(self, trajectory: Trajectory, demo_vector: DemoVector) -> None:
        """Stream one trajectory's mass onto the edges of its path."""
        path = trajectory.path
        # One partition for the whole path unless the trajectory labels its legs
        # (a transit trip: ride legs `transit`, walking legs `walk` — P7.2).
        leg_modes = trajectory.leg_modes
        mixed = path is not None and leg_modes is not None and len(leg_modes) == len(path)
        partitions = trajectory.leg_partitions() if mixed else None
        partition = trajectory.partition

        # Traffic count first: every trajectory that traverses edges contributes to
        # the per-segment path count (line-thickness signal), *independent of
        # demographic mass* — an unresolved-demographic trajectory is still traffic.
        if path:
            canonical = [edge if edge[0] <= edge[1] else (edge[1], edge[0], edge[2]) for edge in path]
            if partitions is None:
                paths = self._paths.setdefault(partition, {})
                for ce in canonical:
                    paths[ce] = paths.get(ce, 0) + 1
            else:
                for ce, leg_partition in zip(canonical, partitions, strict=True):
                    paths = self._paths.setdefault(leg_partition, {})
                    paths[ce] = paths.get(ce, 0) + 1

        if _is_unresolved(demo_vector):
            self.n_unresolved += 1
            return

        mass = self._representative_mass(demo_vector)
        self._emitted += mass

        if path is None:  # unreachable — mass booked, deposited nowhere
            self._unreachable += mass
            self.n_unreachable += 1
            return

        self._deposited += mass
        self.n_reachable += 1
        if not path:
            return

        values = [demo_vector[name][label] for name, label in self._slots]
        slots = range(len(values))
        # Added once per edge, in path order — the running total a test checks the
        # per-edge sums against, so it must see the same additions they do.
        edge_mass_total = self._edge_mass_total
        edges = self._acc.setdefault(partition, {}) if partitions is None else None
        for index, ce in enumerate(canonical):
            if partitions is not None:
                edges = self._acc.setdefault(partitions[index], {})
            bucket = edges.get(ce)
            if bucket is None:
                # `0.0 + x`, not `x`: exactly what adding into a zeroed bucket gives.
                edges[ce] = [0.0 + x for x in values]
            else:
                for i in slots:
                    bucket[i] += values[i]
            edge_mass_total += mass
        self._edge_mass_total = edge_mass_total

    def ledger(self) -> WeightLedger:
        """Trajectory-level conservation ledger over resolved trajectories."""
        return WeightLedger(emitted=self._emitted, deposited=self._deposited, unreachable=self._unreachable)

    @property
    def edge_mass_total(self) -> float:
        """Σ over deposits of representative mass = Σ_trajectory (mass × path length)."""
        return self._edge_mass_total

    def partitions(self) -> list[Partition]:
        return list(self._acc)

    def to_frame(self) -> pd.DataFrame:
        """Tidy long table: one row per ``(partition, edge, attribute, class)``.

        Columns: ``mode, category, binding, u, v, key, layer, corridor, length_m,
        attribute, class, mass, n_paths``. ``n_paths`` (the number of trajectories
        through the segment) is constant across the class rows of a segment — a
        per-segment line-thickness signal carried alongside the demographic mass.
        ``layer`` (P7) is ``"street"`` unless the segment is a transit
        ``rail_line`` edge; the writers split GeoPackage layers on it and the
        metrics ignore it. ``corridor`` and ``length_m`` (P7.2) name the canonical
        street space the segment sits in and how long it is — the pair
        ``cross_mode_exposure`` pools over, so a car and a pedestrian on the same
        street meet even when they are on different OSM ways. Both are passthrough
        properties of the *edge*, never partition keys. Consumed by the metric
        plugins (:mod:`segx.metrics`).
        """
        import pandas as pd

        records: list[dict] = []
        for partition, edges in self._acc.items():
            mode, category, binding = partition
            path_counts = self._paths.get(partition, {})
            for edge, sums in edges.items():
                u, v, key = edge
                n_paths = path_counts.get(edge, 0)
                layer = self._layers.get(edge, "street")
                corridor, length_m = self._corridor_of(edge)
                for (name, label), mass in zip(self._slots, sums, strict=True):
                    records.append(
                        {
                            "mode": mode,
                            "category": category,
                            "binding": binding,
                            "u": u,
                            "v": v,
                            "key": key,
                            "layer": layer,
                            "corridor": corridor,
                            "length_m": length_m,
                            "attribute": name,
                            "class": label,
                            "mass": mass,
                            "n_paths": n_paths,
                        }
                    )
        columns = [
            "mode",
            "category",
            "binding",
            "u",
            "v",
            "key",
            "layer",
            "corridor",
            "length_m",
            "attribute",
            "class",
            "mass",
            "n_paths",
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    def _corridor_of(self, edge: EdgeId) -> tuple[str, float]:
        """``(corridor_id, length_m)`` for a segment, defaulting to its own space.

        A run with no resolved street space (single mode, or the feature switched
        off) gives every segment a corridor of its own, which makes the cross-mode
        pool collapse back to the exact ``(u, v, key)`` join it used before P7.2.
        """
        info = self._corridors.get(edge)
        if info is not None:
            return info.corridor_id, info.length_m
        u, v, key = edge
        return f"own:{u}:{v}:{key}", 0.0

    def path_counts(self) -> pd.DataFrame:
        """Per-segment trajectory counts: one row per ``(partition, edge)``.

        Columns: ``mode, category, binding, u, v, key, layer, corridor, length_m,
        n_paths``. This is the raw line-thickness signal — more
        trajectories through a segment → a thicker line on the map. Counted
        independent of demographic mass, so a segment can appear here even if all
        its traffic was unresolved.
        """
        import pandas as pd

        records: list[dict] = []
        for (mode, category, binding), edges in self._paths.items():
            for edge, n_paths in edges.items():
                u, v, key = edge
                corridor, length_m = self._corridor_of(edge)
                records.append(
                    {
                        "mode": mode,
                        "category": category,
                        "binding": binding,
                        "u": u,
                        "v": v,
                        "key": key,
                        "layer": self._layers.get(edge, "street"),
                        "corridor": corridor,
                        "length_m": length_m,
                        "n_paths": n_paths,
                    }
                )
        columns = [
            "mode",
            "category",
            "binding",
            "u",
            "v",
            "key",
            "layer",
            "corridor",
            "length_m",
            "n_paths",
        ]
        return pd.DataFrame.from_records(records, columns=columns)


__all__ = ["DemoVector", "EdgeAccumulator", "Partition", "canonical_edge"]
