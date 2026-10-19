"""Streaming measurement of **path shape**, partitioned by
``(mode, category, binding)``. P7.3.

The counterpart of :class:`~segx.aggregation.edge_accumulator.EdgeAccumulator`,
for the one measure that is not a property of a segment. Fragmentation — the
linear fractal dimension of a personal trajectory (Netto et al. 2015, via
Mandelbrot 1983) — is a property of the **whole path**, and the edge table has
discarded path identity by the time any ordinary metric sees it. So it is
measured here, as each trajectory streams past, and only bounded running sums
are kept.

Two deliberate differences from :class:`EdgeAccumulator`, both load-bearing:

* **State is keyed by partition × attribute × class, not by edge.** The bucket
  count is the number of demographic classes times the number of partitions —
  independent of both the trajectory count *and* the network size, so invariant
  #6's bounded-memory guarantee holds with room to spare.
* **The partition is the trajectory's, not the leg's.** P7.2 resolves ``mode``
  per *leg* in the edge accumulator, because edge occupancy genuinely varies
  along a transit trip — the rider is on the bus for one stretch and on the
  sidewalk for another. A fractal dimension cannot be split that way: it is a
  measurement of one continuous curve, and the curve a bus passenger traces
  *includes* the walk to the stop. Filing it under the trajectory's declared
  mode is what makes "how fragmented is a transit commute" answerable at all.

Every deposit is weighted by the class's demographic mass on the trajectory, so
the reported value is "the mean fractal dimension of the path travelled by a
member of this group" — the between-class comparison the source text is after.
The route-set choice weight ``route_weight`` is **not** applied here: it is
already folded into the vector by ``orchestrator._demo_vector_for``, exactly as
for the edge accumulator, and applying it twice would quietly square it.

The outcomes are counted separately, because they are different findings and
collapsing them would hide the last ones behind a small ``n`` (invariant #2):
``n_unresolved`` (no demographics), ``n_unreachable`` (no path), ``n_no_geometry``
(a path whose nodes carry no coordinates) and, among the paths with real
geometry, ``n_measured`` versus ``n_saturated`` (box counting: too few
resolutions with more than one pixel) or ``n_too_short`` (divider: too few
segments to fit a slope).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from segx.metrics.fragmentation import EdgeShapeCache, measure_path

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from segx.config import Settings
    from segx.metrics.fragmentation import EdgeXY
    from segx.routing.trajectories import Trajectory

    from .edge_accumulator import DemoVector, Partition

#: a node coordinate in the working CRS (metres)
Point = tuple[float, float]


class _Sums:
    """Weighted running sums for one ``(partition, attribute, class)`` bucket."""

    __slots__ = (
        "n_measured",
        "n_paths",
        "w_extent_m",
        "w_fragmentation",
        "w_fragmentation_sq",
        "w_length_m",
        "w_measured",
        "w_n_segments",
        "w_total",
    )

    def __init__(self) -> None:
        self.w_total = 0.0
        self.w_measured = 0.0
        self.w_fragmentation = 0.0
        self.w_fragmentation_sq = 0.0
        self.w_n_segments = 0.0
        self.w_length_m = 0.0
        self.w_extent_m = 0.0
        self.n_paths = 0
        self.n_measured = 0


class PathAccumulator:
    """Per-partition, per-class path-shape accumulator. P7.3.

    Use :meth:`add` once per trajectory — from the same loop that feeds
    :class:`EdgeAccumulator`, with the same ``demo_vector`` — then
    :meth:`to_frame` for the table the ``path_fragmentation`` metric consumes.
    """

    def __init__(self, settings: Settings, node_xy: dict[int, Point], edge_xy: EdgeXY | None = None):
        self._cfg = settings.metrics.fragmentation
        self._node_xy = node_xy
        self._edge_xy = edge_xy
        # Each edge is rasterised once and shared by every path that rides it —
        # bounded by the edges traversed, not by the trajectories streamed.
        self._shapes = EdgeShapeCache()
        self.n_saturated = 0
        self._attrs: list[tuple[str, list[str]]] = [
            (a.name, list(a.class_labels)) for a in settings.demographics.attributes
        ]
        self._acc: dict[Partition, dict[str, dict[str, _Sums]]] = {}
        self.n_unresolved = 0
        self.n_unreachable = 0
        self.n_no_geometry = 0
        self.n_measured = 0
        self.n_too_short = 0

    def _buckets(self, partition: Partition) -> dict[str, dict[str, _Sums]]:
        """The partition's buckets, materialised for **every** class on first sight.

        Pre-seeding matters: a class that travels nowhere under a partition must
        surface as a row with ``n_paths = 0`` and a ``NaN`` value, not vanish
        from the table. An absent row reads as "not applicable"; the truth is
        "this group made no measurable trip here", which is a finding.
        """
        buckets = self._acc.get(partition)
        if buckets is None:
            buckets = {name: {label: _Sums() for label in labels} for name, labels in self._attrs}
            self._acc[partition] = buckets
        return buckets

    def add(self, trajectory: Trajectory, demo_vector: DemoVector) -> None:
        """Measure one trajectory's path and fold it into the running sums."""
        if any(math.isnan(v) for classes in demo_vector.values() for v in classes.values()):
            self.n_unresolved += 1
            return
        if trajectory.path is None:
            self.n_unreachable += 1
            return

        shape = measure_path(trajectory.path, self._node_xy, self._cfg, self._edge_xy, self._shapes)
        if shape is None:
            self.n_no_geometry += 1
            return

        measured = shape.measured
        if measured:
            self.n_measured += 1
        elif shape.reason == "saturated":
            self.n_saturated += 1
        else:
            self.n_too_short += 1

        buckets = self._buckets(trajectory.partition)
        for name, labels in self._attrs:
            vec = demo_vector[name]
            per_class = buckets[name]
            for label in labels:
                w = vec[label]
                if w <= 0.0:
                    # No mass for this class on this trajectory — not a trip this
                    # group made, so it must not drag the group's mean towards
                    # the shape of a path nobody in it travelled.
                    continue
                s = per_class[label]
                s.w_total += w
                s.n_paths += 1
                s.w_n_segments += w * shape.n_segments
                s.w_length_m += w * shape.length_m
                s.w_extent_m += w * shape.extent_m
                if measured:
                    s.w_measured += w
                    s.n_measured += 1
                    s.w_fragmentation += w * shape.fragmentation
                    s.w_fragmentation_sq += w * shape.fragmentation * shape.fragmentation

    def partitions(self) -> list[Partition]:
        return list(self._acc)

    def to_frame(self) -> pd.DataFrame:
        """Tidy table: one row per ``(partition, attribute, class)``.

        Columns: ``mode, category, binding, attribute, class``, the weighted sums
        (``w_total``, ``w_measured``, ``w_fragmentation``, ``w_fragmentation_sq``,
        ``w_n_segments``, ``w_length_m``, ``w_extent_m``) and the counts
        (``n_paths``, ``n_measured``). Sums rather than means, so the division —
        and the choice of what to divide by — stays in the metric plugin
        (invariant #5).
        """
        import pandas as pd

        records: list[dict] = []
        for (mode, category, binding), attrs in self._acc.items():
            for name, per_class in attrs.items():
                for label, s in per_class.items():
                    records.append(
                        {
                            "mode": mode,
                            "category": category,
                            "binding": binding,
                            "attribute": name,
                            "class": label,
                            "w_total": s.w_total,
                            "w_measured": s.w_measured,
                            "w_fragmentation": s.w_fragmentation,
                            "w_fragmentation_sq": s.w_fragmentation_sq,
                            "w_n_segments": s.w_n_segments,
                            "w_length_m": s.w_length_m,
                            "w_extent_m": s.w_extent_m,
                            "n_paths": s.n_paths,
                            "n_measured": s.n_measured,
                        }
                    )
        return pd.DataFrame.from_records(records, columns=list(_FRAME_COLUMNS))

    def stats(self) -> dict:
        """Audit counts + the ruler-ladder parameters, for the run's metadata.

        The parameters ride along for the same reason ``transit_source =
        "osm_assumed"`` and the P7.2 corridor tolerances do (invariant #7): a
        fractal dimension measured under one ladder and another measured under a
        different one are indistinguishable once they are two numbers in a table.
        """
        return {
            "n_measured": self.n_measured,
            "n_too_short": self.n_too_short,
            "n_saturated": self.n_saturated,
            "n_no_geometry": self.n_no_geometry,
            "n_unreachable": self.n_unreachable,
            "n_unresolved": self.n_unresolved,
            "params": self._cfg.model_dump(),
        }


_FRAME_COLUMNS: tuple[str, ...] = (
    "mode",
    "category",
    "binding",
    "attribute",
    "class",
    "w_total",
    "w_measured",
    "w_fragmentation",
    "w_fragmentation_sq",
    "w_n_segments",
    "w_length_m",
    "w_extent_m",
    "n_paths",
    "n_measured",
)


__all__ = ["PathAccumulator", "Point"]
