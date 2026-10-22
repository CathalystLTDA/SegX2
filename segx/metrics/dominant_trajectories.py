"""Dominant Trajectories — the dominant social group per street segment, weighted
by how sharply it dominates ("Dynamic segregation"). P5.

For each segment the group with the highest mass colours it (``dominant_class``);
the ``dominance`` weight is ``1 − normalized_entropy`` (whole-distribution
homogeneity). A near-tie (e.g. 33/33/34) yields a
dominance near ``0``; a one-class segment (0/0/100) yields ``1``. The overall
indicator is the mean dominance across segments (inherited :meth:`Metric.summarize`).

Convention: a segment with no demographic mass has ``dominant_class = None`` and
``dominance = NaN`` (invariant #2), never a spurious winner.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from segx.metrics._segments import segment_groups
from segx.metrics.base import Metric
from segx.metrics.entropy import normalized_entropy
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

__all__ = ["DominantTrajectoriesMetric"]

#: one segment under a partition and attribute (matches ``EdgeAccumulator.to_frame``)
_SEGMENT_KEYS: tuple[str, ...] = ("mode", "category", "binding", "u", "v", "key", "attribute")


@register_metric
class DominantTrajectoriesMetric(Metric):
    """Per-segment dominant class + homogeneity-weighted dominance. P5."""

    name = "dominant_trajectories"
    description = (
        "Which class has the strongest presence on each segment (its colour on the map), and how "
        "sharply it dominates the rest: ≈ 0 for a near tie, 1 for a single class."
    )
    formula = "dominant(s) = argmaxᵢ Pᵢ(s)\ndominance(s) = 1 − h(s)"
    #: the network-wide indicator averages the dominance weight
    value_column = "dominance"

    def compute(self, edge_table, settings: Settings):  # noqa: ARG002
        import pandas as pd

        columns = [*_SEGMENT_KEYS, "dominant_class", self.value_column]
        if edge_table.empty:
            return pd.DataFrame(columns=columns)

        groups = segment_groups(edge_table, _SEGMENT_KEYS)
        dominants: list[str | None] = []
        dominances: list[float] = []
        for masses, classes in groups.arrays("mass", "class"):
            total = math.fsum(masses)
            if total <= 0.0:
                dominant, dominance = None, math.nan
            else:
                winner = int(masses.argmax())
                dominant = str(classes[winner])
                dominance = 1.0 - normalized_entropy(masses, n_classes=len(masses))
            dominants.append(dominant)
            dominances.append(dominance)
        out = groups.keys
        out["dominant_class"] = pd.Series(dominants, dtype=object)
        out[self.value_column] = pd.Series(dominances, dtype=float)
        return out[columns]
