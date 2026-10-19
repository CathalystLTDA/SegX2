"""Segregated Trajectories — the street segments most exclusively used by a single
social group ("Most exclusive trajectories"). P5.

The unit of analysis is the **street segment**, not the
whole trajectory (so no per-trajectory retention is needed). For each segment we
compute, per class ``g``, a location quotient

    ``LQ_g = p_g(segment) / p_g(partition)``

— the segment's share of group ``g`` relative to that group's share across the
whole partition. ``LQ_g > 1`` means ``g`` is *over-represented* on the segment. A
segment is flagged ``exclusive`` when its **dominant** group's ``LQ`` reaches
``settings.metrics.exclusive_lq_threshold``. The number of exclusive segments a
group gets naturally scales with the trajectory-sample size and the city's degree
of segregation.

Convention: a segment/group with no mass (or a group absent from the whole
partition, ``0/0``) yields ``LQ = NaN`` (invariant #2). The overall indicator is
the count and share of exclusive segments per dominant group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from segx.metrics._segments import segment_groups
from segx.metrics.base import Metric
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

__all__ = ["SegregatedTrajectoriesMetric"]

_PARTITION_KEYS: tuple[str, ...] = ("mode", "category", "binding", "attribute")
_SEGMENT_KEYS: tuple[str, ...] = ("mode", "category", "binding", "u", "v", "key", "attribute")


@register_metric
class SegregatedTrajectoriesMetric(Metric):
    """Per-segment dominant-group location quotient + exclusivity flag. P5."""

    name = "segregated_trajectories"
    value_column = "lq"

    def compute(self, edge_table, settings: Settings):
        import pandas as pd

        columns = [*_SEGMENT_KEYS, "dominant_class", "lq", "exclusive"]
        if edge_table.empty:
            return pd.DataFrame(columns=columns)

        threshold = settings.metrics.exclusive_lq_threshold
        df = edge_table
        part = list(_PARTITION_KEYS)
        # global group share within the partition: total_g / total_all
        total_g = df.groupby([*part, "class"], sort=False)["mass"].transform("sum")
        total_all = df.groupby(part, sort=False)["mass"].transform("sum")
        global_p = total_g / total_all  # NaN/inf where the partition has no mass
        # segment group share: mass_g / segment_total
        seg_total = df.groupby(list(_SEGMENT_KEYS), sort=False)["mass"].transform("sum")
        seg_p = df["mass"] / seg_total
        df = df.assign(_lq=(seg_p / global_p))

        groups = segment_groups(df, _SEGMENT_KEYS)
        dominants: list[str | None] = []
        lqs: list[float] = []
        exclusives: list[bool] = []
        for masses, classes, seg_lq in groups.arrays("mass", "class", "_lq"):
            total = math.fsum(masses)
            if total <= 0.0:
                dominant, lq, exclusive = None, math.nan, False
            else:
                winner = int(masses.argmax())
                dominant = str(classes[winner])
                lq = float(seg_lq[winner])
                exclusive = bool(lq >= threshold) if not math.isnan(lq) else False
            dominants.append(dominant)
            lqs.append(lq)
            exclusives.append(exclusive)
        out = groups.keys
        out["dominant_class"] = pd.Series(dominants, dtype=object)
        out["lq"] = pd.Series(lqs, dtype=float)
        out["exclusive"] = pd.Series(exclusives, dtype=bool)
        return out[columns]

    def summarize(self, per_segment, settings: Settings):  # noqa: ARG002
        """Count and share of exclusive segments per dominant group."""
        import pandas as pd

        keys = ["mode", "category", "binding", "attribute", "dominant_class"]
        out_cols = [*keys, "n_segments", "n_exclusive", "share_exclusive"]
        if per_segment.empty:
            return pd.DataFrame(columns=out_cols)
        grouped = per_segment.groupby(keys, sort=False)
        out = grouped.agg(n_segments=("exclusive", "size"), n_exclusive=("exclusive", "sum")).reset_index()
        out["share_exclusive"] = out["n_exclusive"] / out["n_segments"]
        return out
