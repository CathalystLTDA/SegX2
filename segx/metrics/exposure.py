"""Exposure — the potential group diversity each social group experiences across
the street network ("Social exposure"). P5.

For each segment the composition entropy ``H`` (Shannon, proportions/bits — how
mixed the people on that segment are) is computed once. Then, per group ``g``, we
average ``H`` over the segments ``g`` actually traverses (``mass_g > 0``), weighted
by ``g``'s presence ``mass_g`` on each. The result: one exposure value per group,
per ``(mode, category, binding, attribute)`` partition — how much diversity a group
is potentially exposed to on its daily trajectories.

Convention: a group that traverses no segment (or only single-class ones with
undefined ``H``) gets ``NaN`` (invariant #2). The overall indicator (inherited
:meth:`Metric.summarize`) is the mean exposure across groups per partition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from segx.metrics._segments import per_segment_values
from segx.metrics.base import Metric
from segx.metrics.entropy import shannon_entropy
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

__all__ = ["ExposureMetric"]

_PARTITION_KEYS: tuple[str, ...] = ("mode", "category", "binding", "attribute")
_SEGMENT_KEYS: tuple[str, ...] = (*_PARTITION_KEYS, "u", "v", "key")


@register_metric
class ExposureMetric(Metric):
    """Per-group mean segment diversity over the segments the group traverses. P5."""

    name = "exposure"

    def compute(self, edge_table, settings: Settings):  # noqa: ARG002
        import numpy as np
        import pandas as pd

        columns = [*_PARTITION_KEYS, "class", self.name]
        if edge_table.empty:
            return pd.DataFrame(columns=columns)

        # per-segment composition entropy H (bits), one value per segment
        seg_h = per_segment_values(edge_table, _SEGMENT_KEYS, "mass", shannon_entropy, "_H")
        merged = edge_table.merge(seg_h, on=list(_SEGMENT_KEYS), how="left")

        records: list[dict] = []
        for keys, sub in merged.groupby([*_PARTITION_KEYS, "class"], sort=False):
            mass = sub["mass"].to_numpy(dtype=float)
            h = sub["_H"].to_numpy(dtype=float)
            mask = (mass > 0.0) & ~np.isnan(h)
            if not mask.any():
                value = float("nan")
            else:
                w = mass[mask]
                value = float((w * h[mask]).sum() / w.sum())
            records.append({**dict(zip([*_PARTITION_KEYS, "class"], keys, strict=True)), self.name: value})
        return pd.DataFrame.from_records(records, columns=columns)
