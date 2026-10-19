"""Shannon entropy of demographic diversity per street segment.

The core function :func:`shannon_entropy` is the engine's first real piece of
math and is unit-tested against hand-computed values on the synthetic city
(:mod:`segx.validation.synthetic_city`). The :class:`ShannonEntropyMetric`
wrapper applies the diversity measure to the aggregated per-edge table (P4).

**Two diversity measures live here — they differ, and the choice is the
contract's, not ours:**

* :func:`shannon_entropy` — the normalized ``H = -Σ pᵢ log₂ pᵢ`` (proportions;
  in bits). Measures *mixing* independent of how many agents pass.
* :func:`segment_diversity` — the formula exactly as written in Netto et al.
  (2018), *A temporal geography of encounters*, fig. 5:
  ``D = -Σ Pᵢ log₂(Pᵢ/Pₜ)`` with **absolute counts** ``Pᵢ`` in front of the log.
  Algebraically ``D = Pₜ · H`` — i.e. entropy *weighted by the segment's agent
  volume* ``Pₜ``. Busy, mixed streets score highest.

``ShannonEntropyMetric`` uses :func:`segment_diversity` (the paper's formula); the
normalized ``H`` is the alternative. Swapping the measure is a one-line change
here — nothing in aggregation or routing depends on which is used.

Convention (both measures): an edge with **zero** demographic mass under a
partition has *undefined* diversity → ``NaN`` (never ``0.0``). ``0.0`` is
reserved for the real case "all mass in a single class".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from segx.metrics._segments import per_segment_values
from segx.metrics.base import Metric
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

__all__ = [
    "ExperiencedSegregationMetric",
    "IntergroupContactMetric",
    "ShannonEntropyMetric",
    "gini_simpson",
    "normalized_entropy",
    "segment_diversity",
    "shannon_entropy",
]


def shannon_entropy(counts: Iterable[float], base: float = 2.0) -> float:
    """Shannon entropy ``H = -Σ pᵢ log_base pᵢ`` of a discrete distribution.

    ``counts`` are non-negative masses per class (need not sum to 1). Returns
    ``nan`` if the total mass is zero (entropy undefined), ``0.0`` if all mass is
    in one class. ``base`` defaults to 2 → entropy in bits.
    """
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    log_b = math.log(base)
    h = 0.0
    for c in counts:
        if c > 0.0:
            p = c / total
            h -= p * math.log(p) / log_b
    # clamp tiny negative dust from floating-point
    return 0.0 if h < 0.0 else h


def normalized_entropy(counts: Iterable[float], n_classes: int, base: float = 2.0) -> float:
    """Pielou evenness: ``shannon_entropy / log_base(n_classes)`` ∈ [0, 1].

    ``n_classes`` is the number of *possible* classes for the attribute (from the
    demographic schema), not the number observed. Returns ``nan`` if total mass
    is zero; ``0.0`` if ``n_classes < 2`` (no diversity is possible).
    """
    if n_classes < 2:
        return 0.0
    h = shannon_entropy(counts, base=base)
    if math.isnan(h):
        return math.nan
    return h / (math.log(n_classes) / math.log(base))


def segment_diversity(counts: Iterable[float], base: float = 2.0) -> float:
    """Diversity ``D = -Σ Pᵢ log_base(Pᵢ/Pₜ)`` — Netto et al. (2018) fig. 5.

    ``counts`` are absolute per-class agent counts on a segment; ``Pₜ`` is their
    sum. Unlike :func:`shannon_entropy` (which divides by ``Pₜ`` to get
    proportions), here the ``Pᵢ`` stay as counts *outside* the log, so
    ``D = Pₜ · H`` — the entropy weighted by the number of agents on the segment.
    Returns ``nan`` if total mass is zero, ``0.0`` if all mass is in one class.
    """
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    log_b = math.log(base)
    d = 0.0
    for c in counts:
        if c > 0.0:
            d -= c * math.log(c / total) / log_b  # log(c/total) ≤ 0 → contribution ≥ 0
    return 0.0 if d < 0.0 else d


def gini_simpson(counts: Iterable[float]) -> float:
    """Gini–Simpson index ``1 − Σ pᵢ²`` of a discrete distribution.

    The **probability that two individuals drawn at random from the segment
    belong to different classes** — the "probability of intergroup contact".
    ``counts`` are non-negative masses per class (need not
    sum to 1). Returns ``nan`` if the total mass is zero (undefined), ``0.0`` if
    all mass is in one class (contact with a different group is impossible).
    Distinct from :func:`shannon_entropy`: entropy measures *diversity*, this
    measures *contact likelihood*.
    """
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    sum_sq = math.fsum((c / total) ** 2 for c in counts)
    g = 1.0 - sum_sq
    # clamp tiny negative dust from floating-point
    return 0.0 if g < 0.0 else g


@register_metric
class ShannonEntropyMetric(Metric):
    """Per-segment social diversity across each demographic attribute, on the
    aggregated per-edge table. Uses :func:`segment_diversity` (Netto 2018 fig. 5,
    the paper's formula). P4.

    ``edge_table`` is the tidy long frame from
    :meth:`segx.aggregation.edge_accumulator.EdgeAccumulator.to_frame` — one row
    per ``(mode, category, binding, u, v, key, attribute, class)`` with a
    ``mass`` column. Returns one row per ``(mode, category, binding, u, v, key,
    attribute)`` with a ``diversity`` value (``NaN`` where mass is absent).
    """

    name = "diversity"

    #: the columns identifying one segment-under-a-partition-and-attribute
    GROUP_KEYS = ("mode", "category", "binding", "u", "v", "key", "attribute")

    def compute(self, edge_table, settings: Settings):  # noqa: ARG002
        import pandas as pd

        if edge_table.empty:
            return pd.DataFrame(columns=[*self.GROUP_KEYS, self.name])
        return per_segment_values(edge_table, self.GROUP_KEYS, "mass", segment_diversity, self.name)


#: columns identifying one segment-under-a-partition-and-attribute; every
#: per-segment entropy-family metric groups by these (matches
#: ``EdgeAccumulator.to_frame`` — one row per class within the group).
_SEGMENT_KEYS: tuple[str, ...] = ("mode", "category", "binding", "u", "v", "key", "attribute")


def _per_segment(edge_table, fn, value_name: str):
    """Apply ``fn(mass_array)`` per segment-partition-attribute → tidy frame.

    ``to_frame`` emits one row per *possible* class (zeros included), so each group
    holds the full class vector and ``len(group)`` is the class count.
    """
    import pandas as pd

    if edge_table.empty:
        return pd.DataFrame(columns=[*_SEGMENT_KEYS, value_name])
    return per_segment_values(edge_table, _SEGMENT_KEYS, "mass", fn, value_name)


@register_metric
class ExperiencedSegregationMetric(Metric):
    """Experienced segregation ``1 − h`` per segment — the social *homogeneity*
    along trajectories, where ``h`` is the normalized Shannon entropy (Pielou
    evenness ∈ [0, 1]). Overall indicator = mean over segments.

    ``NaN`` where the segment carries no mass (invariant #2); ``0.0`` means a
    perfectly mixed segment (``h = 1``), ``1.0`` a fully homogeneous one.
    """

    name = "experienced_segregation"

    def compute(self, edge_table, settings: Settings):  # noqa: ARG002
        def _seg(masses):
            h = normalized_entropy(masses, n_classes=len(masses))
            return math.nan if math.isnan(h) else 1.0 - h

        return _per_segment(edge_table, _seg, self.name)


@register_metric
class IntergroupContactMetric(Metric):
    """Probability of intergroup contact ``1 − Σ pᵢ²`` (Gini–Simpson) per segment —
    the chance two people on the segment belong to different groups. Overall
    indicator = mean over segments.

    ``NaN`` where the segment carries no mass; ``0.0`` where all mass is one class.
    """

    name = "intergroup_contact"

    def compute(self, edge_table, settings: Settings):  # noqa: ARG002
        return _per_segment(edge_table, gini_simpson, self.name)
