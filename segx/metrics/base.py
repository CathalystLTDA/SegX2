"""The metric contract.

A metric reads the aggregated per-edge demographic table (partitioned by
``(mode, category, binding)``) plus the resolved :class:`~segx.config.Settings`,
and returns a frame of per-segment values it can be joined back onto the segment
table by ``(graph_id, edge_id)``. Metrics must propagate ``NaN`` explicitly for
edges with no demographic mass under a partition — never silently zero.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings


class Metric(abc.ABC):
    """Base class for all segment metrics."""

    #: unique short name used for registration and as the output column prefix
    name: str

    #: the per-segment value column :meth:`summarize` averages into a
    #: network-wide indicator. Defaults to ``name`` (the convention: a per-segment
    #: metric names its value column after itself). Override when the frame's value
    #: lives under a different column (e.g. ``dominant_group`` → ``dominance``).
    value_column: str = ""

    #: Extra frames this metric needs *besides* the edge table, passed to
    #: :meth:`compute` as keyword arguments of the same name. Empty for every
    #: per-segment metric — the edge table is all they read. ``path_fragmentation``
    #: (P7.3) declares ``("path_table",)`` because path shape is not a property of
    #: a segment and the edge table has discarded path identity by design
    #: (invariant #6). The orchestrator supplies only the names listed here, so a
    #: metric can never reach for a frame it did not declare.
    requires: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) in (None, ""):
            raise TypeError(f"{cls.__name__} must set a non-empty `name` attribute")

    @abc.abstractmethod
    def compute(self, edge_table, settings: Settings):
        """Return a per-segment frame of metric values. Implemented per metric."""
        raise NotImplementedError

    def summarize(self, per_segment, settings: Settings):  # noqa: ARG002
        """Reduce the per-segment frame to a network-wide indicator per
        ``(mode, category, binding, attribute)``.

        Default: the mean of :attr:`value_column` (NaN-skipping, so segments with
        no demographic mass drop out — "the average experienced segregation across
        all segments"). Per-group metrics override this. Returns an empty frame
        with the expected columns when ``per_segment`` is empty.
        """
        import pandas as pd

        col = self.value_column or self.name
        keys = ["mode", "category", "binding", "attribute"]
        if per_segment.empty:
            return pd.DataFrame(columns=[*keys, col])
        return per_segment.groupby(keys, sort=False)[col].mean().reset_index()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Metric {self.name!r}>"
