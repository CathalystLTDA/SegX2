"""Per-segment grouping of the edge table, shared by the metric plugins. P5.

Every per-segment metric does the same thing: split the tidy edge table into one
group per ``(mode, category, binding, u, v, key, attribute)`` and hand each
group's class masses to a scalar formula. ``DataFrame.groupby(...).apply`` does
that by materialising a pandas object per group, which on a city-scale table is
where the time goes — the formulas themselves are a few arithmetic operations.

:func:`segment_groups` computes the grouping once with pandas (so the semantics
are pandas' own: groups in order of first appearance, rows in their original
order within a group, rows with a missing key dropped) and exposes the groups as
slices of plain NumPy arrays. The formulas are untouched and see the same values
in the same order, so the results are identical; only the per-group overhead is
gone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["SegmentGroups", "per_segment_values", "segment_groups"]


class SegmentGroups:
    """The groups of a frame under some key columns, as array slices.

    ``keys`` is a frame with one row per group (the key columns, original
    dtypes); :meth:`arrays` yields, group by group and in the same order, the
    values of the requested columns.
    """

    __slots__ = ("_frame", "_order", "_starts", "keys")

    def __init__(self, frame: pd.DataFrame, keys: pd.DataFrame, order: np.ndarray, starts: np.ndarray):
        self._frame = frame
        self.keys = keys
        self._order = order
        self._starts = starts

    def __len__(self) -> int:
        return len(self.keys)

    def arrays(self, *columns: str) -> Iterator[tuple[np.ndarray, ...]]:
        """One tuple of per-column arrays for each group, in group order."""
        sorted_columns = [self._frame[column].to_numpy()[self._order] for column in columns]
        starts = self._starts
        for i in range(len(starts) - 1):
            lo, hi = starts[i], starts[i + 1]
            yield tuple(values[lo:hi] for values in sorted_columns)


def segment_groups(frame: pd.DataFrame, keys: Sequence[str]) -> SegmentGroups:
    """Group *frame* by *keys* the way ``groupby(keys, sort=False)`` does."""
    keys = list(keys)
    codes = frame.groupby(keys, sort=False).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")  # stable: original row order within a group
    order = order[codes[order] >= 0]  # a row with a missing key belongs to no group
    sorted_codes = codes[order]
    if len(sorted_codes):
        firsts = np.concatenate(([0], np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1))
    else:
        firsts = np.empty(0, dtype=np.int64)
    starts = np.concatenate((firsts, [len(sorted_codes)])).astype(np.int64)
    key_frame = frame[keys].iloc[order[firsts]].reset_index(drop=True)
    return SegmentGroups(frame, key_frame, order, starts)


def per_segment_values(
    frame: pd.DataFrame,
    keys: Sequence[str],
    column: str,
    fn: Callable[[np.ndarray], float],
    value_name: str,
) -> pd.DataFrame:
    """``fn(values of column)`` per group → the key columns plus *value_name*.

    The frame ``frame.groupby(keys, sort=False)[column].apply(lambda s:
    fn(s.to_numpy())).reset_index(name=value_name)`` returns, without building a
    Series per group.
    """
    groups = segment_groups(frame, keys)
    out = groups.keys
    out[value_name] = np.array([fn(values) for (values,) in groups.arrays(column)], dtype=float)
    return out
