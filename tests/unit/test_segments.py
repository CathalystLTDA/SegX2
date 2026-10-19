"""The shared per-segment grouping the metric plugins use (`segx.metrics._segments`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from segx.metrics._segments import per_segment_values, segment_groups
from segx.metrics.entropy import shannon_entropy

KEYS = ["mode", "u", "v"]


def _table() -> pd.DataFrame:
    # deliberately interleaved: a group's rows are not contiguous, and one row has
    # a missing key (pandas drops it from every group)
    return pd.DataFrame(
        {
            "mode": ["walk", "drive", "walk", "walk", "drive", "walk", None, "drive"],
            "u": [1, 1, 2, 1, 1, 2, 9, 1],
            "v": [2, 2, 3, 2, 2, 3, 9, 2],
            "class": ["low", "low", "low", "mid", "mid", "mid", "low", "high"],
            "mass": [3.0, 0.0, 5.0, 1.0, 0.0, 5.0, 7.0, 0.0],
        }
    )


def test_per_segment_values_is_the_pandas_groupby_apply_it_replaces() -> None:
    """Same groups, same order, same values, same dtypes — the helper exists to
    skip the per-group pandas objects, not to change what a metric sees."""
    table = _table()
    want = table.groupby(KEYS, sort=False)["mass"].apply(lambda s: shannon_entropy(s.to_numpy())).reset_index(name="h")
    got = per_segment_values(table, KEYS, "mass", shannon_entropy, "h")
    pd.testing.assert_frame_equal(got, want)
    assert np.isnan(got.loc[1, "h"])  # the all-zero drive segment: undefined, not 0.0


def test_segment_groups_keep_row_order_within_a_group() -> None:
    table = _table()
    groups = segment_groups(table, KEYS)
    assert len(groups) == 3
    assert [tuple(r) for r in groups.keys.itertuples(index=False)] == [("walk", 1, 2), ("drive", 1, 2), ("walk", 2, 3)]
    got = [(list(m), list(c)) for m, c in groups.arrays("mass", "class")]
    assert got == [
        ([3.0, 1.0], ["low", "mid"]),
        ([0.0, 0.0, 0.0], ["low", "mid", "high"]),
        ([5.0, 5.0], ["low", "mid"]),
    ]


def test_segment_groups_of_an_empty_frame() -> None:
    groups = segment_groups(_table().iloc[0:0], KEYS)
    assert len(groups) == 0 and list(groups.arrays("mass")) == []
