"""Segment-level metrics, plugin-style so new formulas can be
swapped in without touching routing or aggregation. ``base`` defines the
:class:`Metric` contract; ``registry`` is the plugin registry. ``entropy`` ships
the diversity family (Netto 2018 ``diversity``, plus ``experienced_segregation``
and ``intergroup_contact``); ``exposure`` /
``segregated_trajectories`` / ``dominant_trajectories`` implement the three P5
deliverables (Social exposure / Most exclusive segments / Dynamic segregation);
``cross_mode_exposure`` (P7.1) re-reads ``exposure`` with the *other* travel modes
on the same street counted at a discount, leaving the P5 ``exposure`` untouched
and comparable. ``fragmentation`` (P7.3) is the odd one out: it measures the
shape of a *path*, not of a segment, so it declares ``requires =
("path_table",)`` and reads the bounded sums streamed by
:class:`~segx.aggregation.path_accumulator.PathAccumulator` instead of the edge
table. Importing each module registers its metrics as a side effect."""

from segx.metrics import cross_mode_exposure as _cross_mode  # noqa: F401
from segx.metrics import dominant_trajectories as _dominant  # noqa: F401
from segx.metrics import entropy as _entropy  # noqa: F401
from segx.metrics import exposure as _exposure  # noqa: F401
from segx.metrics import fragmentation as _fragmentation  # noqa: F401
from segx.metrics import segregated_trajectories as _segregated  # noqa: F401
from segx.metrics.base import Metric
from segx.metrics.registry import METRICS, get_metric, register_metric

__all__ = ["METRICS", "Metric", "get_metric", "register_metric"]
