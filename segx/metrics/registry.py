"""Plugin registry for segment metrics.

Register with the :func:`register_metric` decorator; resolve by name with
:func:`get_metric`. New measures (Exposure / Segregated Trajectories / Dominant
Trajectories were added this way) just become registered
:class:`~segx.metrics.base.Metric` subclasses — nothing in routing or aggregation
changes.
"""

from __future__ import annotations

from segx.metrics.base import Metric

METRICS: dict[str, type[Metric]] = {}


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Class decorator: add ``cls`` to the registry under ``cls.name``."""
    if not issubclass(cls, Metric):
        raise TypeError(f"{cls!r} is not a Metric subclass")
    if cls.name in METRICS and METRICS[cls.name] is not cls:
        raise ValueError(f"metric name {cls.name!r} already registered by {METRICS[cls.name]!r}")
    METRICS[cls.name] = cls
    return cls


def get_metric(name: str) -> type[Metric]:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"no metric registered as {name!r}; available: {sorted(METRICS)}") from None


def available_metrics() -> list[str]:
    return sorted(METRICS)
