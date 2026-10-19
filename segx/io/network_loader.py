"""Obtain per-mode street graphs: OSMnx fetch (default) or a user shapefile /
GeoPackage override, normalised into the shape :mod:`segx.network.builder`
expects. Goes through :mod:`segx.network.cache` so re-runs are reproducible. P0: stub. P1: implement walk + drive."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings


def load_network(boundary, mode: str, settings: Settings):  # noqa: ARG001
    """Return a routable graph for ``mode`` ('walk' | 'drive'). P1."""
    raise NotImplementedError("network loading lands in P1")
