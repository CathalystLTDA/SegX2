"""Snap origins / POIs / OD endpoints to the nearest node of a mode graph.

Every snapped point records the snap distance, so absurd snaps are *visible*
rather than silently used. A point whose nearest node is beyond
``settings.network.snap_tolerance_m`` is flagged via ``over_tolerance`` —
callers (the orchestrator, P3 routing) decide whether to drop it or proceed
with a warning. This is the *typed* form of "unreachable point" (invariant #2).

The graph contract: any NetworkX graph with node attributes ``x`` and ``y``
in the working CRS works — that includes OSMnx ``MultiDiGraph`` outputs from
:mod:`segx.network.builder` and the synthetic test graph in
:mod:`segx.validation.synthetic_city`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from segx.errors import CRSError, GraphError

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

    from segx.config import Settings


@dataclass(frozen=True, slots=True)
class SnappedPoints:
    """Result of :func:`snap_points`.

    :attr:`gdf` is a copy of the input GeoDataFrame with three new columns:

    * ``node_id`` — the snapped graph node identifier (matches ``graph.nodes``);
    * ``snap_distance_m`` — Euclidean distance from the point to that node, in
      the working CRS units (metres for UTM);
    * ``over_tolerance`` — ``True`` if ``snap_distance_m > tolerance_m``.

    :attr:`tolerance_m` and :attr:`n_over_tolerance` summarise the audit at a
    glance so the orchestrator can log/raise without re-scanning the table.
    """

    gdf: gpd.GeoDataFrame
    tolerance_m: float
    n_over_tolerance: int


def _nodes_as_gdf(graph: nx.Graph, working_crs: str, node_subset: set | None = None) -> gpd.GeoDataFrame:
    """Materialise graph nodes as a GeoDataFrame for spatial joining."""
    ids: list = []
    xs: list[float] = []
    ys: list[float] = []
    for nid, data in graph.nodes(data=True):
        if node_subset is not None and nid not in node_subset:
            continue
        if "x" not in data or "y" not in data:
            raise GraphError(f"graph node {nid!r} is missing 'x'/'y' attributes")
        ids.append(nid)
        xs.append(float(data["x"]))
        ys.append(float(data["y"]))
    return gpd.GeoDataFrame(
        {"node_id": ids},
        geometry=gpd.points_from_xy(xs, ys),
        crs=working_crs,
    )


def snap_points(
    points: gpd.GeoDataFrame,
    graph: nx.Graph,
    settings: Settings,
    *,
    working_crs: str,
    node_subset: set | None = None,
) -> SnappedPoints:
    """Snap each point to the nearest graph node, recording the distance.

    ``points`` is assumed to be in the working CRS already (the IO loaders
    guarantee this). The function does not silently reproject — a CRS mismatch
    would mean a contract violation upstream and we'd rather raise than mask it.

    ``node_subset`` restricts the candidate nodes. It exists for the P7 merged
    transit graph, whose synthetic platform and route-stop nodes sit at real
    coordinates and would otherwise attract a nearby origin: snapping an origin
    onto a *route-stop* would let it ride without ever paying the board cost, and
    snapping onto a *platform* would skip the walk to the stop. Real-world points
    always snap to street nodes (see
    :func:`segx.network.transit.street_nodes`); the transit nodes are reachable
    only through the connector edges, as the model intends.
    """
    tolerance = float(settings.network.snap_tolerance_m)

    if points.empty:
        out = points.copy()
        out["node_id"] = pd.Series([], dtype=object)
        out["snap_distance_m"] = pd.Series([], dtype=float)
        out["over_tolerance"] = pd.Series([], dtype=bool)
        return SnappedPoints(gdf=out, tolerance_m=tolerance, n_over_tolerance=0)

    if graph.number_of_nodes() == 0:
        raise GraphError("graph has no nodes; cannot snap")

    if (
        points.crs is not None
        and str(points.crs) != working_crs
        and points.crs.to_epsg() != int(working_crs.split(":")[1])
    ):
        raise CRSError(
            f"snap_points expects input in working CRS {working_crs}; got {points.crs}. "
            "Reproject upstream (the IO loaders do this) rather than here.",
            details={"expected": working_crs, "got": str(points.crs)},
        )

    nodes_gdf = _nodes_as_gdf(graph, working_crs, node_subset)
    if nodes_gdf.empty:
        raise GraphError("no candidate nodes to snap to (node_subset excluded every node)")

    snapped = points.sjoin_nearest(
        nodes_gdf[["node_id", "geometry"]],
        distance_col="snap_distance_m",
        how="left",
    )
    if "index_right" in snapped.columns:
        snapped = snapped.drop(columns=["index_right"])
    # sjoin_nearest can emit duplicates on equidistant ties; keep first
    snapped = snapped[~snapped.index.duplicated(keep="first")]

    snapped["over_tolerance"] = snapped["snap_distance_m"] > tolerance
    n_over = int(snapped["over_tolerance"].sum())
    return SnappedPoints(gdf=snapped, tolerance_m=tolerance, n_over_tolerance=n_over)


__all__ = ["SnappedPoints", "snap_points"]
