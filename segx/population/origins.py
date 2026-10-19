"""Build the origin set (P2).

The origin source is conditional, decided at runtime:

* **OD pairs supplied** → the distinct origins in that file; nothing is
  synthesised from demographics. Their geometry was already resolved by
  :mod:`segx.io.od_pairs` (including the id-reference branch that places a
  census-block ``hGEOID`` at its polygon's representative point).
* **OD pairs absent, demographic input is polygons** → one origin per polygon,
  at its representative point (centroid when interior, else point-on-surface —
  see :func:`segx.io._geometry.representative_points`).
* **OD pairs absent, demographic input is points** → residence points pass
  through unchanged.

The demographic vector is *not* attached here — that is
:func:`segx.population.attributes.attach_attributes`'s job, run immediately
after. This module only fixes each origin's identity, location and base weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd

from segx.io._geometry import representative_points
from segx.io.od_pairs import origin_points

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings
    from segx.io.demographics import LoadedDemographics
    from segx.io.od_pairs import LoadedODPairs


def build_origins(
    demographics: LoadedDemographics,
    od_pairs: LoadedODPairs | None,
    settings: Settings,  # noqa: ARG001 — reserved (e.g. future weight policy); kept for signature stability
) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of origin points: ``origin_id`` (str),
    ``geometry`` (working CRS) and ``weight`` (float, default ``1.0``).

    The demographic vector column is added downstream by
    :func:`segx.population.attributes.attach_attributes`.
    """
    if od_pairs is not None:
        origins = origin_points(od_pairs)
    elif demographics.kind == "point":
        g = demographics.gdf
        origins = gpd.GeoDataFrame(
            {"origin_id": g["id"].astype(str).to_numpy()},
            geometry=g.geometry.to_numpy(),
            crs=g.crs,
        )
    else:  # polygons → representative point per polygon
        g = demographics.gdf
        pts = representative_points(g.geometry)
        origins = gpd.GeoDataFrame(
            {"origin_id": g["id"].astype(str).to_numpy()},
            geometry=pts.to_numpy(),
            crs=g.crs,
        )

    origins = origins.reset_index(drop=True)
    origins["weight"] = 1.0
    return origins


__all__ = ["build_origins"]
