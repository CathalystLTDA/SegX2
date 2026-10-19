"""Small geometry helpers shared across IO and population builders (P2).

The one thing here is :func:`representative_points`: turn an arbitrary
(polygon or point) :class:`~geopandas.GeoSeries` into one representative
**point** per feature. It is used in two places that must agree:

* :mod:`segx.population.origins` — the polygon-centroid origin branch;
* :mod:`segx.io.od_pairs` — resolving an ID-reference OD endpoint
  (``hGEOID``/``wGEOID``) to the point of its census-block geometry.

Policy: use the **centroid** when it falls
inside its own geometry; otherwise fall back to the guaranteed-inside
``representative_point`` (Shapely's point-on-surface). Point inputs return
themselves. Keeping both call sites on this one function guarantees an OD origin
placed by ID lands on exactly the same coordinate as the centroid origin built
from the same polygon.
"""

from __future__ import annotations

import geopandas as gpd


def representative_points(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """One representative interior point per geometry (centroid, else
    point-on-surface). Index and CRS are preserved."""
    centroids = geoms.centroid
    inside = centroids.within(geoms)
    on_surface = geoms.representative_point()
    chosen = [c if bool(ok) else r for c, r, ok in zip(centroids, on_surface, inside, strict=True)]
    return gpd.GeoSeries(chosen, index=geoms.index, crs=geoms.crs)


__all__ = ["representative_points"]
