"""CRS helpers shared by the IO layer.

Two responsibilities, kept tiny on purpose:

* **UTM auto-derivation** — given a WGS84 centroid, return the EPSG code of the
  UTM zone it falls in. Safe because SegX is city-scoped: one city → one UTM zone, no ambiguity.
* **Working-CRS resolution + enforcement** — the boundary loader calls
  :func:`resolve_working_crs` once; every other loader calls :func:`reproject_to`
  on its input so geometry can *enter* in any CRS but never *leave* this layer
  in anything except the run's working CRS.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd

from segx.errors import CRSError

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings


def utm_epsg_for(lon: float, lat: float) -> str:
    """EPSG code (e.g. ``"EPSG:32722"``) of the UTM zone containing ``(lon, lat)``.

    UTM zones are 6°-wide longitude bands numbered 1..60 starting at -180°.
    EPSG: ``32600 + zone`` north of the equator, ``32700 + zone`` south.
    """
    if not -180.0 <= lon <= 180.0:
        raise CRSError(f"longitude out of range: {lon}", details={"lon": lon})
    if not -80.0 <= lat <= 84.0:
        raise CRSError(f"latitude {lat} is outside UTM coverage (-80..84)", details={"lat": lat})
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if lat >= 0 else 32700
    return f"EPSG:{base + zone}"


def resolve_working_crs(centroid_wgs84: tuple[float, float], settings: Settings) -> str:
    """Decide the working CRS for a run.

    Order of precedence, last wins:

    1. an explicit ``crs.working_crs`` override in :class:`~segx.config.Settings`;
    2. UTM zone derived from the boundary centroid (if ``crs.auto_utm``);
    3. otherwise an explicit error — there is no safe global default.
    """
    if settings.crs.working_crs is not None:
        return settings.crs.working_crs
    if settings.crs.auto_utm:
        lon, lat = centroid_wgs84
        return utm_epsg_for(lon, lat)
    raise CRSError("no working CRS configured: set `crs.working_crs` or enable `crs.auto_utm`")


def reproject_to(
    gdf: gpd.GeoDataFrame,
    working_crs: str,
    *,
    assume_if_missing: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Reproject ``gdf`` to ``working_crs``.

    A file without an embedded CRS is assumed to be ``assume_if_missing`` — the
    plan's policy is that *input* may arrive in any CRS but the assumption is
    declared explicitly rather than guessed silently.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs(assume_if_missing)
    return gdf.to_crs(working_crs)


__all__ = ["reproject_to", "resolve_working_crs", "utm_epsg_for"]
