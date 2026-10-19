"""Load the study-area boundary polygon (GeoJSON / GPKG / shapefile).

Dissolves the input features into one (multi)polygon, resolves the working CRS
(UTM zone of the boundary centroid unless ``crs.working_crs`` overrides), reprojects, and reports the WGS84 bounding
box (needed for OSM fetches and the cache key — ``network/cache.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from segx.errors import EmptyInputError, InputFileNotFoundError
from segx.io._crs import resolve_working_crs

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings


@dataclass(frozen=True, slots=True)
class LoadedBoundary:
    """Resolved study-area boundary.

    ``geometry`` is one (multi)polygon in :attr:`crs` (the working CRS for the
    run). ``bbox_wgs84`` is kept because OSMnx/QuackOSM and the cache key both
    need a WGS84 bounding box.
    """

    geometry: BaseGeometry
    crs: str  # working CRS, e.g. "EPSG:32722"
    bbox_wgs84: tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


def load_boundary(path: str | Path, settings: Settings) -> LoadedBoundary:
    """Read a boundary file, dissolve, and reproject to the working CRS."""
    path = Path(path)
    if not path.exists():
        raise InputFileNotFoundError(f"boundary file not found: {path}", details={"path": str(path)})

    raw = gpd.read_file(path)
    if raw.empty:
        raise EmptyInputError(f"boundary file contains no features: {path}", details={"path": str(path)})
    if raw.crs is None:
        # explicit assumption rather than silent guess — see _crs.reproject_to
        raw = raw.set_crs(settings.crs.geographic_crs)

    wgs = raw.to_crs(settings.crs.geographic_crs)
    centroid = wgs.union_all().centroid
    working_crs = resolve_working_crs((centroid.x, centroid.y), settings)

    projected = raw.to_crs(working_crs)
    geometry = projected.union_all()
    minx, miny, maxx, maxy = wgs.total_bounds
    return LoadedBoundary(
        geometry=geometry,
        crs=working_crs,
        bbox_wgs84=(float(minx), float(miny), float(maxx), float(maxy)),
    )


__all__ = ["LoadedBoundary", "load_boundary"]
