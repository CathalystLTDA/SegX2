"""Cheap, display-only layer inspection for the host app's Setup screen.

Two read-only helpers behind the ``inspect_layer`` / ``get_layer_preview`` RPC
methods, so the desktop app needs no GDAL of its own:

* :func:`inspect_layer` — header facts (CRS, feature count, geometry kind,
  WGS84 bbox, driver) from :func:`pyogrio.read_info`; no geometry is loaded.
* :func:`layer_preview` — the first ``max_features`` features as lightweight
  WGS84 GeoJSON for drawing a preview map.

Neither feeds the pipeline: a run always re-reads its inputs through the real
loaders (``io.boundary``, ``io.demographics``, …). A layer without a CRS is
reported as ``crs = None`` and *displayed* as if it were WGS84 — the same
explicit assumption the loaders make (``crs.geographic_crs``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pyogrio
from pyogrio.errors import DataLayerError, DataSourceError
from pyproj import CRS, Transformer

from segx.errors import InputError, InputFileNotFoundError

#: CRS a layer without one is displayed in (matches ``CRSConfig.geographic_crs``).
_DISPLAY_CRS = "EPSG:4326"

#: Non-geometry columns carried into preview properties (keeps the reply small).
_MAX_PREVIEW_PROPERTIES = 5

#: Simplification tolerance as a fraction of the layer's larger WGS84 extent.
_SIMPLIFY_FRACTION = 1 / 2000


def _info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise InputFileNotFoundError(f"layer file not found: {path}", details={"path": str(path)})
    try:
        return pyogrio.read_info(path, force_feature_count=True, force_total_bounds=True)
    except (DataSourceError, DataLayerError) as exc:
        raise InputError(f"cannot read layer: {path}: {exc}", details={"path": str(path)}) from exc


def _crs_label(raw: str | None) -> str | None:
    """``"EPSG:31982"`` when the CRS has an authority code, else its PROJ string."""
    if not raw:
        return None
    crs = CRS.from_user_input(raw)
    authority = crs.to_authority()
    return f"{authority[0]}:{authority[1]}" if authority else crs.to_string()


def _geometry_kind(raw: str | None) -> str:
    """Collapse an OGR geometry type name to POINT / LINE / POLYGON / TABLE."""
    if not raw:
        return "TABLE"
    name = raw.lower()
    if "point" in name:
        return "POINT"
    if "line" in name:
        return "LINE"
    if "polygon" in name:
        return "POLYGON"
    return "TABLE"


def _bbox_wgs84(info: dict[str, Any]) -> list[float] | None:
    bounds = info.get("total_bounds")
    if bounds is None or not all(math.isfinite(b) for b in bounds):
        return None
    source = info.get("crs") or _DISPLAY_CRS
    transformer = Transformer.from_crs(source, _DISPLAY_CRS, always_xy=True)
    return [float(v) for v in transformer.transform_bounds(*bounds)]


def inspect_layer(path: str | Path) -> dict[str, Any]:
    """Header facts about one input layer, without loading its geometry."""
    info = _info(path)
    kind = _geometry_kind(info.get("geometry_type"))
    return {
        "crs": _crs_label(info.get("crs")),
        "feature_count": int(info["features"]),
        "geometry_type": kind,
        "bbox": None if kind == "TABLE" else _bbox_wgs84(info),
        "driver": str(info.get("driver") or ""),
    }


def _scalar(value: Any) -> str | float | None:
    """JSON-safe scalar for a preview property (NaN/None → None)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    if hasattr(value, "item"):  # numpy scalar
        return _scalar(value.item())
    return str(value)


def layer_preview(path: str | Path, max_features: int = 1000) -> dict[str, Any]:
    """The first ``max_features`` features of a layer as simplified WGS84 GeoJSON."""
    info = _info(path)
    kind = _geometry_kind(info.get("geometry_type"))
    result: dict[str, Any] = {
        "geometry_type": kind,
        "total_features": int(info["features"]),
        "features": [],
    }
    if kind == "TABLE" or max_features <= 0:
        return result

    try:
        gdf = pyogrio.read_dataframe(path, max_features=max_features)
    except (DataSourceError, DataLayerError) as exc:
        raise InputError(f"cannot read layer: {path}: {exc}", details={"path": str(path)}) from exc
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return result
    if gdf.crs is None:
        gdf = gdf.set_crs(_DISPLAY_CRS)
    gdf = gdf.to_crs(_DISPLAY_CRS)

    geoms = gdf.geometry
    if kind != "POINT":
        minx, miny, maxx, maxy = geoms.total_bounds
        tolerance = max(maxx - minx, maxy - miny) * _SIMPLIFY_FRACTION
        if tolerance > 0:
            geoms = geoms.simplify(tolerance, preserve_topology=True)

    columns = [c for c in gdf.columns if c != gdf.geometry.name][:_MAX_PREVIEW_PROPERTIES]
    for geom, (_, row) in zip(geoms, gdf.iterrows(), strict=True):
        result["features"].append(
            {
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {str(c): _scalar(row[c]) for c in columns},
            }
        )
    return result
