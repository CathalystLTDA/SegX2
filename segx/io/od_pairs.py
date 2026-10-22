"""Load user-supplied origin → destination pairs (CSV or GeoPackage).

There is **no fixed schema** — :class:`~segx.config.ODColumnMapping` declares
which columns mean what. For each endpoint (origin and destination) we try, in
order:

1. a **geometry column** (only meaningful when the file is a GeoPackage and its
   active geometry column matches the mapping);
2. a **WKT column** of point strings;
3. a **lon + lat** pair of numeric columns;
4. an **id-reference column** — values are IDs that join to the demographics
   ``polygon_id_column`` (e.g. census-block ``hGEOID``/``wGEOID``); the endpoint
   becomes the representative point of the matching geometry. Requires the
   ``demographics`` argument;
5. an **address column** — geocoded; deferred (raises ``NotImplementedError``).

If a column listed in the mapping doesn't exist in the file, we silently skip
to the next strategy — so the user can leave most defaults in place and just
supply *any one* of (geom | WKT | lon+lat | id-ref) per endpoint.

``category``, ``mode``, ``weight`` and ``origin_id`` columns are likewise
optional; missing ones fall back to the mapping's ``default_*`` values (and
``mode`` defaults to :meth:`ODColumnMapping.mode_for` of the row's category).

If OD pairs are supplied, their distinct origins become *the* origin set —
nothing is synthesised from demographics (see :mod:`segx.population.origins`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
from shapely import wkt as shp_wkt

from segx.errors import EmptyInputError, GeometryError, InputFileNotFoundError, SchemaError
from segx.io._geometry import representative_points
from segx.io.demographics import class_columns_for

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import ODColumnMapping, Settings
    from segx.io.demographics import LoadedDemographics


Side = Literal["origin", "dest"]


@dataclass(frozen=True, slots=True)
class LoadedODPairs:
    """Validated, reprojected OD pairs.

    The active geometry on :attr:`gdf` is the **origin** point; the destination
    point lives on a parallel column ``dest_geom`` (also in the working CRS).
    Other columns: ``origin_id`` (str), ``dest_id`` (str), ``category`` (str),
    ``mode`` (str), ``weight`` (float).
    """

    gdf: gpd.GeoDataFrame


def _endpoint_columns(
    mapping: ODColumnMapping, side: Side
) -> tuple[str | None, str | None, str | None, str | None, str | None, str | None]:
    if side == "origin":
        return (
            mapping.origin_geom,
            mapping.origin_wkt,
            mapping.origin_lon,
            mapping.origin_lat,
            mapping.origin_id_ref,
            mapping.origin_address,
        )
    return (
        mapping.dest_geom,
        mapping.dest_wkt,
        mapping.dest_lon,
        mapping.dest_lat,
        mapping.dest_id_ref,
        mapping.dest_address,
    )


def _id_geometry_lookup(demographics: LoadedDemographics) -> tuple[dict[str, object], str]:
    """Map each demographics ``id`` to its representative point (working CRS).

    Returns the mapping plus the working CRS string the points carry, so the
    id-reference branch can declare the right CRS on the resulting GeoSeries.
    """
    g = demographics.gdf
    pts = representative_points(g.geometry)
    crs = str(g.crs)
    return dict(zip(g["id"].astype(str), pts.to_numpy(), strict=True)), crs


def _resolve_endpoint(
    df: pd.DataFrame | gpd.GeoDataFrame,
    mapping: ODColumnMapping,
    side: Side,
    *,
    embedded_crs: str | None,
    fallback_crs: str,
    id_lookup: dict[str, object] | None,
    id_lookup_crs: str | None,
) -> gpd.GeoSeries:
    """Build a GeoSeries of point geometries for one endpoint side."""
    geom_col, wkt_col, lon_col, lat_col, id_ref_col, addr_col = _endpoint_columns(mapping, side)
    cols = set(df.columns)

    if isinstance(df, gpd.GeoDataFrame) and df.geometry.name == geom_col:
        crs = embedded_crs or fallback_crs
        return gpd.GeoSeries(df.geometry.to_numpy(), crs=crs)
    if wkt_col in cols:
        pts = df[wkt_col].apply(shp_wkt.loads)
        return gpd.GeoSeries(pts.to_numpy(), crs=fallback_crs)
    if lon_col in cols and lat_col in cols:
        pts = gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float))
        return gpd.GeoSeries(pts, crs=fallback_crs)
    if id_ref_col in cols:
        if id_lookup is None:
            raise SchemaError(
                f"{side} given by id-reference column {id_ref_col!r}, but no demographics "
                "geometry was supplied to resolve those IDs against. Pass the loaded "
                "demographics to load_od_pairs.",
                details={"side": side, "id_ref_column": id_ref_col},
            )
        ids = df[id_ref_col].astype(str)
        resolved = ids.map(id_lookup)
        unmatched = sorted(set(ids[resolved.isna()]))
        if unmatched:
            raise SchemaError(
                f"{len(unmatched)} {side} id-reference value(s) in column {id_ref_col!r} "
                f"match no demographics geometry id (e.g. {unmatched[:5]}). Every OD "
                "endpoint id must exist in the demographics file.",
                details={"side": side, "id_ref_column": id_ref_col, "unmatched_sample": unmatched[:20]},
            )
        return gpd.GeoSeries(resolved.to_numpy(), crs=id_lookup_crs or fallback_crs)
    if addr_col in cols:
        raise NotImplementedError(
            f"{side} resolution by address ({addr_col!r}) requires geocoding; that is "
            f"deferred to a later phase. Provide {wkt_col!r} or {lon_col!r}+{lat_col!r} "
            "in the OD file for now."
        )
    # Only the strategies the mapping actually uses (None = "not used").
    tried = [c for c in (geom_col, wkt_col, lon_col, lat_col, id_ref_col, addr_col) if c is not None]
    raise SchemaError(
        f"could not resolve {side} location in the OD file: none of the mapped columns "
        f"{tried} are present. Available columns: {sorted(map(str, cols))}",
        details={"side": side, "tried": tried, "available": sorted(map(str, cols))},
    )


def _coordinate_ids(points: gpd.GeoSeries, precision_m: float) -> pd.Series:
    """Destination identity derived from the location, for files that name none.

    Two flows ending at the same place must count as **one** destination, or the
    per-origin destination count in :mod:`segx.analysis.od_spread` degenerates
    into a row count and every class trivially "reaches" as many places as it has
    rows. Absent an id column, the place itself is the identity: the coordinates
    snapped to a ``precision_m`` grid (metres — the points are already in the
    working CRS). The label keeps the snapped coordinates rather than an opaque
    counter so a destination can be found on a map from the table alone.
    """
    import numpy as np

    precision = float(precision_m) if precision_m > 0 else 1.0
    xs = np.round(points.x.to_numpy() / precision) * precision
    ys = np.round(points.y.to_numpy() / precision) * precision
    return pd.Series([f"@{x:.0f},{y:.0f}" for x, y in zip(xs, ys, strict=True)], name="dest_id")


def _validate_points(series: gpd.GeoSeries, side: Side) -> None:
    """Geometries must be non-null and Point — anything else is a load error."""
    if series.isna().any():
        raise GeometryError(f"OD pairs contain null {side} geometries", details={"side": side})
    bad = {t for t in series.geom_type if t != "Point"}
    if bad:
        raise GeometryError(
            f"OD {side} geometries must be Point; got {sorted(bad)}. "
            "Lines/polygons are not a meaningful trajectory endpoint.",
            details={"side": side, "types": sorted(bad)},
        )


def load_od_pairs(
    path: str | Path,
    settings: Settings,
    *,
    working_crs: str,
    demographics: LoadedDemographics | None = None,
) -> LoadedODPairs:
    """Read an OD-pairs file applying the configured column mapping.

    ``demographics`` is only needed when an endpoint is given by id-reference
    (``origin_id_ref`` / ``dest_id_ref``): its geometry is the id → point lookup.
    When demographic attributes are sourced from the OD file
    (``settings.demographics.attribute_source == 'od'``), the schema's per-class
    columns are carried through onto the returned frame so
    :mod:`segx.population.attributes` can sum them per origin.
    """
    path = Path(path)
    if not path.exists():
        raise InputFileNotFoundError(f"OD pairs file not found: {path}", details={"path": str(path)})
    mapping = settings.od_columns
    fallback_crs = settings.crs.geographic_crs

    if path.suffix.lower() == ".csv":
        df: pd.DataFrame | gpd.GeoDataFrame = pd.read_csv(path)
        embedded_crs: str | None = None
    else:
        df = gpd.read_file(path)
        embedded_crs = str(df.crs) if df.crs is not None else None

    if len(df) == 0:
        raise EmptyInputError(f"OD pairs file is empty: {path}", details={"path": str(path)})

    id_lookup: dict[str, object] | None = None
    id_lookup_crs: str | None = None
    if demographics is not None:
        id_lookup, id_lookup_crs = _id_geometry_lookup(demographics)

    endpoint_kwargs = {
        "embedded_crs": embedded_crs,
        "fallback_crs": fallback_crs,
        "id_lookup": id_lookup,
        "id_lookup_crs": id_lookup_crs,
    }
    origin = _resolve_endpoint(df, mapping, "origin", **endpoint_kwargs)
    dest = _resolve_endpoint(df, mapping, "dest", **endpoint_kwargs)
    _validate_points(origin, "origin")
    _validate_points(dest, "dest")

    origin_proj = origin.to_crs(working_crs)
    dest_proj = dest.to_crs(working_crs)

    cols = set(df.columns)
    n = len(df)

    if mapping.origin_id in cols:
        origin_id = df[mapping.origin_id].astype(str).reset_index(drop=True)
    elif mapping.origin_id_ref in cols:
        # No explicit origin_id, but the origin is placed by id-reference — that
        # reference *is* the origin identity (e.g. LODES ``hGEOID``), so flows
        # from the same home block collapse to one distinct origin.
        origin_id = df[mapping.origin_id_ref].astype(str).reset_index(drop=True)
    else:
        origin_id = pd.Series([str(i) for i in range(n)], name="origin_id")

    if mapping.dest_id in cols:
        dest_id = df[mapping.dest_id].astype(str).reset_index(drop=True)
    elif mapping.dest_id_ref in cols:
        # Same reasoning as the origin: the reference that *placed* the endpoint
        # is its identity (LODES ``wGEOID``), so every flow into a work block is
        # one destination, however many home blocks send workers to it.
        dest_id = df[mapping.dest_id_ref].astype(str).reset_index(drop=True)
    else:
        dest_id = _coordinate_ids(dest_proj, mapping.dest_id_precision_m)

    if mapping.category in cols:
        category = df[mapping.category].astype(str).reset_index(drop=True)
    else:
        category = pd.Series([mapping.default_category] * n)

    if mapping.mode in cols:
        mode = df[mapping.mode].astype(str).reset_index(drop=True)
    else:
        mode = category.map(mapping.mode_for)

    if mapping.weight in cols:
        weight = df[mapping.weight].astype(float).reset_index(drop=True)
    else:
        weight = pd.Series([mapping.default_weight] * n, dtype=float)

    out = gpd.GeoDataFrame(
        {
            "origin_id": origin_id,
            "dest_id": dest_id,
            "category": category,
            "mode": mode,
            "weight": weight,
            "dest_geom": dest_proj.reset_index(drop=True).to_numpy(),
        },
        geometry=origin_proj.reset_index(drop=True),
        crs=working_crs,
    )

    # Carry through the schema's per-class count columns when demographics are
    # sourced from this OD file (LODES-style). population.attributes sums them
    # per distinct origin. Missing class columns are tolerated here and surface
    # as a typed fault later, where the offending origin can be named.
    if settings.demographics.attribute_source == "od":
        for attr in settings.demographics.attributes:
            for col in class_columns_for(attr):
                if col in cols:
                    out[col] = df[col].astype(float).reset_index(drop=True)

    return LoadedODPairs(gdf=out)


def origin_points(loaded: LoadedODPairs) -> gpd.GeoDataFrame:
    """Distinct origins, used by :mod:`segx.population.origins` to build the
    origin set when OD pairs are supplied (nothing is synthesised from
    demographics in that branch)."""
    g = loaded.gdf[["origin_id", "geometry"]].drop_duplicates(subset="origin_id")
    return g.reset_index(drop=True)


__all__ = ["LoadedODPairs", "load_od_pairs", "origin_points"]
