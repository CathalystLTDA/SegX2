"""Load demographic input — census/sector **polygons** or residence **points**.

Schema is *not* hard-coded. ``settings.demographics`` (a
:class:`~segx.config.DemographicSchema`) declares which attributes participate,
the classes / bin edges of each, and the polygon-id + population columns; this
loader validates the file against that schema, reprojects to the working CRS,
and returns a GeoDataFrame with a canonical column layout.

**Source column convention for P1:** for each attribute, the input file is
expected to carry **one count column per class**:

* ``categorical`` → ``f"{source_column}_{category_slug}"`` for each declared category;
* ``binned`` → ``f"{source_column}_b{i}"`` for ``i = 0 … N-1`` (one column per bin,
  in ascending order — class label strings like ``"[0, 1320)"`` are not column-safe);
* ``numeric`` → a single raw column ``source_column``.

``source_column`` defaults to the attribute's own ``name``. A bespoke per-class
column-name layout can be supported later; this convention is what the contract
fixtures will use, and `class_columns_for` is the one place that knows it.

Geometries may be polygons (one census sector → one row) **or** points (one
residence → one row); both are accepted and reported via :attr:`LoadedDemographics.kind`,
which the origin-source branch in :mod:`segx.population.origins` reads to decide
whether to take centroids or pass points through (P2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd

from segx.errors import EmptyInputError, GeometryError, InputFileNotFoundError, SchemaError
from segx.io._crs import reproject_to

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import DemographicAttribute, Settings


GeometryKind = Literal["polygon", "point"]


@dataclass(frozen=True, slots=True)
class LoadedDemographics:
    """Validated, reprojected demographics ready for the origins step.

    Columns on :attr:`gdf`:

    * ``id`` — polygon/residence identifier (``str``).
    * ``geometry`` — in the working CRS.
    * ``pop`` — total population in the row (``float``).
    * the per-class count columns for every attribute (names recorded in
      :attr:`class_columns`, keyed by attribute name).
    """

    gdf: gpd.GeoDataFrame
    kind: GeometryKind
    class_columns: dict[str, list[str]]


_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _slug(label: str) -> str:
    return _SLUG_RE.sub("_", label).strip("_")


def class_columns_for(attribute: DemographicAttribute) -> list[str]:
    """Per-class count columns expected in the input for *attribute*.

    Single source of truth for the column-naming convention documented at the
    top of this module. An attribute may override the convention entirely by
    declaring explicit :attr:`~segx.config.DemographicAttribute.class_columns`
    (e.g. LODES ``SE01``/``SE02``/``SE03``); when present, those are returned
    verbatim, in class order.
    """
    if attribute.class_columns is not None:
        return list(attribute.class_columns)
    prefix = attribute.source_column or attribute.name
    if attribute.kind == "categorical":
        assert attribute.categories is not None
        return [f"{prefix}_{_slug(c)}" for c in attribute.categories]
    if attribute.kind == "binned":
        assert attribute.bins is not None
        return [f"{prefix}_b{i}" for i in range(len(attribute.bins) - 1)]
    return [prefix]


def load_demographics(
    path: str | Path,
    settings: Settings,
    *,
    working_crs: str,
) -> LoadedDemographics:
    """Read, validate against the configured schema, and reproject."""
    path = Path(path)
    if not path.exists():
        raise InputFileNotFoundError(f"demographics file not found: {path}", details={"path": str(path)})
    raw = gpd.read_file(path)
    if raw.empty:
        raise EmptyInputError(f"demographics file is empty: {path}", details={"path": str(path)})

    schema = settings.demographics
    # When attributes are sourced from the OD file (LODES-style), the demographics
    # file is only a geometry/ID lookup: it needs the id column + geometry, and
    # the population / per-class columns are optional. Otherwise every per-class
    # column declared by the schema is required.
    geometry_only = schema.attribute_source == "od"

    class_columns: dict[str, list[str]] = {}
    missing: list[str] = []
    if schema.polygon_id_column not in raw.columns:
        missing.append(schema.polygon_id_column)
    if not geometry_only:
        if schema.population_column not in raw.columns:
            missing.append(schema.population_column)
        for attr in schema.attributes:
            cols = class_columns_for(attr)
            class_columns[attr.name] = cols
            for c in cols:
                if c not in raw.columns:
                    missing.append(c)
    if missing:
        raise SchemaError(
            f"demographics file {path.name} is missing required columns "
            f"per the demographic schema: {sorted(set(missing))}. "
            f"Available columns: {sorted(raw.columns)}",
            details={"missing": sorted(set(missing)), "available": sorted(map(str, raw.columns))},
        )

    gdf = reproject_to(raw, working_crs, assume_if_missing=settings.crs.geographic_crs).copy()
    rename = {schema.polygon_id_column: "id"}
    if schema.population_column in gdf.columns:
        rename[schema.population_column] = "pop"
    gdf = gdf.rename(columns=rename)
    gdf["id"] = gdf["id"].astype(str)
    if "pop" in gdf.columns:
        gdf["pop"] = gdf["pop"].astype(float)
    else:
        # geometry-only lookup: population is unknown here (it comes from the OD
        # file). Default to 1.0 so the column contract holds downstream.
        gdf["pop"] = 1.0
    for cols in class_columns.values():
        for c in cols:
            gdf[c] = gdf[c].astype(float)

    types = set(gdf.geometry.geom_type)
    polygons = {"Polygon", "MultiPolygon"}
    points = {"Point", "MultiPoint"}
    if types <= polygons:
        kind: GeometryKind = "polygon"
    elif types <= points:
        kind = "point"
    else:
        raise GeometryError(
            f"demographics file contains mixed/unsupported geometry types {sorted(types)} — "
            "expected all polygons OR all points",
            details={"types": sorted(types)},
        )

    return LoadedDemographics(gdf=gdf, kind=kind, class_columns=class_columns)


__all__ = ["GeometryKind", "LoadedDemographics", "class_columns_for", "load_demographics"]
