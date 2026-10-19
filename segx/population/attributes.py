"""Attach a demographic vector to each origin (P2).

Every origin gets three columns:

* ``demo_vector`` — ``dict[attribute_name, dict[class_label, float]]``: the
  per-class mass for each attribute declared in ``settings.demographics``.
  Categorical/binned classes use the attribute's ``class_labels``; the values
  are raw counts (not shares) so the aggregator can weight them directly.
* ``demo_resolved`` — ``bool``: ``False`` when no demographics could be found
  for the origin (e.g. an OD origin outside every known polygon). Per invariant
  #2 these are **flagged, never silently zeroed** — the vector is filled with
  ``NaN``, which is distinct from a real zero ("all mass in one class").
* ``demo_reason`` — ``str | None``: the reason code when unresolved.

Two attribute sources (``settings.demographics.attribute_source``):

* ``"demographics"`` (default) — read each origin's per-class columns from the
  demographics row with the matching ``id``. Polygon-centroid and residence-point
  origins match by construction; OD origins join by ``origin_id``.
* ``"od"`` — the per-class columns live on the **OD file** (LODES-style
  ``SE01``/``SE02``/``SE03`` worker counts). Sum them per distinct origin.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd

from segx.errors import ConfigValueError, SchemaError
from segx.io.demographics import class_columns_for

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings
    from segx.io.demographics import LoadedDemographics
    from segx.io.od_pairs import LoadedODPairs

#: Reason code stamped on origins for which no demographics could be resolved.
NO_DEMOGRAPHICS: str = "no_demographics"


def _nan_vector(settings: Settings) -> dict[str, dict[str, float]]:
    """A fully-``NaN`` demographic vector (shape matches the schema)."""
    return {a.name: {label: math.nan for label in a.class_labels} for a in settings.demographics.attributes}


def _vector_from_row(settings: Settings, values: dict[str, float]) -> dict[str, dict[str, float]]:
    """Build a demographic vector from a flat ``column -> value`` mapping."""
    vector: dict[str, dict[str, float]] = {}
    for attr in settings.demographics.attributes:
        cols = class_columns_for(attr)
        vector[attr.name] = {label: float(values[col]) for label, col in zip(attr.class_labels, cols, strict=True)}
    return vector


def _required_columns(settings: Settings) -> list[str]:
    cols: list[str] = []
    for attr in settings.demographics.attributes:
        cols.extend(class_columns_for(attr))
    return cols


def attach_attributes(
    origins: gpd.GeoDataFrame,
    demographics: LoadedDemographics,
    od_pairs: LoadedODPairs | None,
    settings: Settings,
) -> gpd.GeoDataFrame:
    """Return ``origins`` with ``demo_vector`` / ``demo_resolved`` / ``demo_reason``."""
    source = settings.demographics.attribute_source
    if source == "od":
        per_origin = _od_class_sums(origins, od_pairs, settings)
    else:
        per_origin = _demographics_lookup(demographics, settings)

    vectors: list[dict[str, dict[str, float]]] = []
    resolved: list[bool] = []
    reasons: list[str | None] = []
    for origin_id in origins["origin_id"].astype(str):
        values = per_origin.get(origin_id)
        if values is None:
            vectors.append(_nan_vector(settings))
            resolved.append(False)
            reasons.append(NO_DEMOGRAPHICS)
        else:
            vectors.append(_vector_from_row(settings, values))
            resolved.append(True)
            reasons.append(None)

    out = origins.copy()
    out["demo_vector"] = vectors
    out["demo_resolved"] = resolved
    out["demo_reason"] = reasons
    return out


def _demographics_lookup(demographics: LoadedDemographics, settings: Settings) -> dict[str, dict[str, float]]:
    """``origin_id -> {column: value}`` from the demographics file's class columns."""
    required = _required_columns(settings)
    g = demographics.gdf
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise SchemaError(
            f"attribute_source='demographics' but the demographics file lacks per-class "
            f"columns {sorted(set(missing))}. Available: {sorted(map(str, g.columns))}",
            details={"missing": sorted(set(missing))},
        )
    frame = g.drop_duplicates(subset="id").copy()
    frame["id"] = frame["id"].astype(str)
    frame = frame.set_index("id")
    return {str(idx): {c: float(frame.at[idx, c]) for c in required} for idx in frame.index}


def _od_class_sums(
    origins: gpd.GeoDataFrame,
    od_pairs: LoadedODPairs | None,
    settings: Settings,
) -> dict[str, dict[str, float]]:
    """``origin_id -> {column: summed value}`` from the OD file's class columns."""
    if od_pairs is None:
        raise ConfigValueError(
            "demographics.attribute_source='od' requires an OD-pairs file to source "
            "per-class counts from, but none was supplied.",
            details={"attribute_source": "od"},
        )
    required = _required_columns(settings)
    df = od_pairs.gdf
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"attribute_source='od' but the OD file lacks per-class columns "
            f"{sorted(set(missing))}. Declare them via each attribute's `class_columns`. "
            f"Available: {sorted(map(str, df.columns))}",
            details={"missing": sorted(set(missing))},
        )
    sums = df.groupby(df["origin_id"].astype(str))[required].sum()
    return {str(idx): {c: float(sums.at[idx, c]) for c in required} for idx in sums.index}


__all__ = ["NO_DEMOGRAPHICS", "attach_attributes"]
