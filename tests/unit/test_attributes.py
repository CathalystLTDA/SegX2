"""P2: attribute attachment — demographics source, OD (LODES) source, flagging."""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from segx.config import DemographicAttribute, DemographicSchema, ODColumnMapping, Settings
from segx.io.demographics import load_demographics
from segx.io.od_pairs import load_od_pairs
from segx.population.attributes import NO_DEMOGRAPHICS, attach_attributes
from segx.population.origins import build_origins

WORKING_CRS = "EPSG:32722"


def _square(cx: float, cy: float, half: float = 0.01) -> Polygon:
    return Polygon([(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])


# --------------------------------------------------------------------------- #
# attribute_source = "demographics"                                           #
# --------------------------------------------------------------------------- #


def _demographics_schema() -> DemographicSchema:
    return DemographicSchema(
        attributes=[
            DemographicAttribute(name="race", kind="categorical", categories=["a", "b"]),
            DemographicAttribute(name="income", kind="binned", bins=[0.0, 100.0, 200.0]),
        ],
    )


def test_attach_from_demographics_polygons(tmp_path) -> None:
    f = tmp_path / "demo.gpkg"
    gpd.GeoDataFrame(
        {
            "id": ["s1", "s2"],
            "pop": [100.0, 50.0],
            "race_a": [40.0, 5.0],
            "race_b": [60.0, 45.0],
            "income_b0": [30.0, 10.0],
            "income_b1": [70.0, 40.0],
        },
        geometry=[_square(-51.20, -30.00), _square(-51.10, -30.05)],
        crs="EPSG:4326",
    ).to_file(f, driver="GPKG")

    s = Settings(demographics=_demographics_schema())
    demo = load_demographics(f, s, working_crs=WORKING_CRS)
    origins = build_origins(demo, None, s)
    attributed = attach_attributes(origins, demo, None, s).set_index("origin_id")

    assert bool(attributed.loc["s1", "demo_resolved"]) is True
    assert attributed.loc["s1", "demo_vector"] == {
        "race": {"a": 40.0, "b": 60.0},
        "income": {"[0, 100)": 30.0, "[100, 200)": 70.0},
    }
    assert attributed.loc["s2", "demo_vector"]["race"] == {"a": 5.0, "b": 45.0}


def test_attach_flags_unresolved_origin(tmp_path) -> None:
    # demographics knows block "s1"; OD origin "ghost" matches nothing
    demo_file = tmp_path / "demo.gpkg"
    gpd.GeoDataFrame(
        {"id": ["s1"], "pop": [1.0], "race_a": [1.0], "race_b": [1.0], "income_b0": [1.0], "income_b1": [1.0]},
        geometry=[_square(-51.20, -30.00)],
        crs="EPSG:4326",
    ).to_file(demo_file, driver="GPKG")
    od_file = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "origin_id": ["ghost"],
            "origin_lon": [-51.30],
            "origin_lat": [-30.10],
            "dest_lon": [-51.1],
            "dest_lat": [-30.0],
        }
    ).to_csv(od_file, index=False)

    s = Settings(demographics=_demographics_schema())
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    od = load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=demo)
    origins = build_origins(demo, od, s)
    attributed = attach_attributes(origins, demo, od, s).set_index("origin_id")

    assert bool(attributed.loc["ghost", "demo_resolved"]) is False
    assert attributed.loc["ghost", "demo_reason"] == NO_DEMOGRAPHICS
    # unresolved is NaN, NOT a real zero (invariant #2)
    assert math.isnan(attributed.loc["ghost", "demo_vector"]["race"]["a"])


# --------------------------------------------------------------------------- #
# attribute_source = "od"  (LODES-style)                                       #
# --------------------------------------------------------------------------- #


def _lodes_settings() -> Settings:
    schema = DemographicSchema(
        attribute_source="od",
        attributes=[
            DemographicAttribute(
                name="income",
                kind="categorical",
                categories=["low", "middle", "high"],
                class_columns=["SE01", "SE02", "SE03"],
            )
        ],
    )
    return Settings(demographics=schema, od_columns=ODColumnMapping(origin_id_ref="hGEOID", dest_id_ref="wGEOID"))


def test_attach_from_od_lodes_sums_per_origin(tmp_path) -> None:
    # census-block geometry, id only (geometry-only lookup)
    demo_file = tmp_path / "blocks.gpkg"
    gpd.GeoDataFrame(
        {"GEOID": ["100", "200", "300"]},
        geometry=[_square(-51.20, -30.00), _square(-51.10, -30.05), _square(-51.05, -30.08)],
        crs="EPSG:4326",
    ).to_file(demo_file, driver="GPKG")

    od_file = tmp_path / "lodes.csv"
    pd.DataFrame(
        {
            "hGEOID": ["100", "100", "200"],
            "wGEOID": ["200", "300", "300"],
            "SE01": [5, 3, 1],
            "SE02": [2, 1, 4],
            "SE03": [0, 1, 2],
        }
    ).to_csv(od_file, index=False)

    s = _lodes_settings()
    s.demographics.polygon_id_column = "GEOID"
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    assert demo.kind == "polygon"

    od = load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=demo)
    origins = build_origins(demo, od, s)
    attributed = attach_attributes(origins, demo, od, s).set_index("origin_id")

    # block 100 contributes two flows: SE summed → low=8, middle=3, high=1
    assert attributed.loc["100", "demo_vector"] == {"income": {"low": 8.0, "middle": 3.0, "high": 1.0}}
    # block 200 has one home-flow row: low=1, middle=4, high=2
    assert attributed.loc["200", "demo_vector"] == {"income": {"low": 1.0, "middle": 4.0, "high": 2.0}}
    assert bool(attributed["demo_resolved"].all())


def test_attach_od_source_without_od_pairs_raises(tmp_path) -> None:
    demo_file = tmp_path / "blocks.gpkg"
    gpd.GeoDataFrame({"GEOID": ["100"]}, geometry=[_square(-51.20, -30.00)], crs="EPSG:4326").to_file(
        demo_file, driver="GPKG"
    )
    s = _lodes_settings()
    s.demographics.polygon_id_column = "GEOID"
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    origins = build_origins(demo, None, s)
    with pytest.raises(ValueError, match="requires an OD-pairs file"):
        attach_attributes(origins, demo, None, s)
