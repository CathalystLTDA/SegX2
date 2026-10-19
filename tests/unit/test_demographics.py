"""P1: demographics loader — schema validation, reprojection, geometry kind."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from segx.config import DemographicAttribute, DemographicSchema, Settings
from segx.io.demographics import class_columns_for, load_demographics


def _tiny_schema() -> DemographicSchema:
    """A small schema so test fixtures stay readable."""
    return DemographicSchema(
        attributes=[
            DemographicAttribute(name="race", kind="categorical", categories=["a", "b"]),
            DemographicAttribute(name="income", kind="binned", bins=[0.0, 100.0, 200.0]),
        ],
    )


def test_class_columns_for_categorical_and_binned() -> None:
    schema = _tiny_schema()
    assert class_columns_for(schema.get("race")) == ["race_a", "race_b"]
    assert class_columns_for(schema.get("income")) == ["income_b0", "income_b1"]


def test_class_columns_for_numeric() -> None:
    attr = DemographicAttribute(name="density", kind="numeric")
    assert class_columns_for(attr) == ["density"]


def test_class_columns_slugifies_categories() -> None:
    attr = DemographicAttribute(name="race", kind="categorical", categories=["sem decl.", "indígena"])
    cols = class_columns_for(attr)
    # non-alphanumeric chars collapse to "_"; this is the contract column-name shape
    assert cols == ["race_sem_decl", "race_ind_gena"]


def _write_demographics(path, polygon, *, crs="EPSG:4326", **extra_cols) -> None:
    data = {"id": ["sector-1"], "pop": [100.0], **{k: [v] for k, v in extra_cols.items()}}
    gpd.GeoDataFrame(data, geometry=[polygon], crs=crs).to_file(path, driver="GPKG")


def test_load_demographics_polygons(tmp_path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    f = tmp_path / "sectors.gpkg"
    _write_demographics(
        f,
        poly,
        race_a=60.0,
        race_b=40.0,
        income_b0=30.0,
        income_b1=70.0,
    )

    s = Settings(demographics=_tiny_schema())
    loaded = load_demographics(f, s, working_crs="EPSG:32722")

    assert loaded.kind == "polygon"
    assert list(loaded.gdf["id"]) == ["sector-1"]
    assert loaded.gdf["pop"].iloc[0] == 100.0
    assert loaded.gdf.crs.to_epsg() == 32722
    assert loaded.class_columns == {"race": ["race_a", "race_b"], "income": ["income_b0", "income_b1"]}
    # the class count columns survive and are float
    assert loaded.gdf["race_a"].dtype.kind == "f"
    assert loaded.gdf["income_b1"].iloc[0] == 70.0


def test_load_demographics_residence_points(tmp_path) -> None:
    pts = [Point(-51.20, -30.00), Point(-51.18, -30.02)]
    gdf = gpd.GeoDataFrame(
        {
            "id": ["r1", "r2"],
            "pop": [1.0, 1.0],
            "race_a": [1.0, 0.0],
            "race_b": [0.0, 1.0],
            "income_b0": [1.0, 0.0],
            "income_b1": [0.0, 1.0],
        },
        geometry=pts,
        crs="EPSG:4326",
    )
    f = tmp_path / "residences.gpkg"
    gdf.to_file(f, driver="GPKG")

    s = Settings(demographics=_tiny_schema())
    loaded = load_demographics(f, s, working_crs="EPSG:32722")
    assert loaded.kind == "point"
    assert len(loaded.gdf) == 2


def test_load_demographics_missing_columns_lists_them(tmp_path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    f = tmp_path / "broken.gpkg"
    # missing income_b1 and race_b
    _write_demographics(f, poly, race_a=10.0, income_b0=5.0)
    s = Settings(demographics=_tiny_schema())
    with pytest.raises(ValueError, match="missing required columns"):
        load_demographics(f, s, working_crs="EPSG:32722")


def test_load_demographics_mixed_geometry_rejected(tmp_path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    pt = Point(-51.20, -30.00)
    gdf = gpd.GeoDataFrame(
        {
            "id": ["a", "b"],
            "pop": [10.0, 1.0],
            "race_a": [1.0, 1.0],
            "race_b": [1.0, 0.0],
            "income_b0": [1.0, 1.0],
            "income_b1": [1.0, 0.0],
        },
        geometry=[poly, pt],
        crs="EPSG:4326",
    )
    f = tmp_path / "mixed.gpkg"
    gdf.to_file(f, driver="GPKG")
    s = Settings(demographics=_tiny_schema())
    with pytest.raises(ValueError, match="mixed/unsupported geometry"):
        load_demographics(f, s, working_crs="EPSG:32722")


def test_load_demographics_custom_id_and_pop_columns(tmp_path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    gdf = gpd.GeoDataFrame(
        {
            "COD_SETOR": ["X-1"],
            "POPULACAO": [123],
            "race_a": [70.0],
            "race_b": [53.0],
            "income_b0": [50.0],
            "income_b1": [73.0],
        },
        geometry=[poly],
        crs="EPSG:4326",
    )
    f = tmp_path / "custom.gpkg"
    gdf.to_file(f, driver="GPKG")

    schema = _tiny_schema()
    schema.polygon_id_column = "COD_SETOR"
    schema.population_column = "POPULACAO"
    s = Settings(demographics=schema)
    loaded = load_demographics(f, s, working_crs="EPSG:32722")
    assert list(loaded.gdf["id"]) == ["X-1"]
    assert loaded.gdf["pop"].iloc[0] == 123.0
