"""P1: OD-pairs loader — column-mapping resolution, defaults, validation."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from segx.config import ODColumnMapping, Settings
from segx.io.od_pairs import load_od_pairs, origin_points


def _write_csv(path, **cols) -> None:
    pd.DataFrame(cols).to_csv(path, index=False)


def test_load_od_pairs_csv_with_lon_lat_defaults(tmp_path) -> None:
    f = tmp_path / "od.csv"
    _write_csv(
        f,
        origin_id=["a", "b"],
        origin_lon=[-51.20, -51.18],
        origin_lat=[-30.00, -30.02],
        dest_lon=[-51.10, -51.12],
        dest_lat=[-30.05, -30.04],
        category=["commute", "school"],
        weight=[2.0, 1.5],
    )
    loaded = load_od_pairs(f, Settings(), working_crs="EPSG:32722")
    g = loaded.gdf
    assert list(g["origin_id"]) == ["a", "b"]
    assert list(g["category"]) == ["commute", "school"]
    # default mode_for: commute -> car, school -> walk
    assert list(g["mode"]) == ["car", "walk"]
    assert list(g["weight"]) == [2.0, 1.5]
    # reprojected into UTM 22S (Porto Alegre): metre-scale coordinates
    assert g.crs.to_epsg() == 32722
    assert all(p.geom_type == "Point" for p in g.geometry)
    assert all(g.geometry.x > 400_000)
    # destination also reprojected
    dests = gpd.GeoSeries(g["dest_geom"].to_list(), crs="EPSG:32722")
    assert all(dests.geom_type == "Point")
    assert all(dests.x > 400_000)


def test_load_od_pairs_csv_missing_columns_use_defaults(tmp_path) -> None:
    # only the bare minimum: lon/lat for both endpoints; everything else defaults
    f = tmp_path / "minimal.csv"
    _write_csv(
        f,
        origin_lon=[-51.20],
        origin_lat=[-30.00],
        dest_lon=[-51.10],
        dest_lat=[-30.05],
    )
    loaded = load_od_pairs(f, Settings(), working_crs="EPSG:32722")
    g = loaded.gdf
    # origin_id auto-generated row index → "0"
    assert list(g["origin_id"]) == ["0"]
    # category falls back to default_category = "commute"
    assert list(g["category"]) == ["commute"]
    # mode follows default_mode_by_category for "commute" → "car"
    assert list(g["mode"]) == ["car"]
    # weight falls back to default_weight = 1.0
    assert list(g["weight"]) == [1.0]


def test_load_od_pairs_csv_with_wkt(tmp_path) -> None:
    f = tmp_path / "wkt.csv"
    _write_csv(
        f,
        origin_id=["x1"],
        origin_wkt=["POINT (-51.20 -30.00)"],
        dest_wkt=["POINT (-51.10 -30.05)"],
    )
    loaded = load_od_pairs(f, Settings(), working_crs="EPSG:32722")
    assert list(loaded.gdf["origin_id"]) == ["x1"]
    assert loaded.gdf.crs.to_epsg() == 32722


def test_load_od_pairs_gpkg_origin_as_geometry(tmp_path) -> None:
    # GPKG with origin in the active geometry column, dest via lon+lat
    gdf = gpd.GeoDataFrame(
        {
            "origin_id": ["g1", "g2"],
            "dest_lon": [-51.10, -51.12],
            "dest_lat": [-30.05, -30.04],
            "category": ["commute", "shopping"],
        },
        geometry=[Point(-51.20, -30.00), Point(-51.18, -30.02)],
        crs="EPSG:4326",
    )
    f = tmp_path / "od.gpkg"
    gdf.to_file(f, driver="GPKG")
    # default mapping uses origin_geom = "geometry" by default? no — default is "origin_geom"
    # so we need to override origin_geom to the active geometry column name
    s = Settings(od_columns=ODColumnMapping(origin_geom="geometry"))
    loaded = load_od_pairs(f, s, working_crs="EPSG:32722")
    assert list(loaded.gdf["origin_id"]) == ["g1", "g2"]
    assert loaded.gdf.crs.to_epsg() == 32722


def test_load_od_pairs_address_only_raises(tmp_path) -> None:
    f = tmp_path / "addr.csv"
    _write_csv(
        f,
        origin_address=["Rua A, 100"],
        dest_lon=[-51.10],
        dest_lat=[-30.05],
    )
    with pytest.raises(NotImplementedError, match="geocoding"):
        load_od_pairs(f, Settings(), working_crs="EPSG:32722")


def test_load_od_pairs_missing_endpoint_columns_raises(tmp_path) -> None:
    f = tmp_path / "broken.csv"
    # no origin location columns at all
    _write_csv(f, dest_lon=[-51.10], dest_lat=[-30.05])
    with pytest.raises(ValueError, match="could not resolve origin"):
        load_od_pairs(f, Settings(), working_crs="EPSG:32722")


def test_load_od_pairs_custom_column_mapping(tmp_path) -> None:
    f = tmp_path / "custom.csv"
    _write_csv(
        f,
        COD_ORIGEM=["A1"],
        ox=[-51.20],
        oy=[-30.00],
        dx=[-51.10],
        dy=[-30.05],
        TIPO=["trabalho"],
        PESO=[3.5],
    )
    mapping = ODColumnMapping(
        origin_id="COD_ORIGEM",
        origin_lon="ox",
        origin_lat="oy",
        dest_lon="dx",
        dest_lat="dy",
        category="TIPO",
        weight="PESO",
        default_mode_by_category={"trabalho": "car"},
    )
    s = Settings(od_columns=mapping)
    loaded = load_od_pairs(f, s, working_crs="EPSG:32722")
    assert list(loaded.gdf["origin_id"]) == ["A1"]
    assert list(loaded.gdf["category"]) == ["trabalho"]
    assert list(loaded.gdf["mode"]) == ["car"]
    assert list(loaded.gdf["weight"]) == [3.5]


def test_origin_points_dedupes(tmp_path) -> None:
    # same origin_id appears twice — origin_points returns one row per id
    f = tmp_path / "dup.csv"
    _write_csv(
        f,
        origin_id=["a", "a", "b"],
        origin_lon=[-51.20, -51.20, -51.18],
        origin_lat=[-30.00, -30.00, -30.02],
        dest_lon=[-51.10, -51.11, -51.12],
        dest_lat=[-30.05, -30.06, -30.04],
    )
    loaded = load_od_pairs(f, Settings(), working_crs="EPSG:32722")
    origins = origin_points(loaded)
    assert sorted(origins["origin_id"].tolist()) == ["a", "b"]
    assert origins.crs.to_epsg() == 32722


def test_load_od_pairs_empty_file(tmp_path) -> None:
    f = tmp_path / "empty.csv"
    pd.DataFrame({"origin_lon": [], "origin_lat": [], "dest_lon": [], "dest_lat": []}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_od_pairs(f, Settings(), working_crs="EPSG:32722")


def test_none_skips_a_strategy_even_if_its_default_column_exists(tmp_path) -> None:
    # The file has BOTH a WKT column (default name, would win) and lon/lat columns.
    # A host mapping that chose lon/lat sets the WKT field to None, so lon/lat is used.
    f = tmp_path / "both.csv"
    _write_csv(
        f,
        origin_wkt=["POINT (-51.30 -30.10)"],
        origin_lon=[-51.20],
        origin_lat=[-30.00],
        dest_lon=[-51.10],
        dest_lat=[-30.05],
    )
    by_default = load_od_pairs(f, Settings(), working_crs="EPSG:4326")
    assert by_default.gdf.geometry.x.iloc[0] == pytest.approx(-51.30)  # WKT wins by default

    settings = Settings(od_columns=ODColumnMapping(origin_wkt=None, dest_wkt=None))
    explicit = load_od_pairs(f, settings, working_crs="EPSG:4326")
    assert explicit.gdf.geometry.x.iloc[0] == pytest.approx(-51.20)


def test_none_optional_columns_fall_back_to_defaults(tmp_path) -> None:
    f = tmp_path / "od.csv"
    _write_csv(
        f,
        origin_lon=[-51.20],
        origin_lat=[-30.00],
        dest_lon=[-51.10],
        dest_lat=[-30.05],
        weight=[7.0],
        mode=["walk"],
    )
    settings = Settings(od_columns=ODColumnMapping(weight=None, mode=None, default_weight=1.0))
    g = load_od_pairs(f, settings, working_crs="EPSG:32722").gdf
    assert list(g["weight"]) == [1.0]  # the "weight" column is ignored
    assert list(g["mode"]) == ["car"]  # default_mode_by_category, not the "mode" column


def test_unresolvable_endpoint_lists_only_the_mapped_columns(tmp_path) -> None:
    f = tmp_path / "od.csv"
    _write_csv(f, a=[1], b=[2])
    settings = Settings(
        od_columns=ODColumnMapping(origin_geom=None, origin_wkt=None, origin_id_ref=None, origin_address=None)
    )
    with pytest.raises(ValueError, match="could not resolve origin") as exc:
        load_od_pairs(f, settings, working_crs="EPSG:32722")
    assert exc.value.details["tried"] == ["origin_lon", "origin_lat"]
