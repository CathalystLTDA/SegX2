"""P2: origin-source branch — OD origins, polygon centroids, point passthrough."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from segx.config import DemographicAttribute, DemographicSchema, ODColumnMapping, Settings
from segx.io.demographics import load_demographics
from segx.io.od_pairs import load_od_pairs
from segx.population.origins import build_origins

WORKING_CRS = "EPSG:32722"


def _tiny_schema(**kw) -> DemographicSchema:
    return DemographicSchema(
        attributes=[DemographicAttribute(name="race", kind="categorical", categories=["a", "b"])],
        **kw,
    )


def _square(cx: float, cy: float, half: float = 0.01) -> Polygon:
    return Polygon([(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)])


def _write_demographics(path, *, geometries, ids, **cols) -> None:
    data = {"id": ids, "pop": [1.0] * len(ids), **cols}
    gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326").to_file(path, driver="GPKG")


def test_polygon_centroid_origins(tmp_path) -> None:
    f = tmp_path / "demo.gpkg"
    _write_demographics(
        f,
        geometries=[_square(-51.20, -30.00), _square(-51.10, -30.05)],
        ids=["s1", "s2"],
        race_a=[10.0, 0.0],
        race_b=[0.0, 5.0],
    )
    s = Settings(demographics=_tiny_schema())
    demo = load_demographics(f, s, working_crs=WORKING_CRS)

    origins = build_origins(demo, None, s)
    assert sorted(origins["origin_id"]) == ["s1", "s2"]
    assert origins.crs.to_epsg() == 32722
    assert all(origins.geometry.geom_type == "Point")
    assert list(origins["weight"]) == [1.0, 1.0]


def test_residence_point_passthrough(tmp_path) -> None:
    f = tmp_path / "res.gpkg"
    _write_demographics(
        f,
        geometries=[Point(-51.20, -30.00), Point(-51.18, -30.02)],
        ids=["r1", "r2"],
        race_a=[1.0, 0.0],
        race_b=[0.0, 1.0],
    )
    s = Settings(demographics=_tiny_schema())
    demo = load_demographics(f, s, working_crs=WORKING_CRS)
    assert demo.kind == "point"

    origins = build_origins(demo, None, s)
    assert list(origins["origin_id"]) == ["r1", "r2"]
    # points pass through unchanged (same coordinates, reprojected)
    reproj = demo.gdf.geometry.reset_index(drop=True)
    assert all(origins.geometry.reset_index(drop=True).geom_equals_exact(reproj, tolerance=1e-6))


def test_od_origins_take_precedence_over_demographics(tmp_path) -> None:
    # demographics present, but OD pairs supplied → origins are the OD origins
    demo_file = tmp_path / "demo.gpkg"
    _write_demographics(
        demo_file,
        geometries=[_square(-51.20, -30.00)],
        ids=["s1"],
        race_a=[10.0],
        race_b=[0.0],
    )
    od_file = tmp_path / "od.csv"
    pd.DataFrame(
        {
            "origin_id": ["A", "A", "B"],
            "origin_lon": [-51.30, -51.30, -51.40],
            "origin_lat": [-30.10, -30.10, -30.20],
            "dest_lon": [-51.10, -51.11, -51.12],
            "dest_lat": [-30.05, -30.06, -30.04],
        }
    ).to_csv(od_file, index=False)

    s = Settings(demographics=_tiny_schema())
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    od = load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=demo)

    origins = build_origins(demo, od, s)
    # distinct OD origins only — "s1" from demographics is NOT synthesised
    assert sorted(origins["origin_id"]) == ["A", "B"]


def test_od_origins_by_id_reference_resolve_to_block_geometry(tmp_path) -> None:
    # LODES-style: hGEOID/wGEOID reference census-block geometry by id
    demo_file = tmp_path / "blocks.gpkg"
    _write_demographics(
        demo_file,
        geometries=[_square(-51.20, -30.00), _square(-51.10, -30.05), _square(-51.05, -30.08)],
        ids=["100", "200", "300"],
        race_a=[1.0, 1.0, 1.0],
        race_b=[1.0, 1.0, 1.0],
    )
    od_file = tmp_path / "lodes.csv"
    pd.DataFrame(
        {
            "hGEOID": ["100", "100", "200"],
            "wGEOID": ["200", "300", "300"],
            "SE01": [5, 2, 1],
        }
    ).to_csv(od_file, index=False)

    mapping = ODColumnMapping(origin_id_ref="hGEOID", dest_id_ref="wGEOID")
    s = Settings(demographics=_tiny_schema(), od_columns=mapping)
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    od = load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=demo)

    origins = build_origins(demo, od, s)
    # two distinct home blocks; their geometry is the block's representative point
    assert sorted(origins["origin_id"]) == ["100", "200"]
    block_100 = demo.gdf.set_index("id").loc["100"].geometry.representative_point()
    o100 = origins.set_index("origin_id").loc["100"].geometry
    assert o100.distance(block_100) < 1.0  # metres, same point


def test_od_by_id_reference_without_demographics_raises(tmp_path) -> None:
    od_file = tmp_path / "lodes.csv"
    pd.DataFrame({"hGEOID": ["100"], "wGEOID": ["200"], "SE01": [5]}).to_csv(od_file, index=False)
    mapping = ODColumnMapping(origin_id_ref="hGEOID", dest_id_ref="wGEOID")
    s = Settings(demographics=_tiny_schema(), od_columns=mapping)
    with pytest.raises(ValueError, match="id-reference"):
        load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=None)


def test_od_by_id_reference_unknown_id_raises(tmp_path) -> None:
    demo_file = tmp_path / "blocks.gpkg"
    _write_demographics(
        demo_file,
        geometries=[_square(-51.20, -30.00)],
        ids=["100"],
        race_a=[1.0],
        race_b=[1.0],
    )
    od_file = tmp_path / "lodes.csv"
    # wGEOID 999 has no matching block geometry
    pd.DataFrame({"hGEOID": ["100"], "wGEOID": ["999"], "SE01": [5]}).to_csv(od_file, index=False)
    mapping = ODColumnMapping(origin_id_ref="hGEOID", dest_id_ref="wGEOID")
    s = Settings(demographics=_tiny_schema(), od_columns=mapping)
    demo = load_demographics(demo_file, s, working_crs=WORKING_CRS)
    with pytest.raises(ValueError, match="match no demographics geometry id"):
        load_od_pairs(od_file, s, working_crs=WORKING_CRS, demographics=demo)
