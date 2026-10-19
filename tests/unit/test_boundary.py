"""P1: boundary loader + working-CRS resolution."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from segx.config import CRSConfig, Settings
from segx.io._crs import resolve_working_crs, utm_epsg_for
from segx.io.boundary import load_boundary

# --------------------------------------------------------------------------- #
# UTM derivation                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (-51.2, -30.0, "EPSG:32722"),  # Porto Alegre → UTM 22 South
        (-46.6, -23.5, "EPSG:32723"),  # São Paulo    → UTM 23 South
        (-43.2, -22.9, "EPSG:32723"),  # Rio          → UTM 23 South
        (-74.0, 40.7, "EPSG:32618"),  # New York     → UTM 18 North
        (2.35, 48.85, "EPSG:32631"),  # Paris        → UTM 31 North
    ],
)
def test_utm_epsg_for_known_cities(lon: float, lat: float, expected: str) -> None:
    assert utm_epsg_for(lon, lat) == expected


def test_utm_rejects_out_of_range_lon() -> None:
    with pytest.raises(ValueError, match="longitude"):
        utm_epsg_for(200.0, 0.0)


def test_utm_rejects_polar_lat() -> None:
    with pytest.raises(ValueError, match="latitude"):
        utm_epsg_for(0.0, 88.0)


# --------------------------------------------------------------------------- #
# Working CRS resolution                                                       #
# --------------------------------------------------------------------------- #


def test_resolve_working_crs_explicit_override_wins() -> None:
    s = Settings(crs=CRSConfig(working_crs="EPSG:31984"))
    # even with auto_utm on, the explicit override is the choice
    assert resolve_working_crs((-51.2, -30.0), s) == "EPSG:31984"


def test_resolve_working_crs_auto_default() -> None:
    s = Settings()
    assert s.crs.auto_utm is True
    assert resolve_working_crs((-51.2, -30.0), s) == "EPSG:32722"


def test_resolve_working_crs_no_auto_no_override_raises() -> None:
    s = Settings(crs=CRSConfig(auto_utm=False, working_crs=None))
    with pytest.raises(ValueError, match="working CRS"):
        resolve_working_crs((-51.2, -30.0), s)


# --------------------------------------------------------------------------- #
# load_boundary                                                                #
# --------------------------------------------------------------------------- #


def _write_polygon_gpkg(path, polygon: Polygon, crs: str = "EPSG:4326") -> None:
    gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=crs).to_file(path, driver="GPKG")


def test_load_boundary_reprojects_to_auto_utm(tmp_path) -> None:
    # tiny square around Porto Alegre
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    f = tmp_path / "boundary.gpkg"
    _write_polygon_gpkg(f, poly)

    loaded = load_boundary(f, Settings())
    assert loaded.crs == "EPSG:32722"

    # bbox preserved in WGS84 to feed OSM fetches
    minx, miny, maxx, maxy = loaded.bbox_wgs84
    assert minx == pytest.approx(-51.25, abs=1e-9)
    assert miny == pytest.approx(-30.05, abs=1e-9)
    assert maxx == pytest.approx(-51.15, abs=1e-9)
    assert maxy == pytest.approx(-29.95, abs=1e-9)

    # projected geometry: metre-scale, Porto Alegre easting ~470_000 m
    pminx, pminy, pmaxx, pmaxy = loaded.geometry.bounds
    assert 400_000 < pminx < 600_000
    assert 6_000_000 < pminy < 7_500_000


def test_load_boundary_dissolves_multiple_features(tmp_path) -> None:
    a = Polygon([(-51.25, -30.05), (-51.20, -30.05), (-51.20, -30.00), (-51.25, -30.00)])
    b = Polygon([(-51.20, -30.05), (-51.15, -30.05), (-51.15, -30.00), (-51.20, -30.00)])
    f = tmp_path / "two_features.gpkg"
    gpd.GeoDataFrame({"id": [1, 2]}, geometry=[a, b], crs="EPSG:4326").to_file(f, driver="GPKG")
    loaded = load_boundary(f, Settings())
    # the two adjacent rectangles dissolve into a single polygon
    assert loaded.geometry.geom_type == "Polygon"


def test_load_boundary_assumes_wgs84_when_crs_missing(tmp_path) -> None:
    # GeoJSON without a CRS embedded → assume EPSG:4326 (per CRSConfig.geographic_crs)
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    f = tmp_path / "no_crs.geojson"
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly]).to_file(f, driver="GeoJSON")
    loaded = load_boundary(f, Settings())
    assert loaded.crs == "EPSG:32722"


def test_load_boundary_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_boundary(tmp_path / "nope.gpkg", Settings())


def test_load_boundary_honours_explicit_working_crs(tmp_path) -> None:
    # use SIRGAS 2000 / UTM 22S (EPSG:31982) as an explicit override
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    f = tmp_path / "boundary.gpkg"
    _write_polygon_gpkg(f, poly)
    s = Settings(crs=CRSConfig(working_crs="EPSG:31982"))
    loaded = load_boundary(f, s)
    assert loaded.crs == "EPSG:31982"
