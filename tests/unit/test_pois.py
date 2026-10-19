"""P1: POI normalisation + cache key + cache-hit path. Live fetchers
(OSMnx Overpass, QuackOSM) are integration territory."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from segx.config import Settings
from segx.io.boundary import LoadedBoundary
from segx.io.pois import (
    LoadedPOIs,
    build_pois_gdf,
    load_pois,
    pois_cache_key,
    pois_cache_path,
)

# --------------------------------------------------------------------------- #
# cache key                                                                    #
# --------------------------------------------------------------------------- #


def test_pois_cache_key_deterministic() -> None:
    bbox = (-51.25, -30.05, -51.15, -29.95)
    k = pois_cache_key(bbox, "2026-05-01", ["parks", "bus_stops"])
    # category order doesn't change the key
    assert pois_cache_key(bbox, "2026-05-01", ["bus_stops", "parks"]) == k
    # adding a category does
    assert pois_cache_key(bbox, "2026-05-01", ["parks"]) != k
    # date change does
    assert pois_cache_key(bbox, "2026-06-01", ["parks", "bus_stops"]) != k


def test_pois_cache_path_layout(tmp_path) -> None:
    p = pois_cache_path(tmp_path, "abcd")
    assert p == tmp_path / "pois" / "abcd.parquet"


# --------------------------------------------------------------------------- #
# build_pois_gdf                                                               #
# --------------------------------------------------------------------------- #


def test_build_pois_gdf_keeps_point_features_and_centroids_polygons() -> None:
    parks = gpd.GeoDataFrame(
        {"osmid": [1, 2], "leisure": ["park", "park"]},
        geometry=[Point(0, 0), Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])],
        crs="EPSG:32722",
    )
    stops = gpd.GeoDataFrame(
        {"osmid": [3], "highway": ["bus_stop"]},
        geometry=[Point(50, 50)],
        crs="EPSG:32722",
    )
    gdf = build_pois_gdf({"parks": parks, "bus_stops": stops}, working_crs="EPSG:32722")
    assert sorted(gdf["category"].unique()) == ["bus_stops", "parks"]
    assert all(g.geom_type == "Point" for g in gdf.geometry)
    by_id = {row["osm_id"]: row for _, row in gdf.iterrows()}
    assert by_id["1"]["geom_source"] == "point"
    assert by_id["2"]["geom_source"] == "centroid"
    # the centroid of a 10..20 square is (15, 15)
    assert by_id["2"]["geometry"].x == pytest.approx(15.0)
    assert by_id["2"]["geometry"].y == pytest.approx(15.0)
    # tags round-trip as JSON for Parquet stability
    import json as _json

    assert _json.loads(by_id["1"]["tags"]) == {"leisure": "park"}
    assert gdf.crs.to_epsg() == 32722


def test_build_pois_gdf_reprojects_per_category_inputs() -> None:
    wgs_parks = gpd.GeoDataFrame(
        {"osmid": [42], "leisure": ["park"]},
        geometry=[Point(-51.20, -30.00)],
        crs="EPSG:4326",
    )
    gdf = build_pois_gdf({"parks": wgs_parks}, working_crs="EPSG:32722")
    assert gdf.crs.to_epsg() == 32722
    assert gdf.iloc[0].geometry.x > 400_000


def test_build_pois_gdf_empty_input() -> None:
    gdf = build_pois_gdf({"parks": gpd.GeoDataFrame(geometry=[], crs="EPSG:32722")}, working_crs="EPSG:32722")
    assert len(gdf) == 0
    assert gdf.crs.to_epsg() == 32722


def test_build_pois_gdf_drops_empty_geometries() -> None:
    gdf = build_pois_gdf(
        {"parks": gpd.GeoDataFrame({"osmid": [1]}, geometry=[None], crs="EPSG:32722")},
        working_crs="EPSG:32722",
    )
    assert len(gdf) == 0


# --------------------------------------------------------------------------- #
# load_pois cache hit                                                          #
# --------------------------------------------------------------------------- #


def _fake_boundary() -> LoadedBoundary:
    """A LoadedBoundary in UTM 22S that's not actually used (we only need its
    `crs` and `bbox_wgs84` for keying + reprojection)."""
    # a tiny square in EPSG:32722 around 470_000 m easting
    poly = Polygon([(470_000, 6_680_000), (471_000, 6_680_000), (471_000, 6_681_000), (470_000, 6_681_000)])
    return LoadedBoundary(
        geometry=poly,
        crs="EPSG:32722",
        bbox_wgs84=(-51.25, -30.05, -51.15, -29.95),
    )


def test_load_pois_reads_cache_when_present(tmp_path) -> None:
    boundary = _fake_boundary()
    settings = Settings()
    settings.cache_dir = tmp_path

    # pre-seed the cache for the expected key
    categories = settings.poi.categories_for("nearest")
    key = pois_cache_key(boundary.bbox_wgs84, settings.network.osm_snapshot_date, categories)
    path = pois_cache_path(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)

    seeded = gpd.GeoDataFrame(
        {"category": ["parks"], "osm_id": ["1"], "geom_source": ["point"], "tags": ["{}"]},
        geometry=[Point(470_500.0, 6_680_500.0)],
        crs="EPSG:32722",
    )
    seeded.to_parquet(path)

    loaded = load_pois(boundary, settings, cache_dir=tmp_path)
    assert isinstance(loaded, LoadedPOIs)
    assert list(loaded.gdf["category"]) == ["parks"]
    assert loaded.gdf.crs.to_epsg() == 32722


def test_load_pois_unknown_fetcher_raises(tmp_path) -> None:
    boundary = _fake_boundary()
    settings = Settings()
    with pytest.raises(ValueError, match="unknown POI fetcher"):
        load_pois(boundary, settings, cache_dir=tmp_path, fetcher="bogus")


# --------------------------------------------------------------------------- #
# A failed fetch is an error, never a cached empty area                        #
# --------------------------------------------------------------------------- #


def _poi_settings() -> Settings:
    return Settings.load(overrides={"poi.presets": {"parks": {"leisure": "park"}, "shops": {"shop": "supermarket"}}})


def _cache_file(tmp_path, boundary, settings):
    categories = settings.poi.categories_for("nearest")
    return pois_cache_path(
        tmp_path, pois_cache_key(boundary.bbox_wgs84, settings.network.osm_snapshot_date, categories)
    )


def test_a_failed_fetch_raises_and_leaves_nothing_in_the_cache(tmp_path, monkeypatch) -> None:
    """A timeout used to become an empty category, get cached, and be read back as
    "no POIs here" by every later run. It must raise, and write nothing."""
    import osmnx as ox

    from segx.errors import POIFetchError

    def boom(polygon, tags):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(ox, "features_from_polygon", boom)
    boundary, settings = _fake_boundary(), _poi_settings()
    with pytest.raises(POIFetchError, match="parks") as err:
        load_pois(boundary, settings, cache_dir=tmp_path)
    assert err.value.code == "network.poi_fetch"
    assert err.value.details == {"category": "parks", "error": "TimeoutError"}
    assert not _cache_file(tmp_path, boundary, settings).exists()


def test_a_category_with_no_features_is_a_real_zero_and_is_cached(tmp_path, monkeypatch) -> None:
    """The query ran and matched nothing: that *is* the answer, so it is kept —
    alongside the category that did match."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    def fake(polygon, tags):
        if "leisure" in tags:
            raise InsufficientResponseError("No matching features. Check query location, tags, and log.")
        return gpd.GeoDataFrame({"osmid": [7]}, geometry=[Point(-51.2, -30.0)], crs="EPSG:4326")

    monkeypatch.setattr(ox, "features_from_polygon", fake)
    boundary, settings = _fake_boundary(), _poi_settings()
    loaded = load_pois(boundary, settings, cache_dir=tmp_path)
    assert list(loaded.gdf["category"]) == ["shops"]
    assert _cache_file(tmp_path, boundary, settings).exists()


def test_an_overpass_remark_is_a_failure_even_though_it_looks_like_no_features(tmp_path, monkeypatch) -> None:
    """Overpass reports a server-side timeout as HTTP 200 with a `remark` and no
    elements; OSMnx then says "No matching features". Without reading the remark
    that is indistinguishable from an empty area."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    from segx.errors import POIFetchError

    monkeypatch.setattr(
        ox._http, "_parse_response", lambda response: {"elements": [], "remark": "runtime error: Query timed out"}
    )

    def fake(polygon, tags):
        ox._http._parse_response(None)  # what OSMnx does with the server's answer
        raise InsufficientResponseError("No matching features. Check query location, tags, and log.")

    monkeypatch.setattr(ox, "features_from_polygon", fake)
    boundary, settings = _fake_boundary(), _poi_settings()
    with pytest.raises(POIFetchError, match="Query timed out"):
        load_pois(boundary, settings, cache_dir=tmp_path)
    assert not _cache_file(tmp_path, boundary, settings).exists()
    assert ox._http._parse_response(None)["remark"]  # the tap was removed: this is the test's own stub again


def test_a_malformed_response_is_not_mistaken_for_an_empty_area(tmp_path, monkeypatch) -> None:
    """OSMnx raises the same exception class for "no features" and for a response it
    could not use; only the first is a zero."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    from segx.errors import POIFetchError

    def fake(polygon, tags):
        raise InsufficientResponseError("Overpass API did not return a dict of results.")

    monkeypatch.setattr(ox, "features_from_polygon", fake)
    with pytest.raises(POIFetchError, match="did not return a dict"):
        load_pois(_fake_boundary(), _poi_settings(), cache_dir=tmp_path)
