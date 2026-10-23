"""P1: POI normalisation + cache key + cache-hit path. Live fetchers
(OSMnx Overpass, QuackOSM) are integration territory."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
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


# --------------------------------------------------------------------------- #
# POIs from a local .osm.pbf (QuackOSM)                                        #
# --------------------------------------------------------------------------- #


def _write_poi_pbf(path, *, box=(-52.0, -31.0, -50.0, -29.0)):
    """A tiny extract inside `_fake_boundary()`: a park, a clinic, a bus stop that is also a platform."""
    import osmium

    header = osmium.io.Header()
    header.add_box(osmium.osm.Box(osmium.osm.Location(box[0], box[1]), osmium.osm.Location(box[2], box[3])))
    header.set("osmosis_replication_timestamp", "2026-10-20T00:00:00Z")
    writer = osmium.SimpleWriter(str(path), header=header)
    try:
        nodes = {
            1: ((-51.30900, -30.00879), {"leisure": "park"}),
            2: ((-51.30692, -30.00699), {"amenity": "clinic"}),
            3: ((-51.30484, -30.00519), {"highway": "bus_stop", "public_transport": "platform"}),
            4: ((-51.30276, -30.00339), {"shop": "bakery"}),  # in no preset
        }
        for node_id, (loc, tags) in nodes.items():
            writer.add_node(osmium.osm.mutable.Node(id=node_id, location=loc, tags=tags))
    finally:
        writer.close()
    return path


def _pbf_settings(path) -> Settings:
    return Settings.load(
        overrides={
            "network.osm_pbf_url": str(path),
            "poi.presets": {
                "parks": {"leisure": "park"},
                "health": {"amenity": ["clinic", "hospital"]},
                "stops": {"highway": "bus_stop"},
                "platforms": {"public_transport": "platform"},
            },
        }
    )


def test_split_quackosm_groups_puts_a_feature_in_every_category_it_matches() -> None:
    from segx.io.pois import split_quackosm_groups

    raw = gpd.GeoDataFrame(
        {
            "stops": ["highway=bus_stop", None],
            "platforms": ["public_transport=platform", None],
            "parks": [None, "leisure=park"],
        },
        geometry=[Point(-51.2, -30.0), Point(-51.3, -30.1)],
        index=pd.Index(["node/3", "node/1"], name="feature_id"),
        crs="EPSG:4326",
    )
    out = split_quackosm_groups(raw, ["stops", "platforms", "parks", "absent"])
    assert list(out["stops"]["id"]) == ["node/3"] and list(out["platforms"]["id"]) == ["node/3"]
    assert out["parks"].iloc[0]["leisure"] == "park"
    assert len(out["absent"]) == 0
    pois = build_pois_gdf(out, working_crs="EPSG:32722")
    assert sorted(pois["category"]) == ["parks", "platforms", "stops"]
    assert set(pois["osm_id"]) == {"node/1", "node/3"}
    assert pois.loc[pois["category"] == "parks", "tags"].iloc[0] == '{"leisure": "park"}'


def test_quackosm_reads_pois_from_a_local_pbf(tmp_path) -> None:
    """The real QuackOSM call on a real (tiny) PBF — pins the API we use."""
    path = _write_poi_pbf(tmp_path / "pois.osm.pbf")
    settings = _pbf_settings(path)
    loaded = load_pois(_fake_boundary(), settings, cache_dir=tmp_path / "cache")
    got = sorted(zip(loaded.gdf["category"], loaded.gdf["osm_id"], strict=True))
    assert got == [("health", "node/2"), ("parks", "node/1"), ("platforms", "node/3"), ("stops", "node/3")]
    assert loaded.gdf.crs.to_epsg() == 32722
    assert (tmp_path / "cache" / "quackosm").is_dir()  # not ./files in the working directory


def test_auto_fetcher_uses_the_pbf_and_never_overpass(tmp_path, monkeypatch) -> None:
    import osmnx as ox
    import quackosm

    def no_overpass(*args, **kwargs):
        raise AssertionError("Overpass must not be contacted when osm_pbf_url is set")

    calls = []

    def fake_quackosm(pbf_path, **kwargs):
        calls.append((pbf_path, kwargs))
        return gpd.GeoDataFrame(
            {"parks": ["leisure=park"]},
            geometry=[Point(-51.2, -30.0)],
            index=pd.Index(["node/1"], name="feature_id"),
            crs="EPSG:4326",
        )

    monkeypatch.setattr(ox, "features_from_polygon", no_overpass)
    monkeypatch.setattr(quackosm, "convert_pbf_to_geodataframe", fake_quackosm)
    path = _write_poi_pbf(tmp_path / "pois.osm.pbf")
    settings = _pbf_settings(path)
    loaded = load_pois(_fake_boundary(), settings, cache_dir=tmp_path)
    assert list(loaded.gdf["category"]) == ["parks"]
    assert len(calls) == 1  # one read for every category
    assert set(calls[0][1]["tags_filter"]) == {"parks", "health", "stops", "platforms"}
    assert calls[0][1]["verbosity_mode"] == "silent"


def test_auto_fetcher_without_a_pbf_uses_overpass(tmp_path, monkeypatch) -> None:
    import osmnx as ox

    seen = []

    def fake(polygon, tags):
        seen.append(tags)
        return gpd.GeoDataFrame({"osmid": [7]}, geometry=[Point(-51.2, -30.0)], crs="EPSG:4326")

    monkeypatch.setattr(ox, "features_from_polygon", fake)
    load_pois(_fake_boundary(), _poi_settings(), cache_dir=tmp_path)
    assert len(seen) == 2


def test_pbf_pois_do_not_share_the_overpass_cache_entry(tmp_path) -> None:
    from segx.network.builder import osm_source

    settings = _pbf_settings(_write_poi_pbf(tmp_path / "pois.osm.pbf"))
    categories = settings.poi.categories_for("nearest")
    bbox = _fake_boundary().bbox_wgs84
    overpass = pois_cache_key(bbox, None, categories)
    assert pois_cache_key(bbox, None, categories, source=None) == overpass
    assert pois_cache_key(bbox, None, categories, source=osm_source(settings)) != overpass


def test_a_pbf_that_misses_the_area_is_a_typed_error(tmp_path) -> None:
    from segx.errors import OSMExtractError

    path = _write_poi_pbf(tmp_path / "far.osm.pbf", box=(-48.0, -27.0, -47.0, -26.0))
    with pytest.raises(OSMExtractError, match="does not cover"):
        load_pois(_fake_boundary(), _pbf_settings(path), cache_dir=tmp_path)


def test_a_missing_pbf_is_input_file_not_found(tmp_path) -> None:
    from segx.errors import InputFileNotFoundError

    with pytest.raises(InputFileNotFoundError):
        load_pois(_fake_boundary(), _pbf_settings(tmp_path / "nope.osm.pbf"), cache_dir=tmp_path)
