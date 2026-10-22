"""RPC sidecar: ``inspect_layer`` / ``get_layer_preview`` and the wire dispatch.

The inspection helpers are exercised directly on small GeoPackages / CSVs built
in ``tmp_path``; the dispatch tests drive :func:`segx.rpc.serve` with a fake
stdin and read the NDJSON it writes, so the reply envelopes and error codes are
checked exactly as the desktop app sees them.
"""

from __future__ import annotations

import io
import json
import math

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from segx import rpc
from segx.errors import InputError, InputFileNotFoundError
from segx.io.inspect import inspect_layer, layer_preview

# Porto Alegre-ish, in SIRGAS 2000 / UTM 22S (EPSG:31982).
_UTM_CRS = "EPSG:31982"
_X0, _Y0 = 478_000.0, 6_676_000.0


def _points_gpkg(tmp_path, n: int = 10, crs: str | None = _UTM_CRS):
    gdf = gpd.GeoDataFrame(
        {"id": range(n), "income": [float(i) for i in range(n)], "label": [f"p{i}" for i in range(n)]},
        geometry=[Point(_X0 + 100 * i, _Y0 + 50 * i) for i in range(n)],
        crs=crs,
    )
    path = tmp_path / "residences.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def _polygons_gpkg(tmp_path):
    squares = [
        Polygon(
            [
                (_X0 + i * 500, _Y0),
                (_X0 + i * 500 + 400, _Y0),
                (_X0 + i * 500 + 400, _Y0 + 400),
                (_X0 + i * 500, _Y0 + 400),
            ]
        )
        for i in range(3)
    ]
    path = tmp_path / "census.gpkg"
    gpd.GeoDataFrame({"pop": [10, float("nan"), 30]}, geometry=squares, crs=_UTM_CRS).to_file(path, driver="GPKG")
    return path


def _lines_gpkg(tmp_path):
    lines = [LineString([(_X0, _Y0 + i * 100), (_X0 + 1_000, _Y0 + i * 100)]) for i in range(4)]
    path = tmp_path / "streets.gpkg"
    gpd.GeoDataFrame({"name": list("abcd")}, geometry=lines, crs=_UTM_CRS).to_file(path, driver="GPKG")
    return path


def _csv(tmp_path):
    path = tmp_path / "od.csv"
    path.write_text("origin_id,dest_id,weight\n1,2,3\n2,3,4\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# inspect_layer                                                                #
# --------------------------------------------------------------------------- #


def test_inspect_points_reports_crs_count_kind_and_wgs84_bbox(tmp_path):
    info = inspect_layer(_points_gpkg(tmp_path))

    assert info["crs"] == _UTM_CRS
    assert info["feature_count"] == 10
    assert info["geometry_type"] == "POINT"
    assert info["driver"] == "GPKG"
    minx, miny, maxx, maxy = info["bbox"]
    # reprojected to lon/lat around Porto Alegre, not left in metres
    assert -52 < minx < maxx < -50
    assert -31 < miny < maxy < -29


@pytest.mark.parametrize(("builder", "kind"), [(_polygons_gpkg, "POLYGON"), (_lines_gpkg, "LINE")])
def test_inspect_geometry_kinds(tmp_path, builder, kind):
    assert inspect_layer(builder(tmp_path))["geometry_type"] == kind


def test_inspect_csv_is_a_table_without_crs_or_bbox(tmp_path):
    info = inspect_layer(_csv(tmp_path))

    assert info["geometry_type"] == "TABLE"
    assert info["feature_count"] == 2
    assert info["crs"] is None
    assert info["bbox"] is None


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_inspect_layer_without_crs_reports_none_but_keeps_bbox(tmp_path):
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(-51.2, -30.0), Point(-51.1, -30.1)])
    path = tmp_path / "nocrs.shp"  # a shapefile without .prj declares no CRS
    gdf.to_file(path, driver="ESRI Shapefile")

    info = inspect_layer(path)

    assert info["crs"] is None
    # displayed as WGS84, the loaders' explicit assumption
    assert info["bbox"] == pytest.approx([-51.2, -30.1, -51.1, -30.0])


def test_inspect_missing_file_raises_typed_error(tmp_path):
    with pytest.raises(InputFileNotFoundError) as exc:
        inspect_layer(tmp_path / "nope.gpkg")
    assert exc.value.code == "input.file_not_found"


def test_inspect_unreadable_file_raises_input_error(tmp_path):
    path = tmp_path / "broken.gpkg"
    path.write_bytes(b"not a geopackage")
    with pytest.raises(InputError) as exc:
        inspect_layer(path)
    assert exc.value.code == "input.invalid"


# --------------------------------------------------------------------------- #
# get_layer_preview                                                            #
# --------------------------------------------------------------------------- #


def test_preview_caps_features_and_returns_wgs84_geojson(tmp_path):
    preview = layer_preview(_points_gpkg(tmp_path, n=10), max_features=4)

    assert preview["geometry_type"] == "POINT"
    assert preview["total_features"] == 10
    assert len(preview["features"]) == 4
    feature = preview["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Point"
    lon, lat = feature["geometry"]["coordinates"]
    assert -52 < lon < -50 and -31 < lat < -29
    assert set(feature["properties"]) == {"id", "income", "label"}


def test_preview_properties_are_json_safe(tmp_path):
    preview = layer_preview(_polygons_gpkg(tmp_path))

    pops = [f["properties"]["pop"] for f in preview["features"]]
    assert pops == [10.0, None, 30.0]  # NaN never reaches the wire
    json.dumps(preview, allow_nan=False)  # strict JSON — the app's JSON.parse rejects NaN


def test_preview_csv_has_no_features(tmp_path):
    preview = layer_preview(_csv(tmp_path))
    assert preview == {"geometry_type": "TABLE", "total_features": 2, "features": []}


# --------------------------------------------------------------------------- #
# Wire dispatch                                                                #
# --------------------------------------------------------------------------- #


def _serve(monkeypatch, *requests: dict) -> list[dict]:
    lines = "".join(json.dumps(r) + "\n" for r in requests)
    out = io.StringIO()
    monkeypatch.setattr(rpc.sys, "stdin", io.StringIO(lines))
    monkeypatch.setattr(rpc.sys, "stdout", out)
    rpc.serve()
    return [json.loads(line) for line in out.getvalue().splitlines()]


def test_serve_inspect_and_preview_replies(monkeypatch, tmp_path):
    path = str(_lines_gpkg(tmp_path))
    replies = _serve(
        monkeypatch,
        {"id": "a", "method": "inspect_layer", "params": {"path": path}},
        {"id": "b", "method": "get_layer_preview", "params": {"path": path, "max_features": 2}},
    )

    assert replies[0]["event"] == "ready"
    inspect, preview = replies[1], replies[2]
    assert inspect["id"] == "a" and inspect["ok"] is True
    assert inspect["result"]["geometry_type"] == "LINE"
    assert inspect["result"]["feature_count"] == 4
    assert preview["id"] == "b" and preview["ok"] is True
    assert len(preview["result"]["features"]) == 2
    assert all(math.isfinite(c) for c in preview["result"]["features"][0]["geometry"]["coordinates"][0])


def test_serve_inspect_missing_file_is_a_typed_error(monkeypatch, tmp_path):
    replies = _serve(monkeypatch, {"id": 1, "method": "inspect_layer", "params": {"path": str(tmp_path / "x.shp")}})

    reply = replies[1]
    assert reply["ok"] is False
    assert reply["error"]["code"] == "input.file_not_found"
    assert reply["error"]["details"]["path"].endswith("x.shp")


def test_serve_inspect_without_path_is_bad_params(monkeypatch):
    replies = _serve(monkeypatch, {"id": 1, "method": "inspect_layer", "params": {}})
    assert replies[1]["error"]["code"] == "rpc.bad_params"


def test_describe_types_match_the_wire(monkeypatch):
    """The tightened protocol types (PoiPreset, NetworkInfo.transit) hold for the real reply."""
    result = _serve(monkeypatch, {"id": 1, "method": "describe"})[1]["result"]

    assert all(set(preset) == {"tags"} for preset in result["poi_presets"].values())
    assert set(result["network"]["transit"]) == {"sources", "note", "service_window_note"}


# --------------------------------------------------------------------------- #
# Columns (for the host's column mapping)                                      #
# --------------------------------------------------------------------------- #


def test_inspect_lists_columns_with_types_and_samples(tmp_path):
    info = inspect_layer(_points_gpkg(tmp_path, n=10))

    by_name = {c["name"]: c for c in info["columns"]}
    assert set(by_name) == {"id", "income", "label"}  # geometry is not an attribute column
    assert by_name["id"] == {"name": "id", "type": "number", "sample": [0, 1, 2]}  # ints stay ints
    assert by_name["income"]["type"] == "number"
    assert by_name["label"] == {"name": "label", "type": "text", "sample": ["p0", "p1", "p2"]}
    # the name geopandas gives the active geometry — what od_columns.origin_geom must say
    assert info["geometry_column"] == "geometry"


def test_inspect_csv_columns_are_typed_like_the_od_loader_reads_them(tmp_path):
    path = tmp_path / "lodes.csv"
    path.write_text(
        "hGEOID,wGEOID,SE01,label\n421010001001000,421010002002000,3,a\n421010001001001,421010002002000,,b\n",
        encoding="utf-8",
    )
    info = inspect_layer(path)

    by_name = {c["name"]: c for c in info["columns"]}
    # pandas inference (as io.od_pairs reads CSV), not GDAL's all-text CSV fields
    assert by_name["hGEOID"]["type"] == "number"
    assert by_name["hGEOID"]["sample"] == [421010001001000, 421010001001001]
    assert by_name["SE01"]["sample"] == [3.0, None]  # NaN never reaches the wire
    assert by_name["label"]["type"] == "text"
    assert info["geometry_column"] is None
    json.dumps(info, allow_nan=False)


# --------------------------------------------------------------------------- #
# Input presets (IBGE / LODES)                                                 #
# --------------------------------------------------------------------------- #


def test_input_presets_are_strict_json_with_resolved_class_columns():
    presets = rpc._describe()["input_presets"]
    json.dumps(presets, allow_nan=False)  # no Infinity: the app's JSON.parse rejects it

    ibge = presets["ibge"]["demographics"]
    income = next(a for a in ibge["attributes"] if a["name"] == "income")
    assert ibge["attribute_source"] == "demographics"
    assert income["bins"][-1] is None  # the open upper edge
    assert income["class_columns"] == [f"income_b{i}" for i in range(6)]  # convention, resolved

    lodes = presets["lodes"]
    assert lodes["demographics"]["attribute_source"] == "od"
    assert lodes["demographics"]["polygon_id_column"] == "GEOID"
    assert lodes["demographics"]["attributes"][0]["class_columns"] == ["SE01", "SE02", "SE03"]
    assert (lodes["od_columns"]["origin_id_ref"], lodes["od_columns"]["dest_id_ref"]) == ("hGEOID", "wGEOID")


@pytest.mark.parametrize("name", ["ibge", "lodes"])
def test_a_preset_sent_back_as_overrides_validates(name):
    from segx.config import Settings

    preset = rpc._describe()["input_presets"][name]
    settings = Settings.load(None, {"demographics": preset["demographics"], "od_columns": preset["od_columns"]})

    assert settings.demographics.attribute_source == preset["demographics"]["attribute_source"]
    for attr in settings.demographics.attributes:
        if attr.bins is not None:
            assert attr.bins[-1] == math.inf  # null read back as an open edge


# --------------------------------------------------------------------------- #
# od_modes                                                                     #
# --------------------------------------------------------------------------- #


def test_od_modes_counts_every_row_and_maps_labels_to_graphs(tmp_path):
    from segx.io.inspect import od_modes

    path = tmp_path / "od.csv"
    path.write_text("o,d,mode\n1,2,car\n1,3,bus\n2,3,subway\n3,1,car\n4,1,\n5,2,walk\n", encoding="utf-8")

    result = od_modes(path, "mode")

    assert result["counts"] == {"car": 2, "bus": 1, "subway": 1, "walk": 1}
    assert result["graph_modes"] == {"car": "drive", "bus": "transit", "subway": "transit", "walk": "walk"}
    assert result["missing"] == 1


def test_serve_od_modes_unknown_column_is_a_typed_error(monkeypatch, tmp_path):
    path = tmp_path / "od.csv"
    path.write_text("o,d\n1,2\n", encoding="utf-8")

    reply = _serve(monkeypatch, {"id": 1, "method": "od_modes", "params": {"path": str(path), "mode_column": "mode"}})[
        1
    ]

    assert reply["ok"] is False
    assert reply["error"]["code"] == "input.invalid"
    assert reply["error"]["details"]["column"] == "mode"
