"""P8: the deliverable writers — ``segments.gpkg`` + ``metrics.xlsx``.

Round-trips the synthetic transit city (a bus ride on the bottom street row, a
metro ride on its own rail line) through :func:`segx.orchestrator.write` and reads
the files back: per-segment masses and metric values must equal the in-memory
frames, the rail segments must be a layer of their own, and the provenance
stamps (invariants #4 and #7) must be in both files.
"""

from __future__ import annotations

import json

import networkx as nx
import pandas as pd
import pyogrio
import pytest

from segx.errors import OutputColumnCollisionError
from segx.io.writers import (
    COLUMNS_TABLE,
    METADATA_TABLE,
    SEGMENT_KEYS,
    resolve_geometries,
    segment_table,
    write_metrics_workbook,
)
from segx.network.registry import GraphRegistry
from segx.orchestrator import aggregate, write
from tests.unit.test_orchestrator import (
    _transit_population,
    _transit_routing,
    _transit_state_and_settings,
)


def _run(settings_update: dict | None = None):
    state, settings, graph = _transit_state_and_settings()
    if settings_update:
        settings = settings.model_copy(update=settings_update)
    routing = _transit_routing(graph, settings)
    aggregation = aggregate(state, _transit_population(), routing, settings)
    return state, settings, routing, aggregation


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    state, settings, routing, aggregation = _run()
    out = tmp_path_factory.mktemp("out")
    outputs = write(state, routing, aggregation, settings, out)
    return aggregation, outputs


def _read_segments(outputs) -> pd.DataFrame:
    frames = [pyogrio.read_dataframe(outputs.segments_path, layer=name) for name in outputs.layers]
    return pd.concat(frames, ignore_index=True)


def test_rail_segments_are_a_layer_of_their_own(written) -> None:
    aggregation, outputs = written
    assert outputs.layers == ["transit_rail_line", "transit_street"]
    rail = pyogrio.read_dataframe(outputs.segments_path, layer="transit_rail_line")
    street = pyogrio.read_dataframe(outputs.segments_path, layer="transit_street")
    assert set(rail["layer"]) == {"rail_line"} and (rail["u"] < 0).all()
    assert set(street["layer"]) == {"street"} and (street["u"] >= 0).all()


def test_every_segment_gets_its_geometry_even_a_one_way_rail_ride(written) -> None:
    # The accumulator stores segments undirected — (min(u, v), max(u, v), key) —
    # while the rail ride is a one-way edge stored the other way round in the graph.
    _, outputs = written
    assert outputs.stats["n_segments_no_geometry"] == 0
    segments = _read_segments(outputs)
    assert segments.geometry.notna().all()
    assert (segments.geometry.length > 0).all()


def test_masses_round_trip_against_the_edge_table(written) -> None:
    aggregation, outputs = written
    segments = _read_segments(outputs).set_index(list(SEGMENT_KEYS))
    edge = aggregation.edge_table
    for (attribute, label), sub in edge.groupby(["attribute", "class"]):
        expected = sub.set_index(list(SEGMENT_KEYS))["mass"]
        column = f"{attribute}_{label}"
        got = segments.loc[expected.index, column]
        assert got.to_numpy() == pytest.approx(expected.to_numpy())
    assert len(segments) == aggregation.stats["n_segments"]


def test_per_segment_metric_values_round_trip(written) -> None:
    aggregation, outputs = written
    segments = _read_segments(outputs).set_index(list(SEGMENT_KEYS))
    diversity = aggregation.metrics["diversity"].set_index(list(SEGMENT_KEYS))
    got = segments.loc[diversity.index, "diversity_race"]
    assert got.to_numpy() == pytest.approx(diversity["diversity"].to_numpy())
    # two metrics both name a `dominant_class`; the prefix keeps them apart
    assert "dominant_trajectories_dominant_class_race" in segments.columns
    assert "segregated_trajectories_dominant_class_race" in segments.columns


def test_provenance_is_in_both_files(written) -> None:
    aggregation, outputs = written
    table = pyogrio.read_dataframe(outputs.segments_path, layer=METADATA_TABLE)
    gpkg_meta = dict(zip(table["key"], table["value"], strict=True))
    xlsx = pd.read_excel(outputs.workbook_path, sheet_name="metadata", keep_default_na=False)
    xlsx_meta = dict(zip(xlsx["key"], xlsx["value"].astype(str), strict=True))
    for meta in (gpkg_meta, xlsx_meta):
        assert meta["osm_snapshot_date"] == "null"  # unpinned in this fixture, and said so
        assert meta["transit_source"] == "gtfs"
        assert meta["transit.transit_service_window_bounds_min"] == "[360, 540]"
        assert meta["fragmentation_params.method"] == "box_counting"
        assert meta["working_crs"] == "EPSG:32618"
        assert "settings_json" in meta
    assert json.loads(gpkg_meta["metrics_enabled"]) == aggregation.stats["metrics_enabled"]
    # and on each line layer, for QGIS's metadata panel
    layer_meta = pyogrio.read_info(outputs.segments_path, layer="transit_street")["layer_metadata"]
    assert layer_meta["transit_source"] == "gtfs"
    assert "settings_json" not in layer_meta


def test_the_columns_table_documents_every_layer_column(written) -> None:
    _, outputs = written
    columns = pyogrio.read_dataframe(outputs.segments_path, layer=COLUMNS_TABLE)
    segments = _read_segments(outputs)
    assert set(segments.columns) - {"geometry"} == set(columns["column"])
    mass = columns[columns["kind"] == "mass"]
    assert set(zip(mass["attribute"], mass["class"], strict=True)) == {("race", "a"), ("race", "b")}


def test_workbook_carries_summaries_group_tables_and_audits(written) -> None:
    aggregation, outputs = written
    sheets = pd.read_excel(outputs.workbook_path, sheet_name=None)
    # per-segment metrics: the summary only (the segments are in the GeoPackage)
    assert len(sheets["diversity"]) == len(aggregation.summaries["diversity"])
    # per-group metrics: the table, and its summary beside it
    assert set(sheets["exposure"]["class"]) == {"a", "b"}
    assert "exposure_summary" in sheets
    conservation = dict(zip(sheets["audit_conservation"]["key"], sheets["audit_conservation"]["value"], strict=True))
    assert conservation["balance"] == pytest.approx(0.0)
    assert conservation["emitted"] == pytest.approx(aggregation.ledger.emitted)
    assert "audit_fragmentation" in sheets and "audit_cross_mode_sets" in sheets


def test_a_disabled_metric_has_no_column_and_no_sheet(tmp_path) -> None:
    state, settings, graph = _transit_state_and_settings()
    metrics = settings.metrics.model_copy(update={"disabled": ["diversity", "exposure"]})
    settings = settings.model_copy(update={"metrics": metrics})
    routing = _transit_routing(graph, settings)
    aggregation = aggregate(state, _transit_population(), routing, settings)
    outputs = write(state, routing, aggregation, settings, tmp_path)

    segments = _read_segments(outputs)
    assert "diversity_race" not in segments.columns
    assert "experienced_segregation_race" in segments.columns
    assert "diversity" not in outputs.sheets and "exposure" not in outputs.sheets
    meta = pd.read_excel(outputs.workbook_path, sheet_name="metadata", keep_default_na=False)
    stamped = dict(zip(meta["key"], meta["value"], strict=True))
    assert stamped["metrics_disabled"] == '["diversity", "exposure"]'


# --------------------------------------------------------------------------- #
# Pieces                                                                       #
# --------------------------------------------------------------------------- #


def _edge_rows(rows: list[tuple]) -> pd.DataFrame:
    """Edge-table rows ``(u, v, attribute, class, mass)`` on one walk partition."""
    return pd.DataFrame(
        [
            {
                "mode": "walk",
                "category": "commute",
                "binding": "od",
                "u": u,
                "v": v,
                "key": 0,
                "layer": "street",
                "corridor": f"own:{u}:{v}:0",
                "length_m": 100.0,
                "attribute": attribute,
                "class": label,
                "mass": mass,
                "n_paths": 1,
            }
            for u, v, attribute, label, mass in rows
        ]
    )


def test_a_segment_a_metric_omits_is_nan_not_zero() -> None:
    edge = _edge_rows([(1, 2, "race", "a", 2.0), (2, 3, "race", "a", 1.0)])
    metric = pd.DataFrame(
        [
            {
                "mode": "walk",
                "category": "commute",
                "binding": "od",
                "u": 1,
                "v": 2,
                "key": 0,
                "attribute": "race",
                "diversity": 0.5,
            }
        ]
    )
    wide, _ = segment_table(edge, {"diversity": metric})
    by_edge = wide.set_index(["u", "v"])["diversity_race"]
    assert by_edge[(1, 2)] == 0.5
    assert pd.isna(by_edge[(2, 3)])


def test_colliding_column_names_raise_a_typed_error() -> None:
    # attribute `a_b` × class `c` and attribute `a` × class `b_c` both spell `a_b_c`
    edge = _edge_rows([(1, 2, "a_b", "c", 1.0), (1, 2, "a", "b_c", 1.0)])
    with pytest.raises(OutputColumnCollisionError) as err:
        segment_table(edge, {})
    assert err.value.code == "output.column_collision"
    assert "a_b_c" in err.value.details["columns"]


def test_a_segment_no_graph_knows_is_kept_and_counted() -> None:
    graph = nx.MultiDiGraph(crs="EPSG:32618")
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=10.0, y=0.0)
    graph.add_edge(1, 2, 0)
    graphs = GraphRegistry()
    graphs.register("walk", graph)
    wide, _ = segment_table(_edge_rows([(1, 2, "race", "a", 1.0), (7, 8, "race", "a", 1.0)]), {})
    geometry, n_missing = resolve_geometries(wide, graphs)
    assert n_missing == 1
    assert geometry[0] is not None and geometry[1] is None


def test_sheet_names_are_truncated_and_kept_unique(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1]})
    long = "a_metric_name_much_longer_than_thirty_one_characters"
    names = write_metrics_workbook([(long, frame), (long, frame), ("bad/name", frame)], tmp_path / "m.xlsx")
    assert all(len(n) <= 31 for n in names)
    assert len(set(names)) == 3
    assert names[2] == "bad_name"
