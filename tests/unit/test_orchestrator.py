"""P1: orchestrator wires the ingest stages end-to-end.

Live OSM access is avoided two ways: ``settings.network.modes`` is set to ``[]``
(no per-mode graph builds) and the POI cache is pre-seeded so ``load_pois``
takes the cache-hit path instead of calling Overpass.
"""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

from segx.config import (
    CatchmentBindingConfig,
    DemographicAttribute,
    DemographicSchema,
    FiguresConfig,
    NearestBindingConfig,
    NetworkConfig,
    POIBindingConfig,
    RoutingConfig,
    Settings,
)
from segx.io.pois import pois_cache_key, pois_cache_path
from segx.network.cache import cache_entry, cache_key, write_graph
from segx.network.registry import GraphRegistry
from segx.network.transit import build_transit_graph
from segx.orchestrator import (
    IngestResult,
    PopulationResult,
    RoutingResult,
    aggregate,
    build_population,
    ingest,
    run_pipeline,
)


def _tiny_schema() -> DemographicSchema:
    return DemographicSchema(
        attributes=[
            DemographicAttribute(name="race", kind="categorical", categories=["a", "b"]),
        ],
    )


def _settings(tmp_path) -> Settings:
    return Settings(
        demographics=_tiny_schema(),
        network=NetworkConfig(modes=[]),  # skip OSMnx fetch in this test
        poi=POIBindingConfig(
            presets={"parks": {"leisure": "park"}},
            nearest=NearestBindingConfig(categories=["parks"]),
            catchment=CatchmentBindingConfig(categories=["parks"]),
        ),
        cache_dir=tmp_path / "cache",
    )


def _write_boundary(path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326").to_file(path, driver="GPKG")


def _write_demographics(path) -> None:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    gpd.GeoDataFrame(
        {"id": ["s1"], "pop": [100.0], "race_a": [40.0], "race_b": [60.0]},
        geometry=[poly],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")


def _write_od_pairs(path) -> None:
    pd.DataFrame(
        {
            "origin_id": ["a"],
            "origin_lon": [-51.20],
            "origin_lat": [-30.00],
            "dest_lon": [-51.10],
            "dest_lat": [-30.05],
        }
    ).to_csv(path, index=False)


def _seed_poi_cache(settings, tmp_path) -> None:
    """Write an empty (but valid) POI parquet at the key the orchestrator will compute."""
    bbox_wgs84 = (-51.25, -30.05, -51.15, -29.95)
    categories = settings.poi.categories_for("nearest")
    key = pois_cache_key(bbox_wgs84, settings.network.osm_snapshot_date, categories)
    path = pois_cache_path(settings.cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"category": ["parks"], "osm_id": ["p1"], "geom_source": ["point"], "tags": ["{}"]},
        geometry=[Point(470_500.0, 6_680_500.0)],
        crs="EPSG:32722",
    ).to_parquet(path)


def test_ingest_without_od_pairs(tmp_path) -> None:
    settings = _settings(tmp_path)
    boundary_file = tmp_path / "boundary.gpkg"
    demo_file = tmp_path / "demo.gpkg"
    _write_boundary(boundary_file)
    _write_demographics(demo_file)
    _seed_poi_cache(settings, tmp_path)

    result = ingest(
        settings,
        boundary_path=boundary_file,
        demographics_path=demo_file,
        od_pairs_path=None,
    )

    assert isinstance(result, IngestResult)
    assert result.boundary.crs == "EPSG:32722"
    assert result.demographics.kind == "polygon"
    assert len(result.demographics.gdf) == 1
    assert result.graphs.modes() == []  # modes=[] in this test
    assert len(result.pois.gdf) == 1
    assert result.od_pairs is None


def test_ingest_with_od_pairs(tmp_path) -> None:
    settings = _settings(tmp_path)
    boundary_file = tmp_path / "boundary.gpkg"
    demo_file = tmp_path / "demo.gpkg"
    od_file = tmp_path / "od.csv"
    _write_boundary(boundary_file)
    _write_demographics(demo_file)
    _write_od_pairs(od_file)
    _seed_poi_cache(settings, tmp_path)

    result = ingest(
        settings,
        boundary_path=boundary_file,
        demographics_path=demo_file,
        od_pairs_path=od_file,
    )
    assert result.od_pairs is not None
    assert len(result.od_pairs.gdf) == 1
    # od pairs reprojected into the boundary's working CRS
    assert result.od_pairs.gdf.crs.to_epsg() == 32722


def _seed_walk_graph_cache(settings, tmp_path) -> None:
    """Seed a tiny OSMnx-shaped ``walk`` graph at the cache key ingest will compute,
    so routing (P3) can run without a live OSM fetch."""
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=-51.20, y=-30.00)
    g.add_node(2, x=-51.19, y=-30.00)
    g.add_edge(1, 2, length=965.0, geometry=LineString([(-51.20, -30.00), (-51.19, -30.00)]))
    g.add_edge(2, 1, length=965.0, geometry=LineString([(-51.19, -30.00), (-51.20, -30.00)]))
    bbox_wgs84 = (-51.25, -30.05, -51.15, -29.95)
    entry = cache_entry(settings.cache_dir, cache_key(bbox_wgs84, settings.network.osm_snapshot_date, "walk"))
    write_graph(g, entry, bbox=bbox_wgs84, snapshot_date=settings.network.osm_snapshot_date, network_type="walk")


def test_run_pipeline_runs_end_to_end_and_writes_the_deliverables(tmp_path) -> None:
    # This test builds a real walk graph (seeded cache) so routing, aggregation
    # and the P8 writers all run; the NetworkX oracle keeps it light (Pandana is
    # exercised by the integration test).
    settings = _settings(tmp_path).model_copy(
        update={
            "network": NetworkConfig(modes=["walk"]),
            "routing": RoutingConfig(engine="networkx"),
            "figures": FiguresConfig(basemap=False, dpi=50),  # no web tiles in tests
        }
    )
    result, stages = _run_end_to_end(settings, tmp_path)

    assert stages[-1] == "render_figures"
    assert stages.index("write_outputs") < stages.index("render_figures")
    assert result.outputs.segments_path == tmp_path / "out" / "segments.gpkg"
    assert result.outputs.segments_path.exists() and result.outputs.workbook_path.exists()
    assert result.outputs.stats["n_segment_rows"] == result.aggregation.stats["n_segments"]
    # the figures stage is wired: it ran, failed nothing, stayed offline. This
    # city routes zero-length paths, so there are no segments to draw — the
    # figures themselves are covered in test_figures.py on the synthetic grid.
    figures = result.figures
    assert figures is not None and figures.failed == [] and figures.basemap == "off"
    assert figures.root == tmp_path / "out"
    assert "overview.png" in figures.files["maps"]
    assert (tmp_path / "out" / "maps" / "overview.png").is_file()
    assert {a["kind"] for a in figures.artifacts()} <= {
        "figures:maps",
        "figures:segments",
        "figures:charts",
        "figures:tables",
    }


def test_run_pipeline_without_figures_ends_at_write_outputs(tmp_path) -> None:
    settings = _settings(tmp_path).model_copy(
        update={
            "network": NetworkConfig(modes=["walk"]),
            "routing": RoutingConfig(engine="networkx"),
            "figures": FiguresConfig(enabled=False),
        }
    )
    result, stages = _run_end_to_end(settings, tmp_path)
    assert stages[-1] == "write_outputs"
    assert result.figures is None
    assert not (tmp_path / "out" / "maps").exists()


def _run_end_to_end(settings, tmp_path):
    boundary_file = tmp_path / "boundary.gpkg"
    demo_file = tmp_path / "demo.gpkg"
    _write_boundary(boundary_file)
    _write_demographics(demo_file)
    _seed_poi_cache(settings, tmp_path)
    _seed_walk_graph_cache(settings, tmp_path)

    stages: list[str] = []
    result = run_pipeline(
        settings,
        boundary_path=boundary_file,
        demographics_path=demo_file,
        od_pairs_path=None,
        output_dir=tmp_path / "out",
        progress=lambda ev: stages.append(ev.stage),
    )
    return result, stages


def test_build_population_centroid_origins(tmp_path) -> None:
    settings = _settings(tmp_path)
    boundary_file = tmp_path / "boundary.gpkg"
    demo_file = tmp_path / "demo.gpkg"
    _write_boundary(boundary_file)
    _write_demographics(demo_file)
    _seed_poi_cache(settings, tmp_path)

    state = ingest(settings, boundary_path=boundary_file, demographics_path=demo_file, od_pairs_path=None)
    population = build_population(state, settings)
    origins = population.origins
    assert list(origins["origin_id"]) == ["s1"]
    assert bool(origins["demo_resolved"].iloc[0]) is True
    assert origins["demo_vector"].iloc[0] == {"race": {"a": 40.0, "b": 60.0}}
    # centroid of the demographics polygon lands in the working CRS
    assert origins.crs.to_epsg() == 32722


# --------------------------------------------------------------------------- #
# P7 — the transit path through `aggregate`                                    #
# --------------------------------------------------------------------------- #


def _transit_state_and_settings():
    """A minimal ``IngestResult`` carrying only a merged transit graph.

    ``aggregate`` reads just ``state.graphs`` and ``state.od_pairs``, so the other
    ingest artifacts can be absent — this keeps the test on the orchestrator's P7
    wiring instead of re-testing IO.
    """
    from segx.config import TransitConfig
    from segx.validation.synthetic_city import (
        SYNTHETIC_CRS,
        TRANSIT_FAST_HEADWAY_MIN,
        build_grid_graph,
        build_transit_feed,
    )

    settings = Settings(
        demographics=_tiny_schema(),
        network=NetworkConfig(modes=["transit"]),
        transit=TransitConfig(enabled=True),
        routing=RoutingConfig(engine="networkx", access_mode="transit"),
    )
    grid = build_grid_graph()
    graph = build_transit_graph(
        build_transit_feed(TRANSIT_FAST_HEADWAY_MIN), grid, grid, settings, working_crs=SYNTHETIC_CRS
    )
    graphs = GraphRegistry()
    graphs.register("transit", graph)
    state = IngestResult(boundary=None, demographics=None, graphs=graphs, pois=None, od_pairs=None)
    return state, settings, graph


def _transit_routing(graph, settings) -> RoutingResult:
    from segx.routing.inputs import RoutableODPair
    from segx.routing.networkx_engine import NetworkXEngine
    from segx.routing.od_router import od_pair_trajectories

    engine = NetworkXEngine({"transit": graph}, settings)
    pairs = [
        RoutableODPair("o1", 0, "d_bus", 3, "transit", "commute", 1.0),
        RoutableODPair("o1", 12, "d_metro", 15, "transit", "commute", 1.0),
    ]
    trajectories = list(od_pair_trajectories(pairs, engine, settings))
    return RoutingResult(trajectories=trajectories, stats={})


def _transit_population() -> PopulationResult:
    origins = gpd.GeoDataFrame(
        {"origin_id": ["o1"], "demo_vector": [{"race": {"a": 3.0, "b": 1.0}}]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:32618",
    )
    return PopulationResult(origins=origins)


def test_aggregate_expands_transit_rides_and_labels_layers() -> None:
    """The `expand_transit_rides` stage must run *before* the accumulator, so the
    edge table holds real street segments and rail lines — never access edges."""
    state, settings, graph = _transit_state_and_settings()
    routing = _transit_routing(graph, settings)
    population = _transit_population()

    result = aggregate(state, population, routing, settings)
    frame = result.edge_table

    assert set(frame["layer"]) == {"street", "rail_line"}
    # the bus commute landed on the three real grid edges of the bottom row
    street = frame[frame["layer"] == "street"]
    assert {(int(r.u), int(r.v), int(r.key)) for r in street.itertuples(index=False)} == {
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
    }
    # and the metro stayed on synthetic rail edges, not the streets above the tunnel
    rail = frame[frame["layer"] == "rail_line"]
    assert (rail["u"] < 0).all() and (rail["v"] < 0).all()
    # conservation still balances with transit trajectories in the stream
    assert result.ledger.emitted == pytest.approx(result.ledger.deposited)
    assert result.stats["n_unreachable"] == 0


def test_aggregate_reports_the_cross_mode_coverage_audit() -> None:
    """P7.1: whether `cross_mode_exposure` had anything to join is not visible
    in its own output (an empty join looks like `== exposure`), so the numbers ride
    in `stats`. `n_surface_rides_unmatched` comes from the transit graph itself."""
    state, settings, graph = _transit_state_and_settings()
    result = aggregate(state, _transit_population(), _transit_routing(graph, settings), settings)

    audit = result.stats["cross_mode"]
    assert audit["n_modes_present"] == 1  # this run is transit-only
    assert audit["n_segments_total"] == 3  # the three bottom-row street segments
    assert audit["n_segments_multimode"] == 0
    assert audit["cross_mode_join_rate"] == 0.0
    assert audit["n_surface_rides_unmatched"] == graph.graph["transit"]["n_surface_rides_unmatched"]


def test_aggregate_is_unchanged_without_a_transit_graph() -> None:
    """`_expand_transit` is a no-op for walk/drive runs: same paths, no layer map."""
    from segx.orchestrator import _expand_transit

    state, settings, graph = _transit_state_and_settings()
    routing = _transit_routing(graph, settings)
    original = [list(t.path) for t in routing.trajectories]

    empty = IngestResult(boundary=None, demographics=None, graphs=GraphRegistry(), pois=None, od_pairs=None)
    trajectories, layers = _expand_transit(routing.trajectories, empty)
    assert layers is None
    assert [list(t.path) for t in trajectories] == original


def test_od_mode_aliases_route_public_transport_onto_the_transit_graph() -> None:
    """An OD file whose `mode` column says `bus` needs no code change (D7)."""
    from segx.orchestrator import _OD_MODE_ALIAS

    for value in ("bus", "metro", "subway", "pt", "public_transport", "onibus"):
        assert _OD_MODE_ALIAS[value] == "transit"
    assert _OD_MODE_ALIAS["car"] == "drive"


def test_a_disabled_metric_is_not_computed_and_its_supporting_work_is_skipped() -> None:
    """Switching `path_fragmentation` off must skip the per-path accumulator, not
    just the metric's `compute` — rasterising every path is the costly part. The
    other metrics are untouched, and `stats["fragmentation"]` is `None` (not
    measured), never an all-zero audit that would read as "measured, found nothing"."""
    state, settings, graph = _transit_state_and_settings()
    on = aggregate(state, _transit_population(), _transit_routing(graph, settings), settings)

    off_settings = settings._with_overrides({"metrics.disabled": ["path_fragmentation", "cross_mode_exposure"]})
    off = aggregate(state, _transit_population(), _transit_routing(graph, off_settings), off_settings)

    assert "path_fragmentation" in on.metrics and on.path_table is not None
    assert set(on.metrics) - set(off.metrics) == {"path_fragmentation", "cross_mode_exposure"}
    assert off.path_table is None
    assert off.stats["fragmentation"] is None
    assert off.stats["metrics_enabled"] == sorted(off.metrics)
    pd.testing.assert_frame_equal(off.metrics["exposure"], on.metrics["exposure"])
    pd.testing.assert_frame_equal(off.edge_table, on.edge_table)
