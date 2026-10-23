"""The render_figures stage on the synthetic city: every figure family draws,
lands in its folder, and a broken figure is listed instead of failing the run."""

from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point, box

from segx.config import (
    CatchmentBindingConfig,
    DemographicAttribute,
    DemographicSchema,
    FiguresConfig,
    NearestBindingConfig,
    POIBindingConfig,
    RoutingConfig,
    Settings,
)
from segx.figures import class_colours, render_figures
from segx.io.boundary import LoadedBoundary
from segx.io.pois import LoadedPOIs
from segx.network.registry import GraphRegistry
from segx.orchestrator import aggregate
from segx.routing.inputs import RoutableODPair, RoutableOrigin, RoutablePOI
from segx.routing.nearest_poi import nearest_poi_trajectories
from segx.routing.networkx_engine import NetworkXEngine
from segx.routing.od_router import od_pair_trajectories
from segx.routing.poi_catchment import catchment_trajectories
from segx.validation.synthetic_city import (
    CATCHMENT_BUDGET_STEPS,
    DEMOGRAPHIC_CLASSES,
    EDGE_M,
    GRID_N,
    ORIGIN_DEMOGRAPHICS,
    ORIGIN_NODES,
    POI_CATEGORIES,
    POIS,
    SYNTHETIC_CRS,
    build_grid_graph,
)


def _settings(tmp_path) -> Settings:
    cats = list(POI_CATEGORIES)
    return Settings(
        poi=POIBindingConfig(
            presets={c: {} for c in cats},
            nearest=NearestBindingConfig(categories=cats),
            catchment=CatchmentBindingConfig(
                categories=cats, budget_minutes=None, distance_budget_m=CATCHMENT_BUDGET_STEPS * EDGE_M
            ),
        ),
        routing=RoutingConfig(engine="networkx", access_mode="walk"),
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=list(DEMOGRAPHIC_CLASSES))]
        ),
        figures=FiguresConfig(basemap=False, dpi=50),
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture
def run(tmp_path):
    """A finished run of the synthetic city: nearest + catchment + two OD commutes."""
    settings = _settings(tmp_path)
    graph = nx.MultiDiGraph(build_grid_graph())  # OSMnx-shaped: keyed, both directions
    graph.graph["crs"] = SYNTHETIC_CRS
    graphs = GraphRegistry()
    graphs.register("walk", graph)
    engine = NetworkXEngine({"walk": graph})

    origins = [RoutableOrigin(oid, node, 1.0) for oid, node in ORIGIN_NODES.items()]
    pois: dict[str, list[RoutablePOI]] = {}
    for poi_id, (category, node) in POIS.items():
        pois.setdefault(category, []).append(RoutablePOI(poi_id, node))
    pairs = [
        RoutableODPair("o1", ORIGIN_NODES["o1"], "work_a", 15, "walk", "commute", 12.0),
        RoutableODPair("o2", ORIGIN_NODES["o2"], "work_b", 0, "walk", "commute", 5.0),
    ]
    trajectories = [
        *nearest_poi_trajectories(origins, pois, engine, settings, mode="walk"),
        *catchment_trajectories(origins, pois, engine, settings, mode="walk"),
        *od_pair_trajectories(pairs, engine, settings),
    ]

    xy = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}
    extent = (GRID_N - 1) * EDGE_M
    state = SimpleNamespace(
        graphs=graphs,
        boundary=LoadedBoundary(
            geometry=box(0, 0, extent, extent), crs=SYNTHETIC_CRS, bbox_wgs84=(-75.2, 39.9, -75.1, 40.0)
        ),
        pois=LoadedPOIs(
            gdf=gpd.GeoDataFrame(
                {"category": [c for c, _ in POIS.values()]},
                geometry=[Point(xy[n]) for _, n in POIS.values()],
                crs=SYNTHETIC_CRS,
            )
        ),
        od_pairs=None,
    )
    population = SimpleNamespace(
        origins=gpd.GeoDataFrame(
            {
                "origin_id": list(ORIGIN_NODES),
                "weight": 1.0,
                "demo_vector": [{"income": dict(ORIGIN_DEMOGRAPHICS[o])} for o in ORIGIN_NODES],
                "demo_resolved": True,
            },
            geometry=[Point(xy[n]) for n in ORIGIN_NODES.values()],
            crs=SYNTHETIC_CRS,
        )
    )
    routing = SimpleNamespace(trajectories=trajectories, stats={})
    aggregation = aggregate(state, population, routing, settings)
    return SimpleNamespace(state=state, population=population, routing=routing, aggregation=aggregation), settings


def test_every_figure_family_is_drawn_into_its_folder(run, tmp_path):
    result, settings = run
    events = []
    figs = render_figures(result, settings, tmp_path / "out", progress=events.append)

    assert figs.failed == []
    assert figs.basemap == "off"
    maps = set(figs.files["maps"])
    assert {"overview.png", "nearest.png", "catchment_park.png", "od_all.png"} <= maps
    assert {"od_income_low.png", "od_income_high.png"} <= maps  # one OD map per class
    segments = set(figs.files["segments"])
    for part in ("od-commute", "nearest-park", "catchment-park"):
        assert f"{part}/dominant_income.png" in segments
        assert f"{part}/diversity_income.png" in segments
    assert "od-commute/experienced_segregation_income.png" in segments
    assert "od-commute/exposure_by_group_income.png" in figs.files["charts"]
    assert "od-commute/dominant_class_counts_income.png" in figs.files["charts"]
    assert {"diversity_by_segment.csv", "partition_summary.csv", "exposure_summary.csv"} <= set(figs.files["tables"])
    for folder, names in figs.files.items():
        for name in names:
            path = tmp_path / "out" / folder / name
            assert path.is_file() and path.stat().st_size > 0
    # progress: one event per figure, all on the render_figures stage
    assert {e.stage for e in events} == {"render_figures"}
    assert len(events) == sum(len(v) for k, v in figs.files.items() if k != "tables") + 1


def test_a_broken_figure_is_listed_and_the_rest_still_draw(run, tmp_path, monkeypatch):
    import segx.figures as figures

    def boom(*args, **kwargs):
        raise RuntimeError("no ink")

    monkeypatch.setattr(figures, "_plot_nearest", boom)
    result, settings = run
    figs = render_figures(result, settings, tmp_path / "out")
    assert figs.failed == [{"file": "maps/nearest.png", "error": "RuntimeError: no ink"}]
    assert "nearest.png" not in figs.files["maps"]
    assert "overview.png" in figs.files["maps"]


def test_route_maps_are_capped_by_max_routes(run, tmp_path):
    result, settings = run
    settings = settings.model_copy(update={"figures": FiguresConfig(basemap=False, dpi=50, max_routes=2)})
    figs = render_figures(result, settings, tmp_path / "out")
    assert figs.failed == [] and "nearest.png" in figs.files["maps"]


def test_basemap_failure_turns_it_off_for_the_run(run, tmp_path, monkeypatch):
    import contextily

    calls = []

    def offline(*args, **kwargs):
        calls.append(1)
        raise ConnectionError("no network")

    monkeypatch.setattr(contextily, "add_basemap", offline)
    result, settings = run
    settings = settings.model_copy(update={"figures": FiguresConfig(basemap=True, dpi=50)})
    figs = render_figures(result, settings, tmp_path / "out")
    assert figs.failed == []
    assert figs.basemap == "unavailable"
    assert len(calls) == 1  # tried once, then every other map skipped it
    assert (tmp_path / "cache" / "tiles").is_dir()


def test_class_colours_keep_the_income_palette_and_cycle_otherwise():
    assert class_colours(["low", "mid", "high"])["high"] == "#d73027"
    other = class_colours(["masc", "fem"])
    assert other["masc"] != other["fem"] and "n/a" in other
