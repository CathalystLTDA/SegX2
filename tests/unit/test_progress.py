"""Progress reporting + cooperative cancellation seam (P0).

The unit-level checks cover :mod:`segx.progress` directly; the integration-ish
check drives ``ingest`` with a recording progress sink and a one-shot cancel to
prove the seam actually threads through the orchestrator.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from segx.config import (
    DemographicAttribute,
    DemographicSchema,
    NearestBindingConfig,
    NetworkConfig,
    POIBindingConfig,
    Settings,
)
from segx.errors import OperationCancelled
from segx.io.pois import pois_cache_key, pois_cache_path
from segx.orchestrator import ingest
from segx.progress import ProgressEvent, check_cancel, report


def test_progress_event_fraction() -> None:
    assert ProgressEvent("s", 0, 4).fraction == pytest.approx(0.25)
    assert ProgressEvent("s", 3, 4).fraction == pytest.approx(1.0)
    assert ProgressEvent("s", 0, 0).fraction == 0.0  # guards div-by-zero


def test_report_emits_and_check_cancel_raises() -> None:
    seen: list[ProgressEvent] = []
    report(seen.append, None, stage="load_boundary", index=0, total=2, message="hi")
    assert seen == [ProgressEvent("load_boundary", 0, 2, "hi")]

    with pytest.raises(OperationCancelled) as ei:
        check_cancel(lambda: True, stage="load_pois")
    assert ei.value.stage == "load_pois"
    assert ei.value.code == "cancelled"

    # a falsey cancel check is a no-op
    check_cancel(lambda: False)


def test_report_honours_cancel_before_emitting() -> None:
    seen: list[ProgressEvent] = []
    with pytest.raises(OperationCancelled):
        report(seen.append, lambda: True, stage="build_networks", index=2, total=4)
    assert seen == []  # cancelled before the event was emitted


# --- seam threads through the orchestrator -------------------------------- #


def _settings(tmp_path) -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="race", kind="categorical", categories=["a", "b"])]
        ),
        network=NetworkConfig(modes=[]),  # no live OSM fetch
        poi=POIBindingConfig(
            presets={"parks": {"leisure": "park"}},
            nearest=NearestBindingConfig(categories=["parks"]),
        ),
        cache_dir=tmp_path / "cache",
    )


def _seed(tmp_path, settings) -> tuple:
    poly = Polygon([(-51.25, -30.05), (-51.15, -30.05), (-51.15, -29.95), (-51.25, -29.95)])
    boundary = tmp_path / "b.gpkg"
    gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326").to_file(boundary, driver="GPKG")
    demo = tmp_path / "d.gpkg"
    gpd.GeoDataFrame(
        {"id": ["s1"], "pop": [100.0], "race_a": [40.0], "race_b": [60.0]},
        geometry=[poly],
        crs="EPSG:4326",
    ).to_file(demo, driver="GPKG")
    key = pois_cache_key(
        (-51.25, -30.05, -51.15, -29.95), settings.network.osm_snapshot_date, settings.poi.categories_for("nearest")
    )
    path = pois_cache_path(settings.cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"category": ["parks"], "osm_id": ["p1"], "geom_source": ["point"], "tags": ["{}"]},
        geometry=[Point(470_500.0, 6_680_500.0)],
        crs="EPSG:32722",
    ).to_parquet(path)
    return boundary, demo


def test_ingest_reports_progress(tmp_path) -> None:
    settings = _settings(tmp_path)
    boundary, demo = _seed(tmp_path, settings)
    events: list[ProgressEvent] = []

    ingest(settings, boundary_path=boundary, demographics_path=demo, progress=events.append)

    stages = [e.stage for e in events]
    assert stages == ["load_boundary", "load_demographics", "build_networks", "load_pois"]
    assert all(e.total == 4 for e in events)
    assert events[-1].fraction == pytest.approx(1.0)


def test_ingest_cancellation_stops_early(tmp_path) -> None:
    settings = _settings(tmp_path)
    boundary, demo = _seed(tmp_path, settings)
    seen: list[str] = []

    # report() checks cancel *before* emitting each stage, so requesting cancel
    # once the first stage has emitted stops the run at the second stage.
    def cancel() -> bool:
        return len(seen) >= 1

    def progress(e: ProgressEvent) -> None:
        seen.append(e.stage)

    with pytest.raises(OperationCancelled) as ei:
        ingest(settings, boundary_path=boundary, demographics_path=demo, progress=progress, cancel=cancel)
    assert seen == ["load_boundary"]  # only the first stage emitted
    assert ei.value.stage == "load_demographics"  # cancelled before the second ran
