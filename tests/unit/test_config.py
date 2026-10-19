"""P0: the configuration object — defaults, validators, and layered resolution."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from segx.config import (
    DEFAULT_POI_PRESETS,
    DemographicAttribute,
    DemographicSchema,
    ODColumnMapping,
    POIBindingConfig,
    Settings,
)
from segx.network.registry import GraphRegistry


def test_default_settings_shape():
    s = Settings()
    # demographic schema: IBGE-flavoured default
    assert s.demographics.attribute_names == ["income", "race", "age"]
    assert s.demographics.polygon_id_column == "id"
    assert s.demographics.population_column == "pop"
    # OD column mapping defaults
    assert s.od_columns.origin_id == "origin_id"
    assert s.od_columns.weight == "weight"
    # POI dual binding
    assert s.poi.nearest.enabled is True
    assert s.poi.catchment.enabled is True
    assert s.poi.catchment.budget_minutes == 30.0
    assert s.poi.catchment.walk_speed_kmh == 5.0
    assert s.poi.catchment.weight_mode == "unit"
    assert set(s.poi.presets) == set(DEFAULT_POI_PRESETS)
    # network
    assert s.network.modes == ["walk", "drive"]


def test_demographic_attribute_kinds_and_labels():
    cat = DemographicAttribute(name="race", kind="categorical", categories=["a", "b", "c"])
    assert cat.class_labels == ["a", "b", "c"]
    assert cat.source_column == "race"  # defaults to name

    binned = DemographicAttribute(name="income", kind="binned", bins=[0.0, 10.0, 20.0])
    assert binned.class_labels == ["[0, 10)", "[10, 20)"]

    num = DemographicAttribute(name="age", kind="numeric", source_column="IDADE")
    assert num.class_labels == ["age"]
    assert num.source_column == "IDADE"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "x", "kind": "categorical"},  # no categories
        {"name": "x", "kind": "categorical", "categories": []},  # empty categories
        {"name": "x", "kind": "binned"},  # no bins
        {"name": "x", "kind": "binned", "bins": [1.0]},  # < 2 edges
        {"name": "x", "kind": "binned", "bins": [0.0, 10.0, 10.0]},  # not ascending
        {"name": "x", "kind": "binned", "bins": [0.0, 20.0, 10.0]},  # not ascending
    ],
)
def test_demographic_attribute_rejects_bad_shapes(kwargs):
    with pytest.raises(ValidationError):
        DemographicAttribute(**kwargs)


def test_demographic_schema_lookup():
    schema = DemographicSchema.ibge_default()
    assert schema.get("race").kind == "categorical"
    assert schema.get("income").kind == "binned"
    with pytest.raises(KeyError):
        schema.get("nope")


def test_od_mode_defaults():
    m = ODColumnMapping()
    assert m.mode_for("commute") == "car"
    assert m.mode_for("school") == "walk"  # falls back


def test_poi_categories_for():
    poi = POIBindingConfig()
    assert set(poi.categories_for("nearest")) == set(DEFAULT_POI_PRESETS)
    assert set(poi.categories_for("catchment")) == set(DEFAULT_POI_PRESETS)
    poi.nearest.categories = ["parks"]
    assert poi.categories_for("nearest") == ["parks"]


def test_catchment_requires_a_budget():
    with pytest.raises(ValidationError):
        POIBindingConfig(catchment={"enabled": True, "budget_minutes": None, "distance_budget_m": None})
    # distance-only budget is fine
    poi = POIBindingConfig(catchment={"enabled": True, "budget_minutes": None, "distance_budget_m": 1500.0})
    assert poi.catchment.distance_budget_m == 1500.0


def test_settings_load_layers_overrides():
    s = Settings.load(overrides={"poi.catchment.budget_minutes": 45, "od_columns.origin_id": "COD_ORIG"})
    assert s.poi.catchment.budget_minutes == 45
    assert s.od_columns.origin_id == "COD_ORIG"


def test_settings_load_bad_override_path():
    with pytest.raises(KeyError):
        Settings.load(overrides={"poi.catchment.does_not_exist": 1})
    with pytest.raises(KeyError):
        Settings.load(overrides={"no_such_section.key": 1})


def test_settings_load_from_toml(tmp_path):
    cfg = tmp_path / "segx.toml"
    cfg.write_text(
        "\n".join(
            [
                "[poi.catchment]",
                "budget_minutes = 20.0",
                "weight_mode = 'inverse_count'",
                "[network]",
                "walk_speed_kmh = 4.5",
            ]
        ),
        encoding="utf-8",
    )
    s = Settings.load(config_path=cfg)
    assert s.poi.catchment.budget_minutes == 20.0
    assert s.poi.catchment.weight_mode == "inverse_count"
    assert s.network.walk_speed_kmh == 4.5
    # untouched values keep their defaults
    assert s.od_columns.origin_id == "origin_id"


def test_settings_roundtrips_through_dump():
    s = Settings()
    again = Settings.model_validate(s.model_dump())
    assert again.model_dump_json() == s.model_dump_json()
    # inf bin edge survives the round trip
    assert math.isinf(again.demographics.get("income").bins[-1])


def test_graph_registry_accepts_transit_and_rejects_unknown_modes():
    """P7 made 'transit' a real mode; an unknown *name* is still refused."""
    reg = GraphRegistry()
    reg.register("walk", object())
    reg.register("transit", object())
    assert "walk" in reg
    assert reg.modes() == ["transit", "walk"]
    with pytest.raises(ValueError):
        reg.register("teleport", object())
    with pytest.raises(ValueError):
        reg.get("teleport")
    # a known mode that simply wasn't built is a *different* failure
    with pytest.raises(KeyError):
        reg.get("drive")


def test_metrics_can_be_switched_off_one_by_one():
    """`metrics.disabled` drops single metrics without restating the rest, is
    applied after `enabled`, and refuses a name the registry does not know — a
    typo there would otherwise leave the metric silently on."""
    from segx.errors import ConfigValueError
    from segx.metrics.registry import available_metrics

    names = available_metrics()
    assert Settings().metrics.active(names) == names  # default: everything runs

    off = Settings.load(overrides={"metrics.disabled": ["path_fragmentation", "exposure"]})
    assert off.metrics.active(names) == [n for n in names if n not in {"path_fragmentation", "exposure"}]

    both = Settings.load(overrides={"metrics.enabled": ["exposure", "diversity"], "metrics.disabled": ["diversity"]})
    assert both.metrics.active(names) == ["exposure"]

    for key in ("metrics.disabled", "metrics.enabled"):
        with pytest.raises(ConfigValueError, match="no_such_metric"):
            Settings.load(overrides={key: ["no_such_metric"]}).metrics.active(names)
