"""P0: CLI smoke tests — `version` and `show-config` are fully working now."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from segx import __version__
from segx.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_show_config_is_valid_json_with_defaults():
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["poi"]["catchment"]["budget_minutes"] == 30.0
    assert [a["name"] for a in data["demographics"]["attributes"]] == ["income", "race", "age"]


def test_show_config_applies_overrides():
    result = runner.invoke(app, ["show-config", "--set", "poi.catchment.budget_minutes=45"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["poi"]["catchment"]["budget_minutes"] == 45


def test_show_config_rejects_unknown_override():
    result = runner.invoke(app, ["show-config", "--set", "poi.catchment.nope=1"])
    assert result.exit_code != 0


def test_run_reaches_pipeline_and_errors_cleanly(tmp_path):
    # Both files must exist for Typer's `exists=True` to pass; content can be
    # invalid because P1 ingest is now real — load_boundary will reject the
    # empty file, exercising the CLI → orchestrator → loader plumbing.
    boundary = tmp_path / "boundary.gpkg"
    demographics = tmp_path / "demo.gpkg"
    boundary.write_bytes(b"")
    demographics.write_bytes(b"")
    result = runner.invoke(app, ["run", "-b", str(boundary), "-d", str(demographics), "-o", str(tmp_path / "out")])
    assert result.exit_code != 0
    # any exception is fine — the test pins that the wiring is reached, not
    # which loader complains first about the empty bytes
    assert result.exception is not None


def test_help_texts_are_ascii():
    # A Windows console on a legacy code page (cp1252, cp850, ...) cannot encode
    # characters like `→` or `—`, and Typer/Rich then crashes on `segx --help`
    # with UnicodeEncodeError — the first command a new user runs.
    import typer.main

    group = typer.main.get_command(app)
    texts = [("segx", group.help or "")]
    for name, command in group.commands.items():
        texts.append((name, command.help or ""))
        texts.extend((f"{name} {param.name}", getattr(param, "help", None) or "") for param in command.params)
    offenders = [(where, text) for where, text in texts if not text.isascii()]
    assert offenders == [], f"non-ASCII help text: {offenders}"
