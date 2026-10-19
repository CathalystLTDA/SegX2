"""Typer command-line interface.

Resolves one :class:`~segx.config.Settings` object by layering — built-in
defaults → ``--config`` TOML file → repeated ``--set dotted.key=value`` flags
(values are parsed as JSON when possible, else kept as strings) — and hands it to
the orchestrator. At P0, ``version`` and ``show-config`` are fully working;
``run`` builds the config and then hits the (not-yet-implemented) pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from segx import __version__
from segx.config import Settings

app = typer.Typer(
    name="segx",
    help="SegX - engine for Segregacao Espacial Dinamica (dynamic spatial segregation).",
    no_args_is_help=True,
    add_completion=False,
)


def _parse_overrides(items: list[str] | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects 'dotted.key=value', got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        try:
            value: object = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[key] = value
    return overrides


def _load_settings(config: Path | None, set_: list[str] | None) -> Settings:
    return Settings.load(config_path=config, overrides=_parse_overrides(set_))


ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="TOML config file."),
]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override one config value: 'dotted.key=value'. Repeatable."),
]


@app.command()
def version() -> None:
    """Print the SegX version."""
    typer.echo(__version__)


@app.command("show-config")
def show_config(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Resolve and print the effective configuration (defaults -> file -> --set)."""
    settings = _load_settings(config, set_)
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def run(
    boundary: Annotated[Path, typer.Option("--boundary", "-b", exists=True, help="Study-area polygon.")],
    demographics: Annotated[
        Path, typer.Option("--demographics", "-d", exists=True, help="Census/sector polygons or residence points.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("out"),
    od: Annotated[
        Path | None,
        typer.Option("--od", exists=True, help="Optional OD-pairs file (CSV/GPKG). Columns mapped via config."),
    ] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Run the full pipeline (ingest -> route -> aggregate -> metrics -> export)."""
    from segx.orchestrator import run_pipeline  # local import: keeps `version` cheap

    settings = _load_settings(config, set_)
    result = run_pipeline(
        settings,
        boundary_path=boundary,
        demographics_path=demographics,
        od_pairs_path=od,
        output_dir=out,
    )
    outputs = result.outputs
    typer.echo(f"wrote {outputs.segments_path}  (layers: {', '.join(outputs.layers) or 'none'})")
    typer.echo(f"wrote {outputs.workbook_path}  ({len(outputs.sheets)} sheets)")
    for path in outputs.extra_paths:
        typer.echo(f"wrote {path}")
    if outputs.stats["n_segments_no_geometry"]:
        typer.echo(f"warning: {outputs.stats['n_segments_no_geometry']} segment(s) written without geometry", err=True)


@app.command("od-report")
def od_report(
    demographics: Annotated[
        Path, typer.Option("--demographics", "-d", exists=True, help="Geometry/ID lookup for the OD endpoints.")
    ],
    od: Annotated[Path, typer.Option("--od", exists=True, help="OD-pairs file (CSV/GPKG).")],
    boundary: Annotated[
        Path | None,
        typer.Option("--boundary", "-b", exists=True, help="Study-area polygon. Default: the demographics extent."),
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write the three tables here as CSV.")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
) -> None:
    """Destination spread per demographic class, read from the OD table alone (P7.5).

    Groups the OD file by origin and reports how many destinations each class
    actually reaches, how concentrated that reach is, and - pairwise - whether
    two classes go to the *same* places. No street network is fetched and nothing
    is routed, so this runs in seconds on a new OD file, before committing to a
    full run.
    """
    from segx.analysis.od_spread import analyse_od_spread
    from segx.io.boundary import load_boundary
    from segx.io.demographics import load_demographics
    from segx.io.od_pairs import load_od_pairs

    settings = _load_settings(config, set_)
    working_crs = load_boundary(boundary or demographics, settings).crs
    loaded_demographics = load_demographics(demographics, settings, working_crs=working_crs)
    od_pairs = load_od_pairs(od, settings, working_crs=working_crs, demographics=loaded_demographics)
    result = analyse_od_spread(od_pairs, settings)

    typer.echo(json.dumps(result.stats, indent=2, default=str))
    typer.echo("")
    typer.echo(result.summary.to_string(index=False))
    if not result.overlap_summary.empty:
        typer.echo("")
        typer.echo(result.overlap_summary.to_string(index=False))
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        result.per_origin.to_csv(out / "od_spread_by_origin.csv", index=False)
        result.summary.to_csv(out / "od_spread_summary.csv", index=False)
        result.overlap.to_csv(out / "od_spread_overlap_by_origin.csv", index=False)
        result.overlap_summary.to_csv(out / "od_spread_overlap_summary.csv", index=False)
        typer.echo(f"\nwrote 4 CSVs to {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
