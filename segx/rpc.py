"""JSON-RPC-over-stdio shim so a host process (the native desktop app) can embed
the engine as a frozen sidecar.

This is the **only** module that speaks the wire protocol; it adds no analysis
logic. It wraps the existing seams — ``orchestrator`` stages,
:mod:`segx.progress` (progress callback + cooperative cancel), and
:mod:`segx.errors` (typed ``SegXError`` with ``.code``/``.to_dict()``) — so the
app talks to the same pipeline the CLI does (invariant #3: the app is just
another producer of :class:`~segx.config.Settings`).

Wire format — **newline-delimited JSON**, one object per line, both directions.

Request (host → engine)::

    {"id": "<corr-id>", "method": "<name>", "params": { ... }}

Reply / events (engine → host), all tagged with the request ``id``::

    {"id": id, "event": "progress", "stage": "route_nearest",
     "index": 8, "total": 14, "fraction": 0.64, "message": "..."}
    {"id": id, "event": "log", "level": "INFO", "message": "...", "logger": "..."}
    {"id": id, "ok": true,  "result": { ... }}          # terminal success
    {"id": id, "ok": false, "error": {"code","message","details"}}  # terminal fault

Methods:

* ``describe``      — capabilities (metrics, POI presets, demographic schema,
                      config JSON-schema, pipeline stages). Sync, no run needed.
* ``run_pipeline``  — run on a worker thread, streaming ``progress``/``log``
                      events, then a terminal ``result``/``error``. Params:
                      ``boundary_path``, ``demographics_path``, optional
                      ``od_pairs_path``, ``config_path``, ``overrides``
                      (dotted-key map), ``output_dir``, ``cache_dir``,
                      ``poi_fetcher``.
* ``cancel``        — set the cancel flag for a running ``runId`` (the
                      ``run_pipeline`` request ``id``); the run unwinds via
                      :class:`~segx.errors.OperationCancelled` and returns a
                      terminal error with ``code == "cancelled"``.
* ``ping`` / ``shutdown`` — liveness / clean stop.

Run:  ``uv run python -m segx.rpc``  (dev)  /  the frozen binary  (prod).

``run_pipeline`` delegates to :func:`segx.orchestrator.run_pipeline` (P8) — the
same entry point the CLI uses — and turns its :class:`~segx.orchestrator.RunResult`
into the JSON reply; ``artifacts`` lists the files the writers produced.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# Importing entropy registers the P4 metrics (``diversity`` etc.) so ``describe``
# reports them as available — same registration side-effect the orchestrator uses.
import segx.metrics.entropy  # noqa: E402,F401
from segx.config import DEFAULT_POI_PRESETS, DemographicSchema, Settings
from segx.errors import OperationCancelled, SegXError
from segx.io.writers import build_provenance
from segx.metrics.registry import available_metrics
from segx.orchestrator import PIPELINE_STAGES, RunResult, run_pipeline
from segx.progress import ProgressEvent

#: Segregation metrics (P5) that aren't registered yet; the
#: app shows them disabled with a reason instead of assuming they exist.
_PLANNED_METRICS = ("exposure", "segregated_trajectories", "dominant_trajectories")

#: Stages only present when OD pairs are supplied.
_OD_ONLY_STAGES = frozenset({"load_od_pairs", "analyse_od_spread", "route_od_pairs"})

_write_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Wire I/O                                                                     #
# --------------------------------------------------------------------------- #


def _json_default(obj: object) -> str:
    """Fallback JSON encoder for the few non-native types that reach the wire."""
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _send(obj: dict[str, Any]) -> None:
    """Write one JSON line to stdout, atomically (worker threads may call this)."""
    line = json.dumps(obj, default=_json_default, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _event(run_id: Any, event: str, **fields: Any) -> None:
    _send({"id": run_id, "event": event, **fields})


def _ok(msg_id: Any, result: Any) -> None:
    _send({"id": msg_id, "ok": True, "result": result})


def _err(msg_id: Any, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    _send({"id": msg_id, "ok": False, "error": {"code": code, "message": message, "details": details or {}}})


def _configure_stdio() -> None:
    """UTF-8, ``\\n``-terminated lines regardless of platform (Windows adds ``\\r``)."""
    for stream in (sys.stdout, sys.stdin):
        # non-reconfigurable stream (e.g. a pipe wrapper in tests) — leave as-is
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# describe — capabilities                                                      #
# --------------------------------------------------------------------------- #


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("segx")
    except Exception:  # pragma: no cover - not installed as a distribution
        return "0.0.0+src"


def _metric_catalog() -> list[dict[str, Any]]:
    """Registered metrics (available) + planned P5 metrics (disabled, with reason)."""
    registered = set(available_metrics())
    catalog = [{"name": name, "available": True, "phase": "P4"} for name in sorted(registered)]
    for name in _PLANNED_METRICS:
        if name not in registered:
            catalog.append(
                {
                    "name": name,
                    "available": False,
                    "phase": "P5",
                    "note": "not registered in this build",
                }
            )
    return catalog


def _describe() -> dict[str, Any]:
    """Everything the app needs to render its UI from the engine, not hardcode it."""
    default_schema = DemographicSchema.ibge_default()
    return {
        "engine": {
            "name": "segx",
            "version": _engine_version(),
            "python": sys.version.split()[0],
            "phase": "P8",  # the whole pipeline runs, writers included
        },
        "pipeline_stages": list(PIPELINE_STAGES),
        "metrics": _metric_catalog(),
        "network": {
            # 'transit' (P7) is the merged walk + public-transport graph. It is
            # offered only when the run enables it (transit.enabled), which the app
            # reads from `config_schema` like every other setting.
            "modes": ["walk", "drive", "transit"],
            "engines": ["pandana", "networkx"],
            "transit": {
                "sources": ["auto", "gtfs", "osm"],
                "note": (
                    "GTFS is exact; the OSM fallback has no timetables, so its headways are "
                    "assumed from config and the run is stamped transit_source='osm_assumed'"
                ),
                "service_window_note": (
                    "headway = window_minutes / n_trips, so a *wider* window raises the average "
                    "headway and makes transit look less competitive: 06:00-09:00 blends the empty "
                    "6 a.m. into the 8 a.m. peak. Prefer running two narrow windows and comparing "
                    "them. `end <= start` means the window crosses midnight, so ('22:00','02:00') "
                    "and the GTFS spelling ('22:00','26:00') are one window; > 4 h warns, "
                    "> 24 h is rejected."
                ),
            },
        },
        "poi_presets": {name: {"tags": tags} for name, tags in DEFAULT_POI_PRESETS.items()},
        "demographics_default": {
            "attribute_source": default_schema.attribute_source,
            "attributes": [
                {"name": a.name, "kind": a.kind, "class_labels": a.class_labels} for a in default_schema.attributes
            ],
        },
        "config_schema": Settings.model_json_schema(),
        "input_layers": {
            "boundary": {"required": True, "formats": ["gpkg", "shp", "geojson"]},
            "demographics": {"required": True, "formats": ["gpkg", "shp", "geojson", "csv"]},
            "od_pairs": {"required": False, "formats": ["csv", "gpkg"]},
        },
        "invariants": [
            "one enforced working CRS (UTM of the boundary); nothing downstream touches lat/lon",
            "'unreachable' is a typed result with a reason, never a silent zero",
            "OSM extracts are cached and pinned by snapshot_date for reproducibility",
            "a transit run built from OSM instead of GTFS is stamped 'osm_assumed' — "
            "invented headways are never presented as measured ones",
        ],
    }


# --------------------------------------------------------------------------- #
# run_pipeline — streamed on a worker thread                                   #
# --------------------------------------------------------------------------- #


def _planned_stages(has_od: bool) -> list[str]:
    """The ordered stages this run will actually execute (for a global % bar)."""
    out = []
    for stage in PIPELINE_STAGES:
        if stage in _OD_ONLY_STAGES and not has_od:
            continue
        out.append(stage)
    return out


class _RunLogHandler(logging.Handler):
    """Forwards log records as ``log`` events for the duration of one run."""

    def __init__(self, run_id: Any) -> None:
        super().__init__(level=logging.INFO)
        self._run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        # logging must never raise into the pipeline
        with contextlib.suppress(Exception):
            _event(
                self._run_id,
                "log",
                level=record.levelname,
                message=record.getMessage(),
                logger=record.name,
                ts=time.time(),
            )


def _assemble_result(settings: Settings, run: RunResult) -> dict[str, Any]:
    """The JSON reply for a finished run: counts, audits, provenance, artifacts.

    Everything here is JSON-native. ``provenance`` is the same stamp the writers
    put in ``segments.gpkg`` / ``metrics.xlsx``, minus the full settings dump (the
    host sent those settings).
    """
    state, population, routing, aggregation, od_spread = (
        run.state,
        run.population,
        run.routing,
        run.aggregation,
        run.od_spread,
    )
    origins = population.origins
    unresolved = int((~origins["demo_resolved"]).sum())
    led = aggregation.ledger
    conservation_ok = abs(led.emitted - led.deposited - led.unreachable) < 1e-6

    diversity_summary: dict[str, Any] = {}
    dv = aggregation.metrics.get("diversity")
    if dv is not None and not dv.empty:
        for binding, sub in dv.groupby("binding"):
            vals = sub["diversity"].dropna()
            if len(vals):
                diversity_summary[str(binding)] = {
                    "n": int(len(vals)),
                    "min": float(vals.min()),
                    "mean": float(vals.mean()),
                    "max": float(vals.max()),
                }

    return {
        "crs": state.boundary.crs,
        "snapshot_date": settings.network.osm_snapshot_date,
        "phase": "P8",
        "counts": {
            "demographic_rows": int(len(state.demographics.gdf)),
            "demographic_kind": state.demographics.kind,
            "pois": int(len(state.pois.gdf)),
            "od_pairs": int(len(state.od_pairs.gdf)) if state.od_pairs else 0,
            "origins": int(len(origins)),
            "origins_unresolved": unresolved,
            "trajectories": int(len(routing.trajectories)),
            "segments": int(aggregation.stats["n_segments"]),
            "partitions": int(len(aggregation.stats["partitions"])),
        },
        "reachability": routing.stats,  # per-binding reachable/unreachable + reasons + snapping audit
        "conservation": {
            "emitted": float(led.emitted),
            "deposited": float(led.deposited),
            "unreachable": float(led.unreachable),
            "ok": conservation_ok,
        },
        "diversity": diversity_summary,
        # P7.5. `null` is the ordinary case, not a failure: the analysis needs
        # per-class counts on each OD flow, which most inputs do not carry.
        "od_spread": (
            {
                "stats": od_spread.stats,
                "summary": od_spread.summary.to_dict(orient="records"),
                "overlap": od_spread.overlap_summary.to_dict(orient="records"),
            }
            if od_spread is not None
            else None
        ),
        "provenance": {
            k: v
            for k, v in build_provenance(
                settings,
                crs=state.boundary.crs,
                graphs=state.graphs,
                stats=aggregation.stats,
            ).items()
            if k != "settings_json"
        },
        "write_stats": run.outputs.stats,
        "artifacts": run.outputs.artifacts(),
    }


def _execute(run_id: Any, params: dict[str, Any], cancel_event: threading.Event) -> None:
    """Run the full pipeline (writers included), streaming progress/log.

    Emits a terminal ``result`` on success, or a terminal ``error`` (code
    ``"cancelled"`` for a user cancel, the ``SegXError.code`` for a typed fault,
    ``"internal"`` otherwise).
    """
    handler = _RunLogHandler(run_id)
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    if prev_level > logging.INFO or prev_level == logging.NOTSET:
        root.setLevel(logging.INFO)

    try:
        od_path = params.get("od_pairs_path")
        planned = _planned_stages(has_od=od_path is not None)
        total = len(planned)

        def on_progress(ev: ProgressEvent) -> None:
            try:
                gi = planned.index(ev.stage)
            except ValueError:  # a stage not in the planned list — fall back to local index
                gi = ev.index
            _event(
                run_id,
                "progress",
                stage=ev.stage,
                index=gi,
                total=total,
                fraction=(gi + 1) / total if total else 0.0,
                message=ev.message,
            )

        cancel = cancel_event.is_set

        settings = Settings.load(params.get("config_path"), params.get("overrides") or None)
        dir_updates: dict[str, Any] = {}
        if params.get("output_dir"):
            dir_updates["output_dir"] = Path(params["output_dir"])
        if params.get("cache_dir"):
            dir_updates["cache_dir"] = Path(params["cache_dir"])
        if dir_updates:
            settings = settings.model_copy(update=dir_updates)

        boundary_path = params["boundary_path"]
        demographics_path = params["demographics_path"]
        cache_dir = Path(params["cache_dir"]) if params.get("cache_dir") else settings.cache_dir
        poi_fetcher = params.get("poi_fetcher", "osmnx")

        run = run_pipeline(
            settings,
            boundary_path=boundary_path,
            demographics_path=demographics_path,
            od_pairs_path=od_path,
            output_dir=settings.output_dir,
            cache_dir=cache_dir,
            poi_fetcher=poi_fetcher,
            progress=on_progress,
            cancel=cancel,
        )

        _ok(run_id, _assemble_result(settings, run))

    except OperationCancelled as exc:
        _err(run_id, "cancelled", exc.message, {"stage": exc.stage})
    except SegXError as exc:
        _err(run_id, exc.code, exc.message, exc.details)
    except KeyError as exc:  # a missing required param (e.g. boundary_path)
        _err(run_id, "rpc.bad_params", f"missing required parameter: {exc}", {})
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak a stack trace as a crash
        _err(run_id, "internal", str(exc), {"traceback": traceback.format_exc()})
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


# --------------------------------------------------------------------------- #
# Dispatch loop                                                                #
# --------------------------------------------------------------------------- #


def serve() -> None:
    """Read requests from stdin until EOF or ``shutdown``; dispatch each one."""
    _configure_stdio()
    logging.captureWarnings(True)  # route warnings.warn(...) into the log stream
    runs: dict[Any, threading.Event] = {}

    _send({"event": "ready", "engine": "segx", "version": _engine_version()})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _err(None, "rpc.parse_error", f"invalid JSON: {exc}", {})
            continue

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "describe":
            _ok(msg_id, _describe())
        elif method == "ping":
            _ok(msg_id, "pong")
        elif method == "cancel":
            target = params.get("runId")
            event = runs.get(target)
            if event is not None:
                event.set()
            _ok(msg_id, {"cancelling": target, "known": event is not None})
        elif method == "run_pipeline":
            event = threading.Event()
            runs[msg_id] = event

            def _worker(rid: Any = msg_id, p: dict[str, Any] = params, ev: threading.Event = event) -> None:
                try:
                    _execute(rid, p, ev)
                finally:
                    runs.pop(rid, None)

            threading.Thread(target=_worker, name=f"segx-run-{msg_id}", daemon=True).start()
        elif method == "shutdown":
            _ok(msg_id, "bye")
            break
        else:
            _err(msg_id, "rpc.unknown_method", f"unknown method {method!r}", {"method": method})


if __name__ == "__main__":
    serve()
