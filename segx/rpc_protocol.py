"""Canonical wire-protocol shapes for the JSON-RPC sidecar (:mod:`segx.rpc`).

This module is the **single source of truth** for the message envelope the engine
and the host app exchange. The TypeScript mirror consumed by the Electron app
(`contracts/segx-rpc.ts`) is *generated* from these definitions by
`scripts/gen_rpc_types.py` — so the two sides cannot drift. Edit the types here,
regenerate, commit both.

These are the *envelope* shapes only. The `Settings` config half of the contract
is delivered live and schema-driven: `describe` returns
`Settings.model_json_schema()` under `config_schema`, so the app builds its Run
Configuration form from the engine at runtime rather than hardcoding it.

Nothing at engine runtime imports this module (`segx.rpc` emits plain dicts); it
exists as the typed contract for tooling, tests, and codegen.

NB: this module deliberately does **not** use ``from __future__ import
annotations``. Stringized annotations would stop ``TypedDict`` from computing
``__required_keys__``/``__optional_keys__`` from ``NotRequired[...]``, which the
codegen relies on to mark optional fields. Types are defined leaf-first so eager
evaluation resolves every reference. (Python is pinned to 3.11.)
"""

from typing import Any, Literal, NotRequired, TypedDict

# --------------------------------------------------------------------------- #
# Envelope                                                                     #
# --------------------------------------------------------------------------- #

#: JSON-RPC correlation id echoed on every reply/event for a request.
CorrelationId = str | int | None

#: The methods the sidecar accepts.
Method = Literal["describe", "run_pipeline", "cancel", "ping", "shutdown"]


class Request(TypedDict):
    """One request line, host → engine."""

    id: NotRequired[CorrelationId]
    method: Method
    params: NotRequired[dict[str, Any]]


class ErrorInfo(TypedDict):
    """The typed-error payload (mirrors :meth:`segx.errors.SegXError.to_dict`)."""

    code: str
    message: str
    details: dict[str, Any]


class OkReply(TypedDict):
    """Terminal success line for a request."""

    id: CorrelationId
    ok: Literal[True]
    result: Any


class ErrorReply(TypedDict):
    """Terminal failure line. ``error.code == "cancelled"`` for a user cancel."""

    id: CorrelationId
    ok: Literal[False]
    error: ErrorInfo


# --------------------------------------------------------------------------- #
# Streaming events (engine → host)                                            #
# --------------------------------------------------------------------------- #


class ReadyEvent(TypedDict):
    """Emitted once at startup so the host knows the sidecar is up."""

    event: Literal["ready"]
    engine: str
    version: str


class ProgressEventMsg(TypedDict):
    """A pipeline progress tick. ``index``/``total`` are global across the run."""

    id: CorrelationId
    event: Literal["progress"]
    stage: str
    index: int
    total: int
    fraction: float
    message: str


class LogEventMsg(TypedDict):
    """A forwarded log record (feeds the live-log terminal)."""

    id: CorrelationId
    event: Literal["log"]
    level: str
    message: str
    logger: str
    ts: float


# --------------------------------------------------------------------------- #
# describe — capabilities result                                              #
# --------------------------------------------------------------------------- #


class EngineInfo(TypedDict):
    name: str
    version: str
    python: str
    phase: str


class MetricInfo(TypedDict):
    name: str
    available: bool
    phase: str
    note: NotRequired[str]


class NetworkInfo(TypedDict):
    modes: list[str]
    engines: list[str]


class AttributeInfo(TypedDict):
    name: str
    kind: str
    class_labels: list[str]


class DemographicsDefault(TypedDict):
    attribute_source: str
    attributes: list[AttributeInfo]


class InputLayerInfo(TypedDict):
    required: bool
    formats: list[str]


class DescribeResult(TypedDict):
    """Payload of a ``describe`` reply — the app renders its UI from this."""

    engine: EngineInfo
    pipeline_stages: list[str]
    metrics: list[MetricInfo]
    network: NetworkInfo
    poi_presets: dict[str, Any]
    demographics_default: DemographicsDefault
    config_schema: dict[str, Any]  # Settings.model_json_schema()
    input_layers: dict[str, InputLayerInfo]
    invariants: list[str]


# --------------------------------------------------------------------------- #
# run_pipeline — result                                                       #
# --------------------------------------------------------------------------- #


class RunParams(TypedDict):
    """Params of a ``run_pipeline`` request."""

    boundary_path: str
    demographics_path: str
    od_pairs_path: NotRequired[str | None]
    config_path: NotRequired[str | None]
    overrides: NotRequired[dict[str, Any]]
    output_dir: NotRequired[str]
    cache_dir: NotRequired[str]
    poi_fetcher: NotRequired[str]


class RunCounts(TypedDict):
    demographic_rows: int
    demographic_kind: str
    pois: int
    od_pairs: int
    origins: int
    origins_unresolved: int
    trajectories: int
    segments: int
    partitions: int


class Conservation(TypedDict):
    emitted: float
    deposited: float
    unreachable: float
    ok: bool


class DiversityStat(TypedDict):
    n: int
    min: float
    mean: float
    max: float


class Artifact(TypedDict):
    """One file the P8 writers produced."""

    kind: str  # "segments" (GeoPackage) | "metrics" (XLSX) | "table" (CSV)
    path: str
    layers: NotRequired[list[str]]  # kind == "segments": the line layers
    sheets: NotRequired[list[str]]  # kind == "metrics": the sheets, in order


class WriteStats(TypedDict):
    n_segment_rows: int
    n_segments_no_geometry: int
    rows_per_layer: dict[str, int]


class RunResult(TypedDict):
    """Payload of a successful ``run_pipeline`` reply (the full pipeline, P8)."""

    crs: str
    snapshot_date: str | None
    phase: str
    counts: RunCounts
    reachability: dict[str, Any]
    conservation: Conservation
    diversity: dict[str, DiversityStat]
    od_spread: dict[str, Any] | None
    provenance: dict[str, Any]  # the stamp written into both files, minus the settings dump
    write_stats: WriteStats
    artifacts: list[Artifact]


#: Emission order for codegen (leaf types before the aggregates that use them).
EXPORTED_TYPES: tuple[type, ...] = (
    Request,
    ErrorInfo,
    OkReply,
    ErrorReply,
    ReadyEvent,
    ProgressEventMsg,
    LogEventMsg,
    EngineInfo,
    MetricInfo,
    NetworkInfo,
    AttributeInfo,
    DemographicsDefault,
    InputLayerInfo,
    DescribeResult,
    RunParams,
    RunCounts,
    Conservation,
    DiversityStat,
    Artifact,
    WriteStats,
    RunResult,
)
