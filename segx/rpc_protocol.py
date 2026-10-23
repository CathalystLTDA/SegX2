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
Method = Literal[
    "describe", "run_pipeline", "cancel", "inspect_layer", "get_layer_preview", "od_modes", "ping", "shutdown"
]


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
    description: NotRequired[str]  # plain-language summary (registered metrics)
    formula: NotRequired[str]  # plain text, Unicode math, one expression per line


class TransitInfo(TypedDict):
    """How the ``transit`` mode is sourced (GTFS vs. the OSM fallback)."""

    sources: list[str]
    note: str
    service_window_note: str


class NetworkInfo(TypedDict):
    modes: list[str]
    engines: list[str]
    transit: TransitInfo


class PoiPreset(TypedDict):
    """One POI category: an OSMnx/QuackOSM-style tag filter."""

    tags: dict[str, Any]


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


class PresetAttribute(TypedDict):
    """One demographic attribute of a preset, with every class's source column resolved."""

    name: str
    kind: str  # "categorical" | "binned" | "numeric"
    categories: list[str] | None
    bins: list[float | None] | None  # null first/last edge = open range (JSON has no infinity)
    class_labels: list[str]
    class_columns: list[str]  # one per class, in class order


class PresetDemographics(TypedDict):
    """A full ``DemographicSchema`` as JSON — sendable back as the ``demographics`` override."""

    attribute_source: Literal["demographics", "od"]
    polygon_id_column: str
    population_column: str
    attributes: list[PresetAttribute]


class ODColumns(TypedDict):
    """``ODColumnMapping`` as JSON. A column field set to null = that column / strategy is not used."""

    origin_id: str | None
    dest_id: str | None
    origin_geom: str | None
    origin_wkt: str | None
    origin_lon: str | None
    origin_lat: str | None
    origin_id_ref: str | None
    origin_address: str | None
    dest_geom: str | None
    dest_wkt: str | None
    dest_lon: str | None
    dest_lat: str | None
    dest_id_ref: str | None
    dest_address: str | None
    category: str | None
    mode: str | None
    weight: str | None
    dest_id_precision_m: float
    default_category: str
    default_mode_by_category: dict[str, str]
    fallback_mode: str
    default_weight: float


class InputPreset(TypedDict):
    """A starting point for the host's column mapping (one per data family)."""

    label: str
    demographics: PresetDemographics
    od_columns: ODColumns


class DescribeResult(TypedDict):
    """Payload of a ``describe`` reply — the app renders its UI from this."""

    engine: EngineInfo
    pipeline_stages: list[str]
    metrics: list[MetricInfo]
    network: NetworkInfo
    poi_presets: dict[str, PoiPreset]
    demographics_default: DemographicsDefault
    input_presets: dict[str, InputPreset]  # "ibge", "lodes"
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


class BindingReachability(TypedDict):
    """Reachable vs. unreachable trajectories of one binding, by reason."""

    total: int
    reachable: int
    unreachable: int
    reasons: dict[str, int]


class SnappingAudit(TypedDict):
    """``reachability["snapping"]``: snaps beyond tolerance are flagged, not dropped."""

    origins_over_tolerance: int
    pois_over_tolerance: int
    tolerance_m: float


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
    # one entry per binding, plus the "snapping" audit
    reachability: dict[str, BindingReachability | SnappingAudit]
    conservation: Conservation
    diversity: dict[str, DiversityStat]
    od_spread: dict[str, Any] | None
    provenance: dict[str, Any]  # the stamp written into both files, minus the settings dump
    write_stats: WriteStats
    artifacts: list[Artifact]


# --------------------------------------------------------------------------- #
# inspect_layer / get_layer_preview — display-only layer reads                #
# --------------------------------------------------------------------------- #

#: Geometry kind of a layer; ``TABLE`` = no geometry (e.g. a plain CSV).
GeometryKind = Literal["POINT", "LINE", "POLYGON", "TABLE"]


class InspectLayerParams(TypedDict):
    path: str


class ColumnInfo(TypedDict):
    """One attribute column of an input file, typed the way the loaders read it."""

    name: str
    type: Literal["number", "text", "boolean", "datetime", "other"]
    sample: list[str | float | None]  # up to 3 values


class InspectLayerResult(TypedDict):
    """Header facts about one input file (no geometry loaded)."""

    crs: str | None  # None = the file declares no CRS
    feature_count: int
    geometry_type: GeometryKind
    bbox: list[float] | None  # WGS84 [minx, miny, maxx, maxy]; None for TABLE
    driver: str
    columns: list[ColumnInfo]
    geometry_column: str | None  # the name the loaders see ("geometry"); None for TABLE
    # OSM extracts (.osm.pbf) only: the header's snapshot date ("2026-10-20"); the
    # file is not counted (feature_count 0) and bbox is the header's coverage box.
    snapshot_date: NotRequired[str | None]


class LayerPreviewParams(TypedDict):
    path: str
    max_features: NotRequired[int]


class PreviewFeature(TypedDict):
    """One GeoJSON feature in WGS84, simplified, with a few scalar properties."""

    type: Literal["Feature"]
    geometry: dict[str, Any]
    properties: dict[str, str | float | None]


class LayerPreviewResult(TypedDict):
    """Display-only sample of a layer: the first ``max_features`` features."""

    geometry_type: GeometryKind
    total_features: int
    features: list[PreviewFeature]


class ODModesParams(TypedDict):
    path: str
    mode_column: str


class ODModesResult(TypedDict):
    """How many OD flows name each mode, and the graph each label routes on."""

    counts: dict[str, int]  # label -> flows, e.g. {"car": 562, "bus": 237}
    graph_modes: dict[str, str]  # label -> graph, e.g. {"car": "drive", "bus": "transit"}
    missing: int  # rows with no mode value


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
    TransitInfo,
    NetworkInfo,
    PoiPreset,
    AttributeInfo,
    DemographicsDefault,
    InputLayerInfo,
    PresetAttribute,
    PresetDemographics,
    ODColumns,
    InputPreset,
    DescribeResult,
    RunParams,
    RunCounts,
    BindingReachability,
    SnappingAudit,
    Conservation,
    DiversityStat,
    Artifact,
    WriteStats,
    RunResult,
    InspectLayerParams,
    ColumnInfo,
    InspectLayerResult,
    LayerPreviewParams,
    PreviewFeature,
    LayerPreviewResult,
    ODModesParams,
    ODModesResult,
)
