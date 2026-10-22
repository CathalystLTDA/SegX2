// AUTO-GENERATED — do not edit by hand.
// Source of truth: segx/rpc_protocol.py
// Regenerate: uv run python scripts/gen_rpc_types.py
//
// JSON-RPC-over-stdio protocol for the SegX engine sidecar (segx/rpc.py).


export type CorrelationId = string | number | null;
export type Method = "describe" | "run_pipeline" | "cancel" | "inspect_layer" | "get_layer_preview" | "ping" | "shutdown";

/** One request line, host → engine. */
export interface Request {
  id?: string | number | null;
  method: "describe" | "run_pipeline" | "cancel" | "inspect_layer" | "get_layer_preview" | "ping" | "shutdown";
  params?: Record<string, unknown>;
}

/** The typed-error payload (mirrors :meth:`segx.errors.SegXError.to_dict`). */
export interface ErrorInfo {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

/** Terminal success line for a request. */
export interface OkReply {
  id: string | number | null;
  ok: true;
  result: unknown;
}

/** Terminal failure line. ``error.code == "cancelled"`` for a user cancel. */
export interface ErrorReply {
  id: string | number | null;
  ok: false;
  error: ErrorInfo;
}

/** Emitted once at startup so the host knows the sidecar is up. */
export interface ReadyEvent {
  event: "ready";
  engine: string;
  version: string;
}

/** A pipeline progress tick. ``index``/``total`` are global across the run. */
export interface ProgressEventMsg {
  id: string | number | null;
  event: "progress";
  stage: string;
  index: number;
  total: number;
  fraction: number;
  message: string;
}

/** A forwarded log record (feeds the live-log terminal). */
export interface LogEventMsg {
  id: string | number | null;
  event: "log";
  level: string;
  message: string;
  logger: string;
  ts: number;
}

export interface EngineInfo {
  name: string;
  version: string;
  python: string;
  phase: string;
}

export interface MetricInfo {
  name: string;
  available: boolean;
  phase: string;
  note?: string;
}

/** How the ``transit`` mode is sourced (GTFS vs. the OSM fallback). */
export interface TransitInfo {
  sources: string[];
  note: string;
  service_window_note: string;
}

export interface NetworkInfo {
  modes: string[];
  engines: string[];
  transit: TransitInfo;
}

/** One POI category: an OSMnx/QuackOSM-style tag filter. */
export interface PoiPreset {
  tags: Record<string, unknown>;
}

export interface AttributeInfo {
  name: string;
  kind: string;
  class_labels: string[];
}

export interface DemographicsDefault {
  attribute_source: string;
  attributes: AttributeInfo[];
}

export interface InputLayerInfo {
  required: boolean;
  formats: string[];
}

/** One demographic attribute of a preset, with every class's source column resolved. */
export interface PresetAttribute {
  name: string;
  kind: string;
  categories: string[] | null;
  bins: (number | null)[] | null;
  class_labels: string[];
  class_columns: string[];
}

/** A full ``DemographicSchema`` as JSON — sendable back as the ``demographics`` override. */
export interface PresetDemographics {
  attribute_source: "demographics" | "od";
  polygon_id_column: string;
  population_column: string;
  attributes: PresetAttribute[];
}

/** ``ODColumnMapping`` as JSON. A column field set to null = that column / strategy is not used. */
export interface ODColumns {
  origin_id: string | null;
  dest_id: string | null;
  origin_geom: string | null;
  origin_wkt: string | null;
  origin_lon: string | null;
  origin_lat: string | null;
  origin_id_ref: string | null;
  origin_address: string | null;
  dest_geom: string | null;
  dest_wkt: string | null;
  dest_lon: string | null;
  dest_lat: string | null;
  dest_id_ref: string | null;
  dest_address: string | null;
  category: string | null;
  mode: string | null;
  weight: string | null;
  dest_id_precision_m: number;
  default_category: string;
  default_mode_by_category: Record<string, string>;
  fallback_mode: string;
  default_weight: number;
}

/** A starting point for the host's column mapping (one per data family). */
export interface InputPreset {
  label: string;
  demographics: PresetDemographics;
  od_columns: ODColumns;
}

/** Payload of a ``describe`` reply — the app renders its UI from this. */
export interface DescribeResult {
  engine: EngineInfo;
  pipeline_stages: string[];
  metrics: MetricInfo[];
  network: NetworkInfo;
  poi_presets: Record<string, PoiPreset>;
  demographics_default: DemographicsDefault;
  input_presets: Record<string, InputPreset>;
  config_schema: Record<string, unknown>;
  input_layers: Record<string, InputLayerInfo>;
  invariants: string[];
}

/** Params of a ``run_pipeline`` request. */
export interface RunParams {
  boundary_path: string;
  demographics_path: string;
  od_pairs_path?: string | null;
  config_path?: string | null;
  overrides?: Record<string, unknown>;
  output_dir?: string;
  cache_dir?: string;
  poi_fetcher?: string;
}

export interface RunCounts {
  demographic_rows: number;
  demographic_kind: string;
  pois: number;
  od_pairs: number;
  origins: number;
  origins_unresolved: number;
  trajectories: number;
  segments: number;
  partitions: number;
}

/** Reachable vs. unreachable trajectories of one binding, by reason. */
export interface BindingReachability {
  total: number;
  reachable: number;
  unreachable: number;
  reasons: Record<string, number>;
}

/** ``reachability["snapping"]``: snaps beyond tolerance are flagged, not dropped. */
export interface SnappingAudit {
  origins_over_tolerance: number;
  pois_over_tolerance: number;
  tolerance_m: number;
}

export interface Conservation {
  emitted: number;
  deposited: number;
  unreachable: number;
  ok: boolean;
}

export interface DiversityStat {
  n: number;
  min: number;
  mean: number;
  max: number;
}

/** One file the P8 writers produced. */
export interface Artifact {
  kind: string;
  path: string;
  layers?: string[];
  sheets?: string[];
}

export interface WriteStats {
  n_segment_rows: number;
  n_segments_no_geometry: number;
  rows_per_layer: Record<string, number>;
}

/** Payload of a successful ``run_pipeline`` reply (the full pipeline, P8). */
export interface RunResult {
  crs: string;
  snapshot_date: string | null;
  phase: string;
  counts: RunCounts;
  reachability: Record<string, BindingReachability | SnappingAudit>;
  conservation: Conservation;
  diversity: Record<string, DiversityStat>;
  od_spread: Record<string, unknown> | null;
  provenance: Record<string, unknown>;
  write_stats: WriteStats;
  artifacts: Artifact[];
}

export interface InspectLayerParams {
  path: string;
}

/** One attribute column of an input file, typed the way the loaders read it. */
export interface ColumnInfo {
  name: string;
  type: "number" | "text" | "boolean" | "datetime" | "other";
  sample: (string | number | null)[];
}

/** Header facts about one input file (no geometry loaded). */
export interface InspectLayerResult {
  crs: string | null;
  feature_count: number;
  geometry_type: "POINT" | "LINE" | "POLYGON" | "TABLE";
  bbox: number[] | null;
  driver: string;
  columns: ColumnInfo[];
  geometry_column: string | null;
}

export interface LayerPreviewParams {
  path: string;
  max_features?: number;
}

/** One GeoJSON feature in WGS84, simplified, with a few scalar properties. */
export interface PreviewFeature {
  type: "Feature";
  geometry: Record<string, unknown>;
  properties: Record<string, string | number | null>;
}

/** Display-only sample of a layer: the first ``max_features`` features. */
export interface LayerPreviewResult {
  geometry_type: "POINT" | "LINE" | "POLYGON" | "TABLE";
  total_features: number;
  features: PreviewFeature[];
}
