"""Deliverable writers (P8): ``segments.gpkg`` + ``metrics.xlsx``.

Turns an :class:`~segx.orchestrator.AggregationResult` into the two files the
contract names (clause 2.5) — the GeoPackage QGIS opens, and the workbook the
analyst reads — and stamps both with the run's provenance.

**``segments.gpkg``** has one line layer per ``(mode, layer)``: ``walk_street``,
``drive_street``, ``transit_street``, ``transit_rail_line``. The split on the P7
``layer`` column is the point: a metro segment is a layer of its own and is never
drawn on top of the avenue it runs under. A row is one segment of one
``(category, binding)`` partition, *wide*: the class masses (``{attribute}_{class}``),
``n_paths`` (line thickness), the P7.2 passthroughs ``corridor`` / ``length_m``,
and every enabled **per-segment** metric, one column per value per attribute.
Which metrics are per-segment is read off the frame (it carries ``u, v, key``),
never off a list of names — a new plugin lands in the file with no change here
(invariant #5). Two non-spatial tables ride along: ``segx_metadata`` (the
provenance, key/value) and ``segx_columns`` (what every column of the line layers
is). The provenance is also written as each layer's GDAL metadata, which is what
QGIS shows under *Layer Properties → Metadata*.

**``metrics.xlsx``**: the network-wide summary of every enabled metric, the
per-group tables (``exposure``, ``cross_mode_exposure``, ``path_fragmentation``,
…), the P7.5 ``od_spread`` summaries, and the audits — conservation ledger,
reachability and snapping, the cross-mode join (with the per-mode-set breakdown),
fragmentation coverage, and what the writer itself could not place. The
per-origin ``od_spread`` tables can exceed a sheet's row limit at city scale, so
they go to CSV beside the workbook.

Provenance is what invariants #4 and #7 were written for: ``osm_snapshot_date``,
``transit_source`` (``gtfs`` vs ``osm_assumed``) with the normalised service
window, the street-space matching parameters, the fragmentation ladder, and
``metrics_enabled`` — a switched-off metric has no column and no sheet, and the
metadata says it was not run rather than leaving a reader to guess at missing
data. The fully resolved :class:`~segx.config.Settings` is stamped too, so a file
can be reproduced from itself.

A segment whose geometry no graph holds is **kept** with an empty geometry and
counted (``n_segments_no_geometry``), never dropped (invariant #2).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from segx.errors import OutputColumnCollisionError, OutputError
from segx.io.demographics import _slug

if TYPE_CHECKING:  # pragma: no cover
    import geopandas as gpd
    import pandas as pd

    from segx.config import Settings
    from segx.network.registry import GraphRegistry

#: A segment of one partition — the grain of a ``segments.gpkg`` row.
SEGMENT_KEYS: tuple[str, ...] = ("mode", "category", "binding", "u", "v", "key")
#: Edge-table columns that are properties of the edge, carried through unchanged.
PASSTHROUGH_COLUMNS: tuple[str, ...] = ("layer", "corridor", "length_m", "n_paths")

SEGMENTS_FILENAME = "segments.gpkg"
WORKBOOK_FILENAME = "metrics.xlsx"
METADATA_TABLE = "segx_metadata"
COLUMNS_TABLE = "segx_columns"
OD_SPREAD_BY_ORIGIN_CSV = "od_spread_by_origin.csv"
OD_SPREAD_OVERLAP_BY_ORIGIN_CSV = "od_spread_overlap_by_origin.csv"

#: Excel's hard limit on a sheet name.
_SHEET_NAME_MAX = 31
_SHEET_FORBIDDEN = re.compile(r"[\[\]:*?/\\]")
#: Provenance keys too large for per-layer GDAL metadata (they stay in the table).
_TABLE_ONLY_KEYS = frozenset({"settings_json"})


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WrittenOutputs:
    """What :func:`write_outputs` put on disk.

    ``layers`` are the GeoPackage's line layers (the metadata/columns tables are
    not listed), ``sheets`` the workbook's sheets in order, ``extra_paths`` any
    side-car CSVs. ``stats`` is the writer's own audit — rows per layer and how
    many segments had no resolvable geometry.
    """

    output_dir: Path
    segments_path: Path
    workbook_path: Path
    layers: list[str]
    sheets: list[str]
    extra_paths: list[Path] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def artifacts(self) -> list[dict[str, Any]]:
        """JSON-native listing for the RPC result."""
        out: list[dict[str, Any]] = [
            {"kind": "segments", "path": str(self.segments_path), "layers": list(self.layers)},
            {"kind": "metrics", "path": str(self.workbook_path), "sheets": list(self.sheets)},
        ]
        out.extend({"kind": "table", "path": str(p)} for p in self.extra_paths)
        return out


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #


def _engine_version() -> str:
    try:
        from importlib.metadata import version

        return version("segx")
    except Exception:  # pragma: no cover - not installed as a distribution
        from segx import __version__

        return __version__


def build_provenance(
    settings: Settings,
    *,
    crs: str | None,
    graphs: GraphRegistry | None,
    stats: dict,
    written_at: datetime | None = None,
) -> dict[str, Any]:
    """Every stamp a reader needs to tell two runs' numbers apart.

    Nested values are kept as they are here; :func:`flatten_provenance` renders
    them for the file formats.
    """
    from segx.network.builder import osm_snapshot_date, osm_source

    transit_meta = None
    if graphs is not None and "transit" in graphs:
        meta = dict(graphs.get("transit").graph.get("transit", {}))
        # the per-stop list is an audit, not a stamp — its count stays
        meta.pop("stops_unreachable", None)
        transit_meta = meta or None

    cross_mode = stats.get("cross_mode") or {}
    street_space = cross_mode.get("street_space") or None
    fragmentation = stats.get("fragmentation")
    route_set = settings.routing.route_set

    return {
        "engine": "segx",
        "engine_version": _engine_version(),
        "written_at": (written_at or datetime.now(UTC)).isoformat(timespec="seconds"),
        "working_crs": crs,
        "osm_snapshot_date": osm_snapshot_date(settings),
        "osm_source": osm_source(settings) or "overpass",
        "network_modes": graphs.modes() if graphs is not None else [],
        "network_undirected": settings.network.undirected,
        "routing_engine": settings.routing.engine,
        "routing_access_mode": settings.routing.access_mode,
        "route_set_k": route_set.k,
        "route_set_method": route_set.method if route_set.k > 1 else None,
        "route_set_weight_scheme": route_set.weight_scheme if route_set.k > 1 else None,
        "metrics_enabled": list(stats.get("metrics_enabled", [])),
        "metrics_disabled": list(settings.metrics.disabled),
        # Invariant #7: `osm_assumed` and a measured headway are indistinguishable
        # once they are rows, so the source is stamped at the top level.
        "transit_source": transit_meta.get("transit_source") if transit_meta else None,
        "transit": transit_meta,
        "street_space_params": street_space.get("params") if street_space else None,
        "fragmentation_params": fragmentation.get("params") if fragmentation else None,
        "settings_json": settings.model_dump_json(),
    }


def _render(value: Any) -> str:
    """One provenance/audit value as text: strings verbatim, the rest as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return json.dumps(value, default=str, sort_keys=True)


def flatten_provenance(prov: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """``(dotted.key, text)`` rows. Dicts are flattened; lists stay JSON."""
    rows: list[tuple[str, str]] = []
    for key, value in prov.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            rows.extend(flatten_provenance(value, prefix=f"{dotted}."))
        else:
            rows.append((dotted, _render(value)))
    return rows


def _key_value_frame(prov: dict[str, Any]) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(flatten_provenance(prov), columns=["key", "value"])


# --------------------------------------------------------------------------- #
# The wide segment table                                                       #
# --------------------------------------------------------------------------- #


def _is_segment_frame(frame: pd.DataFrame) -> bool:
    return {"u", "v", "key"}.issubset(frame.columns)


def _metric_value_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in SEGMENT_KEYS and c != "attribute"]


def _metric_column_name(metric: str, value_col: str, attribute: str) -> str:
    """``diversity_income``; ``dominant_trajectories_dominance_income``.

    The metric prefix is dropped only when the value column *is* the metric's
    name. Prefixing everything else keeps two metrics' ``dominant_class`` apart
    (``dominant_trajectories`` takes the class with the most mass,
    ``segregated_trajectories`` the one with the highest location quotient).
    """
    base = value_col if value_col == metric else f"{metric}_{value_col}"
    return f"{base}_{_slug(attribute)}"


def segment_table(edge_table: pd.DataFrame, metrics: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The wide per-segment frame (no geometry yet) and its column dictionary.

    One row per :data:`SEGMENT_KEYS`. Mass is pivoted on ``(attribute, class)``;
    each per-segment metric is pivoted on ``attribute``. Merges are left joins on
    the edge table, so a segment a metric reports as ``NaN`` stays ``NaN`` and a
    segment a metric omitted is ``NaN`` too — never ``0`` (invariant #2).
    """
    import pandas as pd

    keys = list(SEGMENT_KEYS)
    doc: list[dict[str, Any]] = [
        {"column": k, "kind": "key", "metric": None, "source": k, "attribute": None, "class": None} for k in keys
    ]
    if edge_table.empty:
        return pd.DataFrame(columns=[*keys, *PASSTHROUGH_COLUMNS]), pd.DataFrame(doc)

    passthrough = [c for c in PASSTHROUGH_COLUMNS if c in edge_table.columns]
    base = edge_table.drop_duplicates(keys)[[*keys, *passthrough]].reset_index(drop=True)
    doc.extend(
        {"column": c, "kind": "edge", "metric": None, "source": c, "attribute": None, "class": None}
        for c in passthrough
    )

    mass = edge_table.pivot_table(index=keys, columns=["attribute", "class"], values="mass", aggfunc="sum", sort=False)
    mass_names = []
    for attribute, label in mass.columns:
        name = f"{_slug(attribute)}_{_slug(str(label))}"
        mass_names.append(name)
        doc.append(
            {"column": name, "kind": "mass", "metric": None, "source": "mass", "attribute": attribute, "class": label}
        )
    mass.columns = mass_names
    wide = base.merge(mass.reset_index(), on=keys, how="left")

    for metric, frame in metrics.items():
        if not _is_segment_frame(frame) or frame.empty:
            continue
        value_cols = _metric_value_columns(frame)
        for attribute, sub in frame.groupby("attribute", sort=False):
            renamed = {c: _metric_column_name(metric, c, str(attribute)) for c in value_cols}
            part = sub[[*keys, *value_cols]].rename(columns=renamed)
            wide = wide.merge(part, on=keys, how="left")
            doc.extend(
                {
                    "column": renamed[c],
                    "kind": "metric",
                    "metric": metric,
                    "source": c,
                    "attribute": attribute,
                    "class": None,
                }
                for c in value_cols
            )

    columns = pd.DataFrame(doc)
    duplicated = columns.loc[columns["column"].duplicated(keep=False), "column"].unique().tolist()
    if duplicated or len(set(wide.columns)) != len(wide.columns):
        raise OutputColumnCollisionError(
            f"output column name(s) {duplicated} produced twice — an attribute, class or metric name collides; "
            "rename one in the demographic schema",
            details={"columns": duplicated},
        )
    return wide, columns


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #


def _edge_line(graph, u: int, v: int, key: int):
    from shapely.geometry import LineString

    data = graph.get_edge_data(u, v, key) or {}
    geom = data.get("geometry")
    if geom is not None:
        return geom
    nu, nv = graph.nodes[u], graph.nodes[v]
    return LineString([(nu["x"], nu["y"]), (nv["x"], nv["y"])])


def _find_edge(graph, u: int, v: int, key: int):
    """The ``(u, v, key)`` of *graph* that the segment row names, or ``None``.

    Segment rows are **undirected** — ``EdgeAccumulator`` keys them as
    ``(min(u, v), max(u, v), key)`` (``canonical_edge``) — so a one-way edge,
    a rail ride for instance, is stored the other way round from how the graph
    holds it.
    """
    if graph.has_edge(u, v, key):
        return u, v, key
    if graph.has_edge(v, u, key):
        return v, u, key
    return None


def resolve_geometries(segments: pd.DataFrame, graphs: GraphRegistry) -> tuple[list, int]:
    """One line per row, looked up in the graph that holds the edge.

    A row's own mode graph is tried first, then the others: after
    ``expand_transit_rides`` a ``transit`` row may be a ``drive``-graph street
    (a map-matched bus ride) and only its ``rail_line`` rows live in the merged
    graph. The test is edge membership (either direction, see
    :func:`_find_edge`), not node membership — the merged graph shares the walk
    graph's node ids. Returns the geometries and how many rows no graph knew
    (kept, with ``None``).
    """
    modes = graphs.modes()
    cache: dict[tuple, Any] = {}
    out: list = []
    n_missing = 0
    for mode, u, v, key in segments[["mode", "u", "v", "key"]].itertuples(index=False, name=None):
        token = (mode, u, v, key)
        if token not in cache:
            geom = None
            for candidate in (mode, *(m for m in modes if m != mode)):
                if candidate not in graphs:
                    continue
                graph = graphs.get(candidate)
                found = _find_edge(graph, u, v, key)
                if found is not None:
                    geom = _edge_line(graph, *found)
                    break
            cache[token] = geom
        geom = cache[token]
        if geom is None:
            n_missing += 1
        out.append(geom)
    return out, n_missing


# --------------------------------------------------------------------------- #
# segments.gpkg                                                                #
# --------------------------------------------------------------------------- #


def layer_name(mode: str, layer: str) -> str:
    return f"{_slug(mode)}_{_slug(layer)}"


def write_segments(
    segments: gpd.GeoDataFrame,
    columns: pd.DataFrame,
    provenance: dict[str, Any],
    path: Path,
) -> list[str]:
    """Write the line layers + the metadata/columns tables. Returns the line layers."""
    import pyogrio

    if path.exists():
        path.unlink()
    layer_meta = {k: v for k, v in flatten_provenance(provenance) if k.split(".")[0] not in _TABLE_ONLY_KEYS}
    layers: list[str] = []
    if not segments.empty:
        for (mode, layer), sub in segments.groupby(["mode", "layer"], sort=True):
            name = layer_name(str(mode), str(layer))
            pyogrio.write_dataframe(
                sub.reset_index(drop=True), path, layer=name, driver="GPKG", layer_metadata=layer_meta
            )
            layers.append(name)
    pyogrio.write_dataframe(_key_value_frame(provenance), path, layer=METADATA_TABLE, driver="GPKG")
    pyogrio.write_dataframe(
        columns.astype(object).where(columns.notna(), None), path, layer=COLUMNS_TABLE, driver="GPKG"
    )
    return layers


# --------------------------------------------------------------------------- #
# metrics.xlsx                                                                 #
# --------------------------------------------------------------------------- #


def _sheet_name(raw: str, taken: set[str]) -> str:
    name = _SHEET_FORBIDDEN.sub("_", raw)[:_SHEET_NAME_MAX]
    candidate, i = name, 2
    while candidate.lower() in taken:
        suffix = f"~{i}"
        candidate = f"{name[: _SHEET_NAME_MAX - len(suffix)]}{suffix}"
        i += 1
    taken.add(candidate.lower())
    return candidate


def _flat_frame(obj: dict) -> pd.DataFrame:
    import pandas as pd

    return pd.DataFrame(flatten_provenance(obj), columns=["key", "value"])


def workbook_sheets(
    *,
    provenance: dict[str, Any],
    columns: pd.DataFrame,
    aggregation,
    routing_stats: dict | None,
    od_spread,
    write_stats: dict,
) -> list[tuple[str, pd.DataFrame]]:
    """The workbook as ``(sheet, frame)`` pairs, in order (names not yet truncated)."""
    import pandas as pd

    from segx.metrics.cross_mode_exposure import cross_mode_combination_counts

    sheets: list[tuple[str, pd.DataFrame]] = [("metadata", _key_value_frame(provenance)), ("columns", columns)]

    for metric, frame in aggregation.metrics.items():
        summary = aggregation.summaries.get(metric)
        if _is_segment_frame(frame):
            # the per-segment values live in the GeoPackage; the workbook gets the indicator
            if summary is not None:
                sheets.append((metric, summary))
            continue
        sheets.append((metric, frame))
        if summary is not None and not summary.equals(frame):
            sheets.append((f"{metric}_summary", summary))

    if od_spread is not None:
        sheets.append(("od_spread_summary", od_spread.summary))
        sheets.append(("od_spread_overlap", od_spread.overlap_summary))

    stats = aggregation.stats
    ledger = aggregation.ledger
    sheets.append(
        (
            "audit_conservation",
            pd.DataFrame(
                [
                    ("emitted", ledger.emitted),
                    ("deposited", ledger.deposited),
                    ("unreachable", ledger.unreachable),
                    ("balance", ledger.emitted - ledger.deposited - ledger.unreachable),
                ],
                columns=["key", "value"],
            ),
        )
    )
    counts = {
        k: stats[k]
        for k in ("n_reachable", "n_unreachable", "n_unresolved", "n_segments", "edge_mass_total")
        if k in stats
    }
    sheets.append(("audit_counts", _flat_frame(counts)))
    sheets.append(
        ("audit_partitions", pd.DataFrame(stats.get("partitions", []), columns=["mode", "category", "binding"]))
    )
    if routing_stats:
        sheets.append(("audit_reachability", _flat_frame(routing_stats)))
    if stats.get("cross_mode") is not None:
        sheets.append(("audit_cross_mode", _flat_frame(stats["cross_mode"])))
        sheets.append(("audit_cross_mode_sets", cross_mode_combination_counts(aggregation.edge_table)))
    if stats.get("fragmentation") is not None:
        sheets.append(("audit_fragmentation", _flat_frame(stats["fragmentation"])))
    if od_spread is not None:
        sheets.append(("audit_od_spread", _flat_frame(od_spread.stats)))
    sheets.append(("audit_output", _flat_frame(write_stats)))
    return sheets


def write_metrics_workbook(sheets: list[tuple[str, pd.DataFrame]], path: Path) -> list[str]:
    """Write *sheets* to ``metrics.xlsx``. Returns the (Excel-safe) sheet names."""
    import pandas as pd

    taken: set[str] = set()
    names: list[str] = []
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for raw, frame in sheets:
            name = _sheet_name(raw, taken)
            frame.to_excel(writer, sheet_name=name, index=False)
            names.append(name)
    return names


# --------------------------------------------------------------------------- #
# The stage                                                                    #
# --------------------------------------------------------------------------- #


def write_outputs(
    aggregation,
    settings: Settings,
    output_dir: str | Path,
    *,
    graphs: GraphRegistry,
    crs: str | None,
    routing_stats: dict | None = None,
    od_spread=None,
    written_at: datetime | None = None,
) -> WrittenOutputs:
    """The ``write_outputs`` stage: ``segments.gpkg`` + ``metrics.xlsx`` (+ CSVs).

    *crs* is the working CRS the graphs are in (``boundary.crs``); when ``None``
    the first graph's ``crs`` attribute is used.
    """
    import geopandas as gpd
    from pyogrio.errors import DataLayerError, DataSourceError

    PyogrioError = (DataSourceError, DataLayerError)  # noqa: N806 - local alias of an exception tuple
    out = Path(output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(f"cannot create output directory {out}: {exc}", details={"path": str(out)}) from exc

    if crs is None:
        for mode in graphs.modes():
            crs = graphs.get(mode).graph.get("crs")
            if crs is not None:
                break

    provenance = build_provenance(settings, crs=crs, graphs=graphs, stats=aggregation.stats, written_at=written_at)

    wide, columns = segment_table(aggregation.edge_table, aggregation.metrics)
    geometry, n_missing = resolve_geometries(wide, graphs) if not wide.empty else ([], 0)
    segments = gpd.GeoDataFrame(wide, geometry=gpd.GeoSeries(geometry, index=wide.index, crs=crs), crs=crs)

    rows_per_layer: dict[str, int] = {}
    if not segments.empty:
        for (mode, layer), n in segments.groupby(["mode", "layer"]).size().items():
            rows_per_layer[layer_name(str(mode), str(layer))] = int(n)
    write_stats = {
        "n_segment_rows": int(len(segments)),
        "n_segments_no_geometry": int(n_missing),
        "rows_per_layer": rows_per_layer,
    }

    segments_path = out / SEGMENTS_FILENAME
    workbook_path = out / WORKBOOK_FILENAME
    extra: list[Path] = []
    try:
        layers = write_segments(segments, columns, provenance, segments_path)
        sheets = workbook_sheets(
            provenance=provenance,
            columns=columns,
            aggregation=aggregation,
            routing_stats=routing_stats,
            od_spread=od_spread,
            write_stats=write_stats,
        )
        sheet_names = write_metrics_workbook(sheets, workbook_path)
        if od_spread is not None:
            for frame, filename in (
                (od_spread.per_origin, OD_SPREAD_BY_ORIGIN_CSV),
                (od_spread.overlap, OD_SPREAD_OVERLAP_BY_ORIGIN_CSV),
            ):
                frame.to_csv(out / filename, index=False)
                extra.append(out / filename)
    except (OSError, PyogrioError) as exc:  # a locked file (open in Excel/QGIS) is the usual case
        raise OutputError(f"cannot write outputs to {out}: {exc}", details={"path": str(out)}) from exc

    return WrittenOutputs(
        output_dir=out,
        segments_path=segments_path,
        workbook_path=workbook_path,
        layers=layers,
        sheets=sheet_names,
        extra_paths=extra,
        stats=write_stats,
    )


__all__ = [
    "COLUMNS_TABLE",
    "METADATA_TABLE",
    "SEGMENTS_FILENAME",
    "SEGMENT_KEYS",
    "WORKBOOK_FILENAME",
    "WrittenOutputs",
    "build_provenance",
    "flatten_provenance",
    "layer_name",
    "resolve_geometries",
    "segment_table",
    "write_metrics_workbook",
    "write_outputs",
    "write_segments",
]
