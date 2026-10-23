"""On-disk cache for raw OSM extracts, keyed by ``(bbox, snapshot_date, network_type)``.

The cache exists to make academic results reproducible:
re-running months later must not silently pick up a different OSM state. Each
entry stores the graph as GeoParquet (one ``nodes.parquet`` + one
``edges.parquet`` per mode) alongside a small ``manifest.json`` recording the
bbox, snapshot date, network type, and creation timestamp — that manifest is
what propagates into the deliverable's metadata.

This module owns:

* **key derivation** — deterministic hash of ``(rounded_bbox, snapshot_date,
  network_type)`` so equal inputs always hit the same entry;
* **path layout** — one folder per key under ``cache_dir / "osm" / <key>``;
* **read/write of GeoParquet** — the OSMnx ``graph_to_gdfs`` /
  ``graph_from_gdfs`` round-trip is the canonical serialisation.

The actual *fetch* (live OSMnx call or QuackOSM PBF read) lives in
:mod:`segx.network.builder`; this file is pure cache plumbing so the policy
("never silently re-download") is enforceable at one seam.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from segx.errors import CacheMissError

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


# bbox coords are rounded to this many decimal degrees before hashing —
# absorbs floating-point drift on identical input boundaries (~ 0.1 m at
# the equator) without collapsing meaningfully different bboxes.
_BBOX_ROUND_DECIMALS = 6


def _json_encode_lists(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    """JSON-encode every non-geometry column that holds any list cell.

    OSMnx stores some attributes (``osmid``, ``highway``, ``maxspeed``, ``name``,
    ``lanes``, …) as *either* a scalar or a list per row when ways are merged
    during simplification. Arrow/Parquet cannot store a column mixing list and
    scalar values, so such a column is encoded whole as JSON strings (scalars
    too, to keep the column uniform); :func:`_json_decode_columns` reverses it on
    read, giving an exact round-trip (preserves ``maxspeed`` lists, so drive
    travel-times stay reproducible). The encoded column
    names are recorded in the manifest.
    """
    out = gdf.copy()
    geom_name = out.geometry.name
    encoded = [c for c in out.columns if c != geom_name and out[c].map(lambda v: isinstance(v, list)).any()]
    for col in encoded:
        out[col] = out[col].map(_encode_cell)
    return out, encoded


def _encode_cell(value: object) -> str | None:
    if isinstance(value, list):
        return json.dumps(value)
    if value is None or (not isinstance(value, list) and pd.isna(value)):
        return None
    return json.dumps(value)


def _json_decode_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> None:
    """Reverse :func:`_json_encode_lists` in place for the recorded *columns*."""
    for col in columns:
        if col in gdf.columns:
            gdf[col] = gdf[col].map(lambda s: None if (s is None or pd.isna(s)) else json.loads(s))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Paths backing one cached graph extract."""

    root: Path
    nodes_path: Path
    edges_path: Path
    manifest_path: Path

    @property
    def exists(self) -> bool:
        return self.nodes_path.exists() and self.edges_path.exists()


def cache_key(
    bbox: tuple[float, float, float, float],
    snapshot_date: str | None,
    network_type: str,
    *,
    source: str | None = None,
) -> str:
    """Deterministic 16-char hex digest of ``(bbox, snapshot_date, network_type)``.

    ``source`` names a local OSM extract (``None`` = live Overpass); it only
    enters the payload when set, so existing Overpass entries keep their keys.
    """
    rounded = [round(c, _BBOX_ROUND_DECIMALS) for c in bbox]
    fields: dict[str, object] = {"bbox": rounded, "snapshot_date": snapshot_date, "network_type": network_type}
    if source is not None:
        fields["source"] = source
    payload = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_entry(cache_dir: Path, key: str) -> CacheEntry:
    """Locate the paths for a cache entry (does not create them)."""
    root = Path(cache_dir) / "osm" / key
    return CacheEntry(
        root=root,
        nodes_path=root / "nodes.parquet",
        edges_path=root / "edges.parquet",
        manifest_path=root / "manifest.json",
    )


def read_manifest(entry: CacheEntry) -> dict:
    """Return the manifest dict for an entry; raises if the file is missing."""
    if not entry.manifest_path.exists():
        raise CacheMissError(f"manifest missing for cache entry {entry.root}", details={"root": str(entry.root)})
    return json.loads(entry.manifest_path.read_text(encoding="utf-8"))


def write_graph(
    graph: nx.MultiDiGraph,
    entry: CacheEntry,
    *,
    bbox: tuple[float, float, float, float],
    snapshot_date: str | None,
    network_type: str,
    source: str | None = None,
) -> None:
    """Serialise *graph* as GeoParquet + manifest at *entry*.

    Uses OSMnx's canonical ``graph_to_gdfs`` so the round-trip preserves the
    full attribute set that ``graph_from_gdfs`` will need to rebuild the graph.
    """
    import osmnx as ox

    entry.root.mkdir(parents=True, exist_ok=True)
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes, nodes_json_cols = _json_encode_lists(nodes)
    edges, edges_json_cols = _json_encode_lists(edges)
    nodes.to_parquet(entry.nodes_path)
    edges.to_parquet(entry.edges_path)
    manifest = {
        "bbox": list(bbox),
        "snapshot_date": snapshot_date,
        "network_type": network_type,
        "source": source or "overpass",
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "json_columns": {"nodes": nodes_json_cols, "edges": edges_json_cols},
        "created_utc": datetime.now(UTC).isoformat(),
    }
    entry.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_graph(entry: CacheEntry) -> nx.MultiDiGraph:
    """Reconstruct a graph from a cache entry."""
    import osmnx as ox

    if not entry.exists:
        raise CacheMissError(f"cache miss: no parquet files at {entry.root}", details={"root": str(entry.root)})
    nodes = gpd.read_parquet(entry.nodes_path)
    edges = gpd.read_parquet(entry.edges_path)
    if entry.manifest_path.exists():
        json_cols = read_manifest(entry).get("json_columns", {})
        _json_decode_columns(nodes, json_cols.get("nodes", []))
        _json_decode_columns(edges, json_cols.get("edges", []))
    return ox.graph_from_gdfs(nodes, edges)


__all__ = [
    "CacheEntry",
    "cache_entry",
    "cache_key",
    "read_graph",
    "read_manifest",
    "write_graph",
]
