"""POI auto-binding (contract item 2.3): extract POIs per category from OSM,
cache as GeoParquet, return one tidy GeoDataFrame the dual binding consumes.

Three concerns, separated:

* **Tag presets → features (per category).** :func:`fetch_pois_quackosm` reads a
  PBF via QuackOSM (the contract default); :func:`fetch_pois_osmnx` uses the
  Overpass API as a fallback when no PBF is available. Both are "live"
  operations — covered by P3+ integration tests, not unit tests.
* **In-memory normalisation.** :func:`build_pois_gdf` takes a ``{category:
  GeoDataFrame}`` dict and produces the canonical ``[category, osm_id,
  geometry, tags]`` table — pure data manipulation, unit-testable.
* **Cache.** :func:`pois_cache_key` keys the on-disk parquet by
  ``(bbox, snapshot_date, sorted(categories))`` so that adding or removing a
  category from the preset list is itself a cache invalidation, not a silent
  reuse of stale data.

**A failed fetch is an error, never an empty category.** "No park in this area"
and "Overpass timed out while looking for parks" both arrive as zero features,
and only the first may be cached: the second, once written, is read back as "no
POIs here" by every later run without a word (invariants #2 and #4). So
:func:`fetch_pois_osmnx` raises :class:`~segx.errors.POIFetchError` on anything
but a clean "no matching features" answer — including the case Overpass reports
with HTTP 200 and a ``remark`` (a server-side timeout or memory limit), which
OSMnx only logs — and :func:`load_pois` writes the cache only after a fetch that
did not raise.

QuackOSM is the contract-preferred backend (multithreaded DuckDB-backed PBF
reader, GeoParquet-native, 2–10× faster than OSMnx for bulk features). The
OSMnx fallback exists so that a contributor without a local PBF can still
exercise the pipeline against Overpass — slower, rate-limited, but always
available.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd

from segx.errors import ConfigValueError, OSMFetchError, POIFetchError
from segx.network.retry import with_overpass_retries

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings
    from segx.io.boundary import LoadedBoundary
    from segx.progress import CancelCheck


@dataclass(frozen=True, slots=True)
class LoadedPOIs:
    """Validated POIs ready for the routing layer.

    Columns on :attr:`gdf`:

    * ``category`` — one of the keys in ``settings.poi.presets`` (str);
    * ``osm_id`` — best-effort OSM identifier (may be ``None``);
    * ``geometry`` — Point in the working CRS (non-point features are
      reduced to their centroid; the reduction is logged in ``geom_source``);
    * ``geom_source`` — ``"point"`` if the raw feature was already a point,
      ``"centroid"`` if reduced from a polygon/line;
    * ``tags`` — the raw OSM tag dict serialised as JSON (kept for downstream
      filters; deserialise with ``json.loads`` when needed). JSON-on-disk
      round-trips cleanly through Parquet for any tag shape, including empty.
    """

    gdf: gpd.GeoDataFrame


# --------------------------------------------------------------------------- #
# Cache key                                                                    #
# --------------------------------------------------------------------------- #

_BBOX_ROUND_DECIMALS = 6


def pois_cache_key(
    bbox: tuple[float, float, float, float],
    snapshot_date: str | None,
    categories: list[str],
) -> str:
    """Deterministic 16-char hex digest. Different category sets → different keys."""
    payload = json.dumps(
        {
            "bbox": [round(c, _BBOX_ROUND_DECIMALS) for c in bbox],
            "snapshot_date": snapshot_date,
            "categories": sorted(categories),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pois_cache_path(cache_dir: Path, key: str) -> Path:
    """Where the GeoParquet for *key* would live."""
    return Path(cache_dir) / "pois" / f"{key}.parquet"


# --------------------------------------------------------------------------- #
# In-memory normalisation                                                      #
# --------------------------------------------------------------------------- #


def build_pois_gdf(
    features_by_category: dict[str, gpd.GeoDataFrame],
    *,
    working_crs: str,
) -> gpd.GeoDataFrame:
    """Concatenate per-category feature tables into the canonical POI layout.

    Each row keeps the category label, the chosen point geometry, an ``osm_id``
    if discoverable, and a ``tags`` dict. Non-point features are reduced to
    their centroid — POIs are point opportunities for routing purposes, even
    when the underlying OSM feature is a park polygon or a bus-route line.
    """
    rows: list[dict[str, Any]] = []
    for category, raw in features_by_category.items():
        if raw is None or len(raw) == 0:
            continue
        if raw.crs is not None and str(raw.crs) != working_crs:
            raw = raw.to_crs(working_crs)
        for idx, feat in raw.iterrows():
            geom = feat.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type == "Point":
                point, source = geom, "point"
            else:
                point, source = geom.centroid, "centroid"
            osm_id = feat.get("osmid") if "osmid" in raw.columns else (feat.get("id") if "id" in raw.columns else idx)
            tags = {k: v for k, v in feat.items() if k not in {"geometry", "osmid", "id"} and pd.notna(v)}
            rows.append(
                {
                    "category": category,
                    "osm_id": str(osm_id) if osm_id is not None else None,
                    "geometry": point,
                    "geom_source": source,
                    "tags": json.dumps(tags, default=str, sort_keys=True),
                }
            )
    if not rows:
        return gpd.GeoDataFrame(
            {"category": [], "osm_id": [], "geom_source": [], "tags": []},
            geometry=gpd.GeoSeries([], crs=working_crs),
            crs=working_crs,
        )
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=working_crs)


# --------------------------------------------------------------------------- #
# Live fetchers (integration territory — P3+)                                  #
# --------------------------------------------------------------------------- #


#: How OSMnx words a query that ran and matched nothing — the one outcome that is
#: a real zero. Matched on the message because OSMnx raises the same exception
#: class for malformed responses; if the wording ever changes, a real zero turns
#: into a loud error rather than a failure turning into a quiet zero.
_NO_FEATURES_MESSAGE = "No matching features"


@contextmanager
def _overpass_remarks(ox):
    """Collect the ``remark`` of every Overpass response received inside the block.

    Overpass reports a server-side timeout or memory limit as HTTP 200 with a
    ``remark`` and no elements. OSMnx logs it and carries on, so the query then
    looks exactly like one that matched nothing. This taps OSMnx's response
    parser for the duration of a fetch to see the remark. It leans on a private
    OSMnx function; if that is ever gone the tap is simply off and the fetch
    behaves as OSMnx alone would.
    """
    remarks: list[str] = []
    http = getattr(ox, "_http", None)
    original = getattr(http, "_parse_response", None)
    if original is None:  # pragma: no cover - guards an OSMnx internal
        yield remarks
        return

    def recording(response):
        payload = original(response)
        if isinstance(payload, dict) and "remark" in payload:
            remarks.append(str(payload["remark"]))
        return payload

    http._parse_response = recording
    try:
        yield remarks
    finally:
        http._parse_response = original


def fetch_pois_osmnx(
    boundary: LoadedBoundary,
    settings: Settings,
    *,
    cancel: CancelCheck | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Live OSM POI fetch via OSMnx (Overpass) for every preset category.

    Returns a per-category dict so :func:`build_pois_gdf` can normalise.
    Boundary is reprojected to WGS84 for the Overpass query. A category the
    query ran for and found nothing in comes back empty; a query that *failed*
    raises :class:`~segx.errors.POIFetchError` (see the module docstring).
    Transient server errors are first retried with backoff
    (:mod:`segx.network.retry`) and end in :class:`~segx.errors.OSMFetchError`.
    """
    import osmnx as ox

    wgs_boundary = gpd.GeoSeries([boundary.geometry], crs=boundary.crs).to_crs(settings.crs.geographic_crs).iloc[0]
    out: dict[str, gpd.GeoDataFrame] = {}
    for category in settings.poi.categories_for("nearest"):
        tags = settings.poi.presets[category]
        failure: Exception | None = None
        with _overpass_remarks(ox) as remarks:
            try:
                out[category] = with_overpass_retries(
                    lambda tags=tags: ox.features_from_polygon(wgs_boundary, tags=tags),
                    what=f"the {category!r} POIs",
                    settings=settings,
                    cancel=cancel,
                )
            except OSMFetchError:
                raise  # the server stayed unavailable through every retry: already typed
            except Exception as exc:  # noqa: BLE001 — classified just below
                failure = exc
        if remarks:
            raise POIFetchError(
                f"Overpass did not complete the POI query for category {category!r}: {remarks[0]}. "
                "Nothing was cached; re-run to retry.",
                details={"category": category, "remarks": remarks},
            ) from failure
        if failure is None:
            continue
        if str(failure).startswith(_NO_FEATURES_MESSAGE):
            out[category] = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")  # a real zero
            continue
        raise POIFetchError(
            f"POI fetch failed for category {category!r}: {type(failure).__name__}: {failure}. "
            "Nothing was cached; re-run to retry.",
            details={"category": category, "error": type(failure).__name__},
        ) from failure
    return out


def fetch_pois_quackosm(
    boundary: LoadedBoundary,
    settings: Settings,
) -> dict[str, gpd.GeoDataFrame]:
    """Live POI fetch via QuackOSM from the PBF at ``network.osm_pbf_url``.

    Requires ``settings.network.osm_pbf_url`` to point to a local PBF or HTTPS URL.
    """
    if settings.network.osm_pbf_url is None:
        raise ConfigValueError(
            "fetch_pois_quackosm requires `network.osm_pbf_url` to be set; "
            "either configure one or use fetch_pois_osmnx."
        )
    from quackosm import convert_geometry_to_geodataframe

    wgs_boundary = gpd.GeoSeries([boundary.geometry], crs=boundary.crs).to_crs(settings.crs.geographic_crs).iloc[0]
    out: dict[str, gpd.GeoDataFrame] = {}
    for category in settings.poi.categories_for("nearest"):
        tags = settings.poi.presets[category]
        out[category] = convert_geometry_to_geodataframe(
            geometry_filter=wgs_boundary,
            tags_filter=tags,
            pbf_path=settings.network.osm_pbf_url,
        )
    return out


# --------------------------------------------------------------------------- #
# Cache-aware orchestration                                                    #
# --------------------------------------------------------------------------- #


def load_pois(
    boundary: LoadedBoundary,
    settings: Settings,
    *,
    cache_dir: Path | None = None,
    fetcher: str = "osmnx",
    cancel: CancelCheck | None = None,
) -> LoadedPOIs:
    """Cache-aware POI extraction. Reads parquet if cached, else fetches.

    ``fetcher`` ∈ ``{"osmnx", "quackosm"}``. The plan's default is QuackOSM
    (faster, GeoParquet-native), but it requires a PBF source; OSMnx via
    Overpass is the always-available fallback.

    The cache is written only after the fetch returned: a fetch that raises
    leaves no entry behind, so the next run asks again instead of reading a
    failure back as an empty area.
    """
    cache_dir = cache_dir or settings.cache_dir
    categories = settings.poi.categories_for("nearest")
    key = pois_cache_key(boundary.bbox_wgs84, settings.network.osm_snapshot_date, categories)
    cache_path = pois_cache_path(cache_dir, key)

    if cache_path.exists():
        gdf = gpd.read_parquet(cache_path)
        if gdf.crs is not None and str(gdf.crs) != boundary.crs:
            gdf = gdf.to_crs(boundary.crs)
        return LoadedPOIs(gdf=gdf)

    if fetcher == "osmnx":
        per_category = fetch_pois_osmnx(boundary, settings, cancel=cancel)
    elif fetcher == "quackosm":
        per_category = fetch_pois_quackosm(boundary, settings)
    else:
        raise ConfigValueError(
            f"unknown POI fetcher {fetcher!r}; expected 'osmnx' or 'quackosm'",
            details={"fetcher": fetcher, "expected": ["osmnx", "quackosm"]},
        )

    gdf = build_pois_gdf(per_category, working_crs=boundary.crs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_path)
    return LoadedPOIs(gdf=gdf)


__all__ = [
    "LoadedPOIs",
    "build_pois_gdf",
    "fetch_pois_osmnx",
    "fetch_pois_quackosm",
    "load_pois",
    "pois_cache_key",
    "pois_cache_path",
]
