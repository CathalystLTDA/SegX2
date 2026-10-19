"""OSM-derived transit feed — the fallback when no GTFS is supplied (P7).

OSM knows where stops are and where metro lines run, but it is a map of
*infrastructure*, not of *service*: it carries **no timetables at all**. A transit
graph without headways cannot answer "is the bus faster than walking?", which is
the entire question a transit router exists to answer. So this module supplies
the geometry and topology OSM does have, and *assumes* the headways from
``transit.osm_assumed_headway_min``.

Because those headways are invented, the resulting feed is stamped
``source = "osm_assumed"`` and that stamp rides through
:meth:`~segx.io.gtfs.LoadedGTFS.metadata` into the output. This is the same
discipline as ``osm_snapshot_date`` and invariant #2: an
approximate result is allowed, but it must be **labelled** — a headway we
invented and a headway we measured look identical in a GeoPackage.

Ordered stop sequences come from OSM ``relation[type=route]`` members. That is
reliable for metro and rail and **patchy for Brazilian buses**,
which is exactly why GTFS is primary.
Relations that yield fewer than two in-boundary stops are dropped.

The output is a :class:`~segx.io.gtfs.LoadedGTFS`, so
:func:`segx.network.transit.build_transit_graph` has **one** input contract and
not two code paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

from segx.errors import TransitDataMissingError
from segx.io.gtfs import (
    LoadedGTFS,
    fill_missing_hop_times,
    overlay_hash,
    read_cached_feed,
    transit_cache_dir,
    transit_cache_key,
    write_cached_feed,
)

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings
    from segx.io.boundary import LoadedBoundary

#: OSM ``route`` relation values we treat as public transport, mapped to the GTFS
#: ``route_type`` the rest of the engine reasons in. The mapping is what decides
#: whether a ride is map-matched onto streets or kept on its own line layer, via
#: ``transit.surface_route_types``.
OSM_ROUTE_TYPES: dict[str, int] = {
    "bus": 3,
    "trolleybus": 11,
    "minibus": 3,
    "share_taxi": 3,
    "tram": 0,
    "light_rail": 0,
    "subway": 1,
    "train": 2,
    "railway": 2,
    "monorail": 12,
    "ferry": 4,
}

#: ``route_type`` → the key looked up in ``transit.osm_assumed_headway_min``.
_HEADWAY_KEY: dict[int, str] = {0: "tram", 1: "subway", 2: "rail", 3: "bus", 4: "rail", 11: "bus", 12: "subway"}

#: Assumed in-vehicle speed per ``route_type``, km/h — used to turn the
#: straight-line distance between consecutive stops into a running time, since
#: OSM has no schedule to read one from. Deliberately modest: a bus that looks
#: too fast would beat walking everywhere and silently inflate transit's share.
_ASSUMED_SPEED_KMH: dict[int, float] = {0: 18.0, 1: 32.0, 2: 45.0, 3: 15.0, 4: 20.0, 11: 15.0, 12: 32.0}

_DEFAULT_HEADWAY_MIN = 20.0


def assumed_headway_min(route_type: int, settings: Settings) -> float:
    """The invented headway for *route_type*, from config (invariant #3)."""
    key = _HEADWAY_KEY.get(int(route_type), "bus")
    table = settings.transit.osm_assumed_headway_min
    return float(table.get(key, table.get("bus", _DEFAULT_HEADWAY_MIN)))


def assumed_speed_kmh(route_type: int) -> float:
    return _ASSUMED_SPEED_KMH.get(int(route_type), _ASSUMED_SPEED_KMH[3])


# --------------------------------------------------------------------------- #
# Live fetch (the only place that hits Overpass)                               #
# --------------------------------------------------------------------------- #


#: Overpass endpoint. Configurable only by monkeypatching in tests — the engine
#: makes exactly one kind of network call here, and it is cache-fronted.
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

#: Relation member roles that identify a served stop, best first. ``stop``
#: members are the vehicle's actual stopping positions; ``platform`` members are
#: the waiting areas. A relation tagged either way is usable, but mixing them
#: would double every stop, so only the best available role is taken.
_STOP_ROLES: tuple[str, ...] = ("stop", "platform")


def overpass_query(bbox_wgs84: tuple[float, float, float, float], snapshot_date: str | None = None) -> str:
    """The Overpass QL that returns route relations *with their ordered members*.

    This is why the fallback talks to Overpass directly instead of going through
    OSMnx: ``features_from_polygon`` flattens a relation into geometry and drops
    member **roles and order**, and the ordered stop sequence is precisely what a
    transit graph needs. It is the same Overpass service OSMnx already uses, so
    no new dependency and no new network surface — the raw JSON is parsed by
    :func:`parse_overpass_relations`, which is pure and unit-tested.

    When ``network.osm_snapshot_date`` is pinned, an Overpass *attic* query
    (``[date:...]``) reproduces the network as of that date — invariant #4
    extended to transit.
    """
    minx, miny, maxx, maxy = bbox_wgs84
    date_clause = f'[date:"{snapshot_date}T00:00:00Z"]' if snapshot_date else ""
    routes = "|".join(sorted(OSM_ROUTE_TYPES))
    return (
        f"[out:json][timeout:180]{date_clause};\n"
        f'relation["type"="route"]["route"~"^({routes})$"]({miny},{minx},{maxy},{maxx});\n'
        "out body;\n>;\nout skel qt;\n"
    )


def parse_overpass_relations(payload: dict) -> list[dict]:
    """Overpass JSON → ``{"route", "route_id", "name", "stops"}`` records.

    ``stops`` is the ordered list of ``(osm_id, lon, lat, name)`` the relation
    serves. Pure function over an already-fetched payload, so the OSM fallback's
    only interesting logic is unit-testable offline.
    """
    elements = payload.get("elements", [])
    nodes: dict[int, tuple[float, float, str]] = {}
    for element in elements:
        if element.get("type") == "node" and "lat" in element and "lon" in element:
            tags = element.get("tags") or {}
            nodes[int(element["id"])] = (
                float(element["lon"]),
                float(element["lat"]),
                str(tags.get("name") or element["id"]),
            )

    records: list[dict] = []
    for element in elements:
        if element.get("type") != "relation":
            continue
        tags = element.get("tags") or {}
        route = str(tags.get("route") or "")
        if route not in OSM_ROUTE_TYPES:
            continue
        members = element.get("members") or []
        stops: list[tuple[str, float, float, str]] = []
        for role in _STOP_ROLES:
            candidates = [
                m
                for m in members
                if m.get("type") == "node" and str(m.get("role") or "").startswith(role) and int(m["ref"]) in nodes
            ]
            if candidates:
                seen: set[int] = set()
                for member in candidates:
                    ref = int(member["ref"])
                    if ref in seen:
                        continue  # a loop route may list a terminus twice
                    seen.add(ref)
                    lon, lat, name = nodes[ref]
                    stops.append((str(ref), lon, lat, name))
                break
        if len(stops) < 2:
            continue
        records.append(
            {
                "route": route,
                "route_id": f"osm{element['id']}",
                "name": str(tags.get("name") or tags.get("ref") or route),
                "stops": stops,
            }
        )
    return records


def fetch_route_relations(boundary: LoadedBoundary, settings: Settings) -> list[dict]:
    """Live Overpass fetch of the boundary's public-transport route relations.

    The only place this module hits the network. A failing or empty Overpass
    response yields ``[]``; the caller turns that into a typed
    :class:`~segx.errors.TransitDataMissingError` rather than a silent
    transit-free run.
    """
    from urllib.request import Request, urlopen

    query = overpass_query(boundary.bbox_wgs84, settings.network.osm_snapshot_date)
    request = Request(OVERPASS_URL, data=query.encode("utf-8"), headers={"User-Agent": "segx"})
    try:
        with urlopen(request, timeout=300) as response:  # noqa: S310 — fixed https endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 — Overpass quirks (timeouts, 429s) become "no transit"
        return []
    return parse_overpass_relations(payload)


# --------------------------------------------------------------------------- #
# Pure conversion: relations → the LoadedGTFS shape                            #
# --------------------------------------------------------------------------- #


def build_osm_feed(
    relations: list[dict],
    settings: Settings,
    *,
    working_crs: str,
    geographic_crs: str = "EPSG:4326",
) -> LoadedGTFS:
    """Turn route-relation records into the derived-table shape GTFS produces.

    Headways are *assumed* per mode; inter-stop times are the straight-line
    distance at an assumed in-vehicle speed. Both are labelled by the
    ``osm_assumed`` source stamp — see the module docstring.
    """
    stop_rows: dict[str, tuple[float, float, str]] = {}
    pattern_rows: list[dict] = []
    pattern_stop_rows: list[dict] = []

    for relation in relations:
        route = str(relation["route"])
        route_type = OSM_ROUTE_TYPES[route]
        route_id = str(relation["route_id"])
        pattern_id = f"{route_id}::0"
        stops = relation["stops"]
        if len(stops) < 2:
            continue
        for stop_id, lon, lat, name in stops:
            stop_rows.setdefault(str(stop_id), (float(lon), float(lat), str(name)))
        headway = assumed_headway_min(route_type, settings)
        pattern_rows.append(
            {
                "pattern_id": pattern_id,
                "route_id": route_id,
                "direction_id": 0,
                "route_type": int(route_type),
                "route_name": str(relation.get("name") or route_id),
                "headway_min": headway,
                "n_trips": settings.transit.window_minutes / headway,
            }
        )
        for seq, (stop_id, *_rest) in enumerate(stops):
            pattern_stop_rows.append(
                {
                    "pattern_id": pattern_id,
                    "seq": seq,
                    "stop_id": str(stop_id),
                    # filled from inter-stop distance below, once stops are projected
                    "travel_time_min": 0.0 if seq == 0 else float("nan"),
                }
            )

    stops_gdf = gpd.GeoDataFrame(
        {
            "stop_id": list(stop_rows),
            "stop_name": [v[2] for v in stop_rows.values()],
        },
        geometry=gpd.points_from_xy([v[0] for v in stop_rows.values()], [v[1] for v in stop_rows.values()]),
        crs=geographic_crs,
    ).to_crs(working_crs)

    patterns = pd.DataFrame.from_records(
        pattern_rows,
        columns=["pattern_id", "route_id", "direction_id", "route_type", "route_name", "headway_min", "n_trips"],
    )
    pattern_stops = pd.DataFrame.from_records(
        pattern_stop_rows, columns=["pattern_id", "seq", "stop_id", "travel_time_min"]
    )

    # One fill per route type, so a metro hop is timed at metro speed.
    if not pattern_stops.empty:
        speeds = patterns.set_index("pattern_id")["route_type"].map(assumed_speed_kmh)
        filled: list[pd.DataFrame] = []
        for pattern_id, group in pattern_stops.groupby("pattern_id", sort=False):
            filled.append(fill_missing_hop_times(stops_gdf, group, speed_kmh=float(speeds.loc[pattern_id])))
        pattern_stops = pd.concat(filled, ignore_index=True)

    return LoadedGTFS(
        stops=stops_gdf,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source="osm_assumed",
        feed_hash=_relations_hash(relations),
        service_day=settings.transit.service_day,
        service_window=settings.transit.service_window,
        crs=working_crs,
    )


def _relations_hash(relations: list[dict]) -> str:
    payload = json.dumps(
        [
            {"route": r["route"], "route_id": r["route_id"], "stops": [str(s[0]) for s in r["stops"]]}
            for r in sorted(relations, key=lambda r: str(r["route_id"]))
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Cache-aware orchestration                                                    #
# --------------------------------------------------------------------------- #


def fetch_transit_osm(
    boundary: LoadedBoundary,
    settings: Settings,
    *,
    cache_dir: Path | None = None,
) -> LoadedGTFS:
    """Cache-aware OSM transit fetch, returning the :class:`LoadedGTFS` shape.

    Keyed the same way as the GTFS path (``(bbox, feed_hash, service_day,
    service_window, overlay_hash)``) but with ``osm_snapshot_date`` standing in
    for the feed hash, so the "never silently re-download" policy of invariant #4
    covers the fallback too.
    """
    from segx.io.gtfs import apply_overlay

    transit = settings.transit
    cache_dir = Path(cache_dir or settings.cache_dir)
    pseudo_feed_hash = f"osm:{settings.network.osm_snapshot_date or 'live'}"
    overlay_digest = overlay_hash(transit.overlay_path)
    key = transit_cache_key(
        boundary.bbox_wgs84, pseudo_feed_hash, transit.service_day, transit.service_window, overlay_digest
    )
    root = transit_cache_dir(cache_dir, key)
    if (root / "manifest.json").exists():
        return read_cached_feed(root)

    relations = fetch_route_relations(boundary, settings)
    feed = build_osm_feed(relations, settings, working_crs=boundary.crs, geographic_crs=settings.crs.geographic_crs)
    if transit.overlay_path is not None:
        feed = apply_overlay(feed, Path(transit.overlay_path), working_crs=boundary.crs)

    if feed.n_patterns == 0:
        raise TransitDataMissingError(
            "no usable OSM public-transport route relations were found in the study area; "
            "supply a GTFS feed via transit.gtfs_path / transit.gtfs_url",
            details={"bbox": list(boundary.bbox_wgs84), "n_relations": len(relations)},
        )

    write_cached_feed(feed, root)
    return feed


__all__ = [
    "OSM_ROUTE_TYPES",
    "assumed_headway_min",
    "assumed_speed_kmh",
    "build_osm_feed",
    "fetch_route_relations",
    "fetch_transit_osm",
]
