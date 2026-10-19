"""GTFS feed reader — the primary source for the ``transit`` mode (P7).

A GTFS feed is a zip of CSVs, so this module needs **no new dependency**:
``zipfile`` + ``pandas`` reads it. That is a deliberate choice, not laziness —
``urbanaccess`` builds its *own* OSM network (fighting invariant #1 and our
cache), ``peartree`` needs ``fiona``, and ``r5py`` needs a JVM (fatal for the
frozen desktop bundle).

What comes out is **not** the raw feed but the three derived tables the transit
graph builder needs, in the run's working CRS, clipped to the boundary:

* :attr:`LoadedGTFS.stops` — one row per stop actually served, geometry in the
  working CRS (invariant #1: nothing downstream touches lat/lon);
* :attr:`LoadedGTFS.patterns` — one row per ``(route_id, direction_id)`` *pattern*
  with its **headway** in the configured service window and its ``route_type``;
* :attr:`LoadedGTFS.pattern_stops` — the ordered stop sequence of each pattern,
  each hop carrying its **mean inter-stop travel time**.

Those three quantities — stops, ordered sequence, headway — are exactly what OSM
cannot supply (it has no timetables at all), which is why GTFS is primary and the
OSM fallback in :mod:`segx.network.transit_osm` is stamped ``osm_assumed``.

The derived tables are cached as parquet under ``cache_dir/transit/<key>/``,
keyed by ``(bbox, feed_hash, service_day, service_window, overlay_hash)`` — the
same "never silently re-read a changed input" discipline as
:mod:`segx.network.cache` (invariant #4). The window enters that key as its
**normalised bounds** (:func:`segx.config.parse_window`), not as the raw strings,
so the two spellings of one night window (``("22:00", "02:00")`` and
``("22:00", "26:00")``) share a cache entry instead of computing the same tables
twice. The honest consequence: the ``service_window`` read back from a cache hit
is whichever spelling wrote the entry first.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd

from segx.config import DAY_MINUTES, parse_window
from segx.errors import GTFSError, GTFSSchemaError, TransitDataMissingError

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings, TransitConfig
    from segx.io.boundary import LoadedBoundary

#: Feed source stamp. ``"osm_assumed"`` means the headways were *invented* from
#: ``transit.osm_assumed_headway_min``, not measured — it propagates into output
#: metadata so an approximate run is never mistaken for a real one (D1).
TransitSource = Literal["gtfs", "osm_assumed"]

#: The GTFS files we require. Everything else (``calendar``, ``calendar_dates``,
#: ``frequencies``, ``shapes``) is read when present and skipped when absent.
REQUIRED_FILES: tuple[str, ...] = ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt")

OPTIONAL_FILES: tuple[str, ...] = ("calendar.txt", "calendar_dates.txt", "frequencies.txt")

#: ``service_day`` → the ``calendar.txt`` column that flags it, and the
#: ``date.weekday()`` values it covers (used with ``calendar_dates.txt``).
_DAY_COLUMNS: dict[str, tuple[str, frozenset[int]]] = {
    "weekday": ("monday", frozenset({0, 1, 2, 3, 4})),
    "saturday": ("saturday", frozenset({5})),
    "sunday": ("sunday", frozenset({6})),
}

#: Extended GTFS ``route_type`` ranges (the "Google extended" scheme) collapsed
#: onto the five basic types the rest of the engine reasons about, so that
#: ``transit.surface_route_types`` stays a short, understandable list.
_EXTENDED_ROUTE_TYPES: tuple[tuple[int, int, int], ...] = (
    (100, 199, 2),  # railway service            → rail
    (200, 299, 3),  # coach service              → bus
    (400, 499, 1),  # urban railway / metro      → subway
    (700, 799, 3),  # bus service                → bus
    (800, 899, 11),  # trolleybus service        → trolleybus
    (900, 999, 0),  # tram service               → tram
    (1000, 1099, 4),  # water transport          → ferry
    (1200, 1299, 4),  # ferry service            → ferry
    (1400, 1499, 12),  # funicular               → monorail-ish (own right of way)
)


def normalize_route_type(value: object) -> int:
    """Collapse an (possibly extended) GTFS ``route_type`` onto a basic type.

    Basic types pass through unchanged: ``0`` tram, ``1`` subway, ``2`` rail,
    ``3`` bus, ``4`` ferry, ``11`` trolleybus, ``12`` monorail. Extended types
    (``100``–``1499``) map onto the nearest basic equivalent. An unparseable value
    becomes ``3`` (bus) — the commonest mode and the conservative choice, since a
    bus deposits on the street layer where it is visible and auditable rather than
    quietly on a separate ``rail_line`` layer.
    """
    try:
        rt = int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 3
    if rt in (0, 1, 2, 3, 4, 5, 6, 7, 11, 12):
        return rt
    for lo, hi, basic in _EXTENDED_ROUTE_TYPES:
        if lo <= rt <= hi:
            return basic
    return 3


# --------------------------------------------------------------------------- #
# The loaded feed                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LoadedGTFS:
    """Derived, boundary-clipped, working-CRS transit tables.

    This is the **single input contract** of :func:`segx.network.transit.
    build_transit_graph`; :mod:`segx.network.transit_osm` produces the same shape
    from OSM so the graph builder has one code path, not two.

    ``stops`` columns: ``stop_id`` (str), ``stop_name`` (str), ``geometry``
    (Point, working CRS).

    ``patterns`` columns: ``pattern_id`` (str), ``route_id``, ``direction_id``
    (int), ``route_type`` (int, normalised), ``route_name`` (str), ``headway_min``
    (float), ``n_trips`` (float — fractional when derived from
    ``frequencies.txt``).

    ``pattern_stops`` columns: ``pattern_id``, ``seq`` (0-based int), ``stop_id``,
    ``travel_time_min`` (float; minutes from the *previous* stop, ``0.0`` on the
    first stop of a pattern).
    """

    stops: gpd.GeoDataFrame
    patterns: pd.DataFrame
    pattern_stops: pd.DataFrame
    source: TransitSource
    feed_hash: str
    service_day: str
    service_window: tuple[str, str]
    crs: str

    @property
    def n_stops(self) -> int:
        return int(len(self.stops))

    @property
    def n_patterns(self) -> int:
        return int(len(self.patterns))

    def metadata(self) -> dict:
        """The provenance stamp that propagates into output metadata (D1).

        Both the raw ``service_window`` the user wrote *and* its normalised
        ``(start, end)`` bounds are stamped: the strings are what belongs in a
        report, but only the bounds say unambiguously whether the window crossed
        midnight and how many minutes the headways were measured over.
        """
        start, end = parse_window(self.service_window)
        return {
            "transit_source": self.source,
            "transit_feed_hash": self.feed_hash,
            "transit_service_day": self.service_day,
            "transit_service_window": list(self.service_window),
            "transit_service_window_bounds_min": [start, end],
            "transit_n_stops": self.n_stops,
            "transit_n_patterns": self.n_patterns,
        }


# --------------------------------------------------------------------------- #
# Feed reading                                                                 #
# --------------------------------------------------------------------------- #


def feed_hash(path: Path) -> str:
    """Content hash of a feed — a zip file's bytes, or a directory's members.

    Folded into the cache key so an updated feed is a cache *miss*, never a silent
    reuse of stale derived tables (invariant #4).
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for member in sorted(p for p in path.rglob("*.txt") if p.is_file()):
            digest.update(member.name.encode("utf-8"))
            digest.update(member.read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def read_feed_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read a GTFS zip (or an extracted directory) into ``{filename: DataFrame}``.

    Every column is read as ``str`` — GTFS ids are opaque strings and pandas
    would otherwise turn ``"007"`` into ``7`` and break the joins. Numeric
    coercion happens explicitly, per column, downstream.
    """
    path = Path(path)
    if not path.exists():
        raise GTFSError(f"GTFS feed not found: {path}", details={"path": str(path)})

    tables: dict[str, pd.DataFrame] = {}
    wanted = set(REQUIRED_FILES) | set(OPTIONAL_FILES)

    def _parse(name: str, raw: bytes) -> None:
        try:
            tables[name] = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_values=[""])
        except Exception as exc:  # noqa: BLE001 — any CSV fault is one typed error
            raise GTFSError(
                f"could not parse {name} in {path}: {exc}", details={"path": str(path), "file": name}
            ) from exc

    if path.is_dir():
        for name in wanted:
            member = path / name
            if member.exists():
                _parse(name, member.read_bytes())
    else:
        try:
            with zipfile.ZipFile(path) as zf:
                # Some agencies nest the feed one directory deep inside the zip.
                by_basename = {Path(n).name: n for n in zf.namelist() if Path(n).name in wanted}
                for name, member in by_basename.items():
                    _parse(name, zf.read(member))
        except zipfile.BadZipFile as exc:
            raise GTFSError(f"not a readable GTFS zip: {path}", details={"path": str(path)}) from exc

    missing = [f for f in REQUIRED_FILES if f not in tables]
    if missing:
        raise GTFSSchemaError(
            f"GTFS feed {path} is missing required file(s): {', '.join(missing)}",
            details={"path": str(path), "missing": missing, "found": sorted(tables)},
        )
    return tables


def _require_columns(df: pd.DataFrame, columns: list[str], *, file: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise GTFSSchemaError(
            f"{file} is missing required column(s): {', '.join(missing)}",
            details={"file": file, "missing": missing, "found": list(df.columns)},
        )


def download_feed(url: str, cache_dir: Path) -> Path:
    """Download a GTFS zip into ``cache_dir/gtfs/`` once and return its path.

    Cache-first and idempotent: a feed already downloaded under the same URL hash
    is reused, so an offline re-run works (a shipped app cannot assume
    connectivity).
    """
    from urllib.request import urlopen

    target_dir = Path(cache_dir) / "gtfs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}.zip"
    if target.exists():
        return target
    try:
        with urlopen(url) as response:  # noqa: S310 — the URL is operator-supplied config
            payload = response.read()
    except Exception as exc:  # noqa: BLE001
        raise GTFSError(f"could not download GTFS feed from {url}: {exc}", details={"url": url}) from exc
    target.write_bytes(payload)
    return target


# --------------------------------------------------------------------------- #
# Time / calendar helpers                                                      #
# --------------------------------------------------------------------------- #


def gtfs_time_to_minutes(series: pd.Series) -> pd.Series:
    """``"07:14:00"`` → ``434.0`` minutes after midnight, vectorised.

    GTFS deliberately allows hours ``>= 24`` for trips that run past midnight;
    that is preserved rather than wrapped, so a 25:10 departure sorts after a
    23:50 one on the same service day.
    """
    parts = series.astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 2:
        return pd.Series([float("nan")] * len(series), index=series.index, dtype=float)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce") if parts.shape[1] > 2 else 0.0
    return hours * 60.0 + minutes + (seconds if isinstance(seconds, float) else seconds.fillna(0.0)) / 60.0


def active_service_ids(tables: dict[str, pd.DataFrame], service_day: str) -> set[str] | None:
    """Service ids running on *service_day*, or ``None`` when undeterminable.

    ``calendar.txt`` is authoritative when present. Otherwise ``calendar_dates.txt``
    is used: a service counts as active if it has at least one *added* exception
    (``exception_type == 1``) falling on a matching weekday. A feed with neither
    file returns ``None``, meaning "keep every service" — an over-count that
    inflates trip counts and so *shortens* headways; it is reported by the caller
    rather than silently assumed correct.
    """
    day_column, weekdays = _DAY_COLUMNS[service_day]

    calendar = tables.get("calendar.txt")
    if calendar is not None and day_column in calendar.columns and "service_id" in calendar.columns:
        flag = pd.to_numeric(calendar[day_column], errors="coerce").fillna(0)
        return set(calendar.loc[flag > 0, "service_id"].astype(str))

    dates = tables.get("calendar_dates.txt")
    if dates is not None and {"service_id", "date", "exception_type"} <= set(dates.columns):
        added = dates[pd.to_numeric(dates["exception_type"], errors="coerce") == 1].copy()
        parsed = pd.to_datetime(added["date"], format="%Y%m%d", errors="coerce")
        keep = parsed.dt.weekday.isin(sorted(weekdays))
        return set(added.loc[keep.fillna(False), "service_id"].astype(str))

    return None


# --------------------------------------------------------------------------- #
# Pattern derivation — the actual work                                         #
# --------------------------------------------------------------------------- #


def _stop_times_in_service(
    tables: dict[str, pd.DataFrame], transit: TransitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(trips, stop_times)`` restricted to services running on the service day.

    Filtering *before* touching ``stop_times.txt`` matters: it is by far the
    largest file in a real feed (millions of rows), and every later operation is
    proportional to what survives here.
    """
    trips = tables["trips.txt"].copy()
    _require_columns(trips, ["trip_id", "route_id"], file="trips.txt")
    if "direction_id" not in trips.columns:
        trips["direction_id"] = "0"
    trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce").fillna(0).astype(int)

    services = active_service_ids(tables, transit.service_day)
    if services is not None and "service_id" in trips.columns:
        trips = trips[trips["service_id"].astype(str).isin(services)]

    stop_times = tables["stop_times.txt"].copy()
    _require_columns(stop_times, ["trip_id", "stop_id", "stop_sequence"], file="stop_times.txt")
    stop_times = stop_times[stop_times["trip_id"].isin(set(trips["trip_id"]))].copy()
    stop_times["stop_sequence"] = pd.to_numeric(stop_times["stop_sequence"], errors="coerce")
    stop_times = stop_times.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])

    arrival = stop_times["arrival_time"] if "arrival_time" in stop_times.columns else pd.Series(index=stop_times.index)
    departure = (
        stop_times["departure_time"] if "departure_time" in stop_times.columns else pd.Series(index=stop_times.index)
    )
    t = gtfs_time_to_minutes(departure).fillna(gtfs_time_to_minutes(arrival))
    if t.isna().any():
        # GTFS allows blank times at non-timepoint stops. Interpolate *within each
        # trip* (never across trip boundaries, which would invent a hop).
        t = t.groupby(stop_times["trip_id"]).transform(lambda s: s.interpolate(limit_direction="both"))
    stop_times["t_min"] = t
    return trips, stop_times


def _trips_in_window(
    trips: pd.DataFrame,
    stop_times: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    transit: TransitConfig,
) -> pd.Series:
    """Trips per ``(route_id, direction_id)`` inside the service window.

    Two sources are summed. Scheduled trips count as ``1`` each when their first
    departure falls inside the window. ``frequencies.txt`` entries contribute
    ``overlap_minutes / headway_min`` — a fractional count, which is the honest
    conversion of "every 8 minutes from 07:00 to 09:00" into a trip count over an
    arbitrary window.

    Midnight is handled on both sides. The window's ``end`` may exceed 1440
    (:func:`segx.config.parse_window`), a ``frequencies.txt`` row that wraps past
    midnight is normalised the same way, and when the window itself crosses
    midnight a departure written in plain small-hours form (``01:30``) is matched
    as well as one written in GTFS extended form (``25:30``).
    """
    window_start, window_end = transit.window_bounds_min
    key = ["route_id", "direction_id"]

    # ``.first()``, not ``.min()``: ``stop_times`` arrives sorted by
    # ``(trip_id, stop_sequence)``, and a trip written ``23:50 -> 00:05 -> 00:20``
    # has a *minimum* of 5 but a *first departure* of 1430. Taking the minimum
    # would file a late-night trip into the small hours of the same service day.
    first = stop_times.groupby("trip_id")["t_min"].first().rename("first_departure_min")
    trip_rd = trips.set_index("trip_id")[key].join(first, how="inner")

    frequency_trip_ids: set[str] = set()
    frequency_counts = pd.Series(dtype=float)
    freq = tables.get("frequencies.txt")
    if freq is not None and {"trip_id", "start_time", "end_time", "headway_secs"} <= set(freq.columns):
        f = freq.copy()
        f["start_min"] = gtfs_time_to_minutes(f["start_time"])
        f["end_min"] = gtfs_time_to_minutes(f["end_time"])
        f["headway_min"] = pd.to_numeric(f["headway_secs"], errors="coerce") / 60.0
        f = f[(f["headway_min"] > 0) & f["start_min"].notna() & f["end_min"].notna()].copy()
        # A row that ends at or before it starts runs past midnight ("06:00" ->
        # "01:00"). Left un-normalised, the clipped overlap goes negative, clamps
        # to zero, and ``build_patterns`` drops the *whole pattern* — a silent
        # zero (invariant #2) that deletes a real line from the graph.
        f["end_min"] = f["end_min"].where(f["end_min"] > f["start_min"], f["end_min"] + DAY_MINUTES)
        overlap = (f["end_min"].clip(upper=window_end) - f["start_min"].clip(lower=window_start)).clip(lower=0.0)
        f["n_trips"] = overlap / f["headway_min"]
        f = f[f["trip_id"].isin(trip_rd.index)]
        frequency_trip_ids = set(f["trip_id"])
        if not f.empty:
            joined = f.join(trip_rd[key], on="trip_id")
            frequency_counts = joined.groupby(key)["n_trips"].sum()

    scheduled = trip_rd[~trip_rd.index.isin(frequency_trip_ids)]
    departures = scheduled["first_departure_min"]
    direct = (departures >= window_start) & (departures < window_end)
    shifted = pd.Series(False, index=departures.index, dtype=bool)
    if window_end > DAY_MINUTES:
        # The window runs past midnight, so a departure filed as ``01:30`` belongs
        # in it just as much as one filed as ``25:30``. The ``< DAY_MINUTES`` guard
        # is what stops a ``25:30`` — already matched directly — from counting twice.
        moved = departures + DAY_MINUTES
        shifted = (departures < DAY_MINUTES) & (moved >= window_start) & (moved < window_end)
        _warn_on_mixed_midnight_encodings(scheduled, direct, shifted, key)

    scheduled_counts = scheduled[direct | shifted].groupby(key).size().astype(float)

    counts = scheduled_counts.add(frequency_counts, fill_value=0.0) if len(frequency_counts) else scheduled_counts
    return counts.rename("n_trips")


def _warn_on_mixed_midnight_encodings(
    scheduled: pd.DataFrame,
    direct: pd.Series,
    shifted: pd.Series,
    key: list[str],
) -> None:
    """Warn when one route-direction uses *both* past-midnight encodings.

    A feed may write a late trip as GTFS extended hours (``25:30``) or file an
    owl-service trip under a plain ``01:30``. Both are legal; a feed that uses
    both for the same ``(route_id, direction_id)`` almost certainly describes the
    same vehicle twice, which doubles ``n_trips`` and **halves** the headway. It
    cannot be told apart from genuine extra service without real dates, so this
    is a warning, not a silent correction — the same discipline as the
    ``osm_assumed`` stamp: an approximation must be labelled.
    """
    n_shifted = int(shifted.sum())
    if n_shifted == 0:
        return
    extended = scheduled[direct & (scheduled["first_departure_min"] >= DAY_MINUTES)].groupby(key).size()
    wrapped = scheduled[shifted].groupby(key).size()
    both = sorted(set(extended.index) & set(wrapped.index))
    if not both:
        return
    warnings.warn(
        f"n_trips_matched_by_midnight_shift={n_shifted}, and {len(both)} route-direction(s) "
        f"{both[:5]} mix GTFS extended hours (24:00+) with plain small-hours departures in the "
        "same service window. If those are the same vehicles recorded twice, their trip counts "
        "are doubled and their headways halved. Check the feed's encoding.",
        UserWarning,
        stacklevel=3,
    )


def _representative_sequences(trips: pd.DataFrame, stop_times: pd.DataFrame) -> dict[tuple[str, int], list[str]]:
    """The modal stop sequence of each ``(route_id, direction_id)``.

    Trips on the same route-direction usually differ slightly (short turns,
    depot runs). The sequence served by the *most* trips is the representative
    one; taking the union instead would invent stops no vehicle actually serves in
    order, and taking the longest would over-weight rare full-length runs.
    """
    per_trip = stop_times.groupby("trip_id")["stop_id"].agg(tuple)
    joined = trips.set_index("trip_id")[["route_id", "direction_id"]].join(per_trip.rename("seq"), how="inner")
    joined = joined[joined["seq"].map(len) >= 2]
    out: dict[tuple[str, int], list[str]] = {}
    for (route_id, direction_id), group in joined.groupby(["route_id", "direction_id"]):
        modal = group["seq"].value_counts().idxmax()
        out[(str(route_id), int(direction_id))] = list(modal)
    return out


def _mean_hop_times(trips: pd.DataFrame, stop_times: pd.DataFrame) -> pd.Series:
    """Mean minutes for every ``(route_id, direction_id, from_stop, to_stop)`` hop.

    Averaged over all the route-direction's trips, so a hop's cost reflects
    typical running time rather than one trip's schedule.
    """
    st = stop_times[["trip_id", "stop_id", "t_min"]].copy()
    st["to_stop"] = st.groupby("trip_id")["stop_id"].shift(-1)
    st["dt"] = st.groupby("trip_id")["t_min"].shift(-1) - st["t_min"]
    st = st.dropna(subset=["to_stop", "dt"])
    st = st[st["dt"] >= 0]
    st = st.join(trips.set_index("trip_id")[["route_id", "direction_id"]], on="trip_id")
    return st.groupby(["route_id", "direction_id", "stop_id", "to_stop"])["dt"].mean()


def build_patterns(
    tables: dict[str, pd.DataFrame],
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive ``(patterns, pattern_stops)`` from the raw GTFS tables.

    Pure pandas over already-read tables — no IO, no CRS, no boundary — so it is
    unit-testable against a tiny fixture feed. The headway is
    ``window_minutes / n_trips``: the *average* headway a passenger faces in the
    window, which is exactly what the frequency-based routing model consumes.
    """
    transit = settings.transit
    trips, stop_times = _stop_times_in_service(tables, transit)
    if trips.empty or stop_times.empty:
        return _empty_patterns(), _empty_pattern_stops()

    routes = tables["routes.txt"].copy()
    _require_columns(routes, ["route_id"], file="routes.txt")
    if "route_type" not in routes.columns:
        routes["route_type"] = "3"
    routes["route_type_basic"] = routes["route_type"].map(normalize_route_type)
    name_column = next(
        (c for c in ("route_short_name", "route_long_name", "route_id") if c in routes.columns), "route_id"
    )
    routes["route_name"] = routes[name_column].fillna(routes["route_id"]).astype(str)
    routes = routes[~routes["route_id"].duplicated(keep="first")]
    route_meta = routes.set_index("route_id")[["route_type_basic", "route_name"]]

    counts = _trips_in_window(trips, stop_times, tables, transit).to_dict()
    sequences = _representative_sequences(trips, stop_times)
    hop_times = _mean_hop_times(trips, stop_times).to_dict()

    window_minutes = transit.window_minutes
    pattern_rows: list[dict] = []
    stop_rows: list[dict] = []

    for (route_id, direction_id), sequence in sorted(sequences.items()):
        n_trips = float(counts.get((route_id, direction_id), 0.0))  # noqa: PD011
        if n_trips <= 0:
            continue  # no service in the analysis window — a real result, not an error
        if route_id not in route_meta.index:
            continue
        meta = route_meta.loc[route_id]
        pattern_id = f"{route_id}::{direction_id}"
        pattern_rows.append(
            {
                "pattern_id": pattern_id,
                "route_id": str(route_id),
                "direction_id": int(direction_id),
                "route_type": int(meta["route_type_basic"]),
                "route_name": str(meta["route_name"]),
                "headway_min": window_minutes / n_trips,
                "n_trips": n_trips,
            }
        )
        previous: str | None = None
        for seq, stop_id in enumerate(sequence):
            if previous is None:
                hop = 0.0
            else:
                key = (route_id, direction_id, previous, stop_id)
                hop = float(hop_times.get(key, float("nan")))
            stop_rows.append({"pattern_id": pattern_id, "seq": seq, "stop_id": str(stop_id), "travel_time_min": hop})
            previous = stop_id

    patterns = pd.DataFrame.from_records(pattern_rows, columns=_PATTERN_COLUMNS) if pattern_rows else _empty_patterns()
    pattern_stops = (
        pd.DataFrame.from_records(stop_rows, columns=_PATTERN_STOP_COLUMNS) if stop_rows else _empty_pattern_stops()
    )
    return patterns, pattern_stops


_PATTERN_COLUMNS = ["pattern_id", "route_id", "direction_id", "route_type", "route_name", "headway_min", "n_trips"]
_PATTERN_STOP_COLUMNS = ["pattern_id", "seq", "stop_id", "travel_time_min"]


def _empty_patterns() -> pd.DataFrame:
    return pd.DataFrame(columns=_PATTERN_COLUMNS)


def _empty_pattern_stops() -> pd.DataFrame:
    return pd.DataFrame(columns=_PATTERN_STOP_COLUMNS)


# --------------------------------------------------------------------------- #
# Stops → working CRS, boundary clip, sequence repair                          #
# --------------------------------------------------------------------------- #


def build_stops_gdf(stops: pd.DataFrame, *, working_crs: str, geographic_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """``stops.txt`` → points in the working CRS (invariant #1, at the boundary)."""
    _require_columns(stops, ["stop_id", "stop_lat", "stop_lon"], file="stops.txt")
    df = stops.copy()
    df["stop_id"] = df["stop_id"].astype(str)
    df["stop_name"] = df["stop_name"].astype(str) if "stop_name" in df.columns else df["stop_id"]
    lat = pd.to_numeric(df["stop_lat"], errors="coerce")
    lon = pd.to_numeric(df["stop_lon"], errors="coerce")
    keep = lat.notna() & lon.notna() & ~df["stop_id"].duplicated(keep="first")
    df, lat, lon = df[keep], lat[keep], lon[keep]
    gdf = gpd.GeoDataFrame(
        df[["stop_id", "stop_name"]].reset_index(drop=True),
        geometry=gpd.points_from_xy(lon.to_numpy(), lat.to_numpy()),
        crs=geographic_crs,
    )
    return gdf.to_crs(working_crs)


def clip_patterns_to_boundary(
    stops: gpd.GeoDataFrame,
    patterns: pd.DataFrame,
    pattern_stops: pd.DataFrame,
    boundary_geometry,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict the feed to the study area, splitting broken sequences.

    A line that leaves the boundary and comes back would, if its outside stops
    were simply dropped, gain a phantom hop straight across the gap. Each pattern
    is therefore cut into its **contiguous in-boundary runs**, each becoming its
    own pattern (``<pattern_id>#<n>``) inheriting the original's headway and route
    type. A run of fewer than two stops carries no ride and is discarded.
    """
    inside_ids = set(stops.loc[stops.geometry.intersects(boundary_geometry), "stop_id"])
    kept_stops = stops[stops["stop_id"].isin(inside_ids)].reset_index(drop=True)

    new_pattern_rows: list[dict] = []
    new_stop_rows: list[dict] = []
    pattern_meta = patterns.set_index("pattern_id")

    for pattern_id, group in pattern_stops.sort_values(["pattern_id", "seq"]).groupby("pattern_id", sort=True):
        if pattern_id not in pattern_meta.index:
            continue
        meta = pattern_meta.loc[pattern_id]
        runs: list[list[tuple[str, float]]] = []
        current: list[tuple[str, float]] = []
        for row in group.itertuples(index=False):
            if row.stop_id in inside_ids:
                current.append((str(row.stop_id), float(row.travel_time_min)))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)

        for index, run in enumerate(r for r in runs if len(r) >= 2):
            new_id = pattern_id if len(runs) == 1 else f"{pattern_id}#{index}"
            new_pattern_rows.append(
                {
                    "pattern_id": new_id,
                    "route_id": str(meta["route_id"]),
                    "direction_id": int(meta["direction_id"]),
                    "route_type": int(meta["route_type"]),
                    "route_name": str(meta["route_name"]),
                    "headway_min": float(meta["headway_min"]),
                    "n_trips": float(meta["n_trips"]),
                }
            )
            for seq, (stop_id, hop) in enumerate(run):
                new_stop_rows.append(
                    {
                        "pattern_id": new_id,
                        "seq": seq,
                        "stop_id": stop_id,
                        "travel_time_min": 0.0 if seq == 0 else hop,
                    }
                )

    out_patterns = (
        pd.DataFrame.from_records(new_pattern_rows, columns=_PATTERN_COLUMNS) if new_pattern_rows else _empty_patterns()
    )
    out_stop_rows = (
        pd.DataFrame.from_records(new_stop_rows, columns=_PATTERN_STOP_COLUMNS)
        if new_stop_rows
        else _empty_pattern_stops()
    )
    served = set(out_stop_rows["stop_id"])
    kept_stops = kept_stops[kept_stops["stop_id"].isin(served)].reset_index(drop=True)
    return gpd.GeoDataFrame(kept_stops, geometry="geometry", crs=stops.crs), out_patterns, out_stop_rows


def fill_missing_hop_times(
    stops: gpd.GeoDataFrame,
    pattern_stops: pd.DataFrame,
    *,
    speed_kmh: float,
) -> pd.DataFrame:
    """Replace ``NaN`` hop times with a straight-line estimate at *speed_kmh*.

    A ``NaN`` reaches here only when a hop exists in the representative sequence
    but in no trip's timings — rare, and a hard ``NaN`` would poison Dijkstra
    silently. Estimating from the inter-stop distance keeps the edge usable and,
    because it is an *upper*-speed guess on a straight line, errs toward making the
    ride look slightly too attractive rather than dropping the line entirely.
    """
    if pattern_stops.empty:
        return pattern_stops
    out = pattern_stops.sort_values(["pattern_id", "seq"]).reset_index(drop=True)
    missing = out["travel_time_min"].isna()
    if not missing.any():
        return out
    coords = stops.set_index("stop_id").geometry
    previous = out.groupby("pattern_id")["stop_id"].shift(1)
    for index in out.index[missing]:
        here, before = out.at[index, "stop_id"], previous.get(index)
        if before is None or pd.isna(before) or here not in coords.index or before not in coords.index:
            out.at[index, "travel_time_min"] = 0.0
            continue
        metres = float(coords.loc[before].distance(coords.loc[here]))
        out.at[index, "travel_time_min"] = metres * 60.0 / (speed_kmh * 1000.0)
    return out


# --------------------------------------------------------------------------- #
# Editing overlay (D5 — the seam, not the editor)                              #
# --------------------------------------------------------------------------- #


def overlay_hash(path: Path | None) -> str | None:
    """Content hash of the overlay file, folded into the transit cache key."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise GTFSError(f"transit.overlay_path does not exist: {path}", details={"path": str(path)})
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def apply_overlay(feed: LoadedGTFS, overlay_path: Path, *, working_crs: str) -> LoadedGTFS:
    """Apply a scenario-edit overlay to a loaded feed.

    The overlay is a GeoJSON/GPKG whose rows each carry an ``action`` and the
    fields that action needs:

    ============================  ==========================================
    ``action``                    fields
    ============================  ==========================================
    ``add_stop``                  ``stop_id``, geometry, optional ``stop_name``
    ``remove_stop``               ``stop_id``
    ``move_stop``                 ``stop_id``, geometry
    ``edit_route_sequence``       ``pattern_id``, ``stop_ids`` (comma-separated)
    ============================  ==========================================

    This is the engine half of the editing feature. The *interactive editor* that
    writes such a file is deliberately
    not built here — but because the overlay hash is part of the transit cache
    key, an edited scenario caches separately from the base feed, which makes
    scenario comparison ("what if we add a metro station here?") work today.
    """
    overlay = gpd.read_file(overlay_path)
    if overlay.empty:
        return feed
    if overlay.crs is not None and str(overlay.crs) != working_crs:
        overlay = overlay.to_crs(working_crs)
    if "action" not in overlay.columns:
        raise GTFSSchemaError(
            f"transit overlay {overlay_path} has no 'action' column",
            details={"path": str(overlay_path), "found": list(overlay.columns)},
        )

    stops = feed.stops.copy()
    pattern_stops = feed.pattern_stops.copy()
    known_actions = {"add_stop", "remove_stop", "move_stop", "edit_route_sequence"}

    for row in overlay.itertuples(index=False):
        action = str(getattr(row, "action", "")).strip()
        if action not in known_actions:
            raise GTFSSchemaError(
                f"transit overlay {overlay_path}: unknown action {action!r}",
                details={"path": str(overlay_path), "action": action, "known": sorted(known_actions)},
            )
        stop_id = str(getattr(row, "stop_id", "") or "")
        if action == "add_stop":
            name = str(getattr(row, "stop_name", "") or stop_id)
            stops = pd.concat(
                [
                    stops,
                    gpd.GeoDataFrame(
                        {"stop_id": [stop_id], "stop_name": [name]}, geometry=[row.geometry], crs=working_crs
                    ),
                ],
                ignore_index=True,
            )
        elif action == "remove_stop":
            stops = stops[stops["stop_id"] != stop_id]
            pattern_stops = pattern_stops[pattern_stops["stop_id"] != stop_id]
        elif action == "move_stop":
            stops.loc[stops["stop_id"] == stop_id, "geometry"] = row.geometry
        else:  # edit_route_sequence
            pattern_id = str(getattr(row, "pattern_id", "") or "")
            stop_ids = [s.strip() for s in str(getattr(row, "stop_ids", "") or "").split(",") if s.strip()]
            pattern_stops = pattern_stops[pattern_stops["pattern_id"] != pattern_id]
            replacement = pd.DataFrame(
                {
                    "pattern_id": pattern_id,
                    "seq": range(len(stop_ids)),
                    "stop_id": stop_ids,
                    "travel_time_min": float("nan"),
                }
            )
            replacement.loc[replacement["seq"] == 0, "travel_time_min"] = 0.0
            pattern_stops = pd.concat([pattern_stops, replacement], ignore_index=True)

    stops = gpd.GeoDataFrame(stops.reset_index(drop=True), geometry="geometry", crs=working_crs)
    # A removed stop can orphan a pattern down to a single stop; drop those.
    sizes = pattern_stops.groupby("pattern_id")["seq"].size()
    alive = set(sizes[sizes >= 2].index)
    pattern_stops = pattern_stops[pattern_stops["pattern_id"].isin(alive)].reset_index(drop=True)
    pattern_stops["seq"] = pattern_stops.groupby("pattern_id").cumcount()
    patterns = feed.patterns[feed.patterns["pattern_id"].isin(alive)].reset_index(drop=True)
    return LoadedGTFS(
        stops=stops,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source=feed.source,
        feed_hash=feed.feed_hash,
        service_day=feed.service_day,
        service_window=feed.service_window,
        crs=feed.crs,
    )


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #

_BBOX_ROUND_DECIMALS = 6


def transit_cache_key(
    bbox: tuple[float, float, float, float],
    feed_digest: str,
    service_day: str,
    service_window: tuple[str, str],
    overlay_digest: str | None = None,
) -> str:
    """Deterministic 16-char digest of everything that shapes the derived tables.

    The window enters as its **normalised bounds**, not as the raw strings: two
    spellings of one night window (``("22:00", "02:00")`` and
    ``("22:00", "26:00")``) are the same run and must share one cache entry.
    """
    payload = json.dumps(
        {
            "bbox": [round(c, _BBOX_ROUND_DECIMALS) for c in bbox],
            "feed": feed_digest,
            "service_day": service_day,
            "service_window": list(parse_window(service_window)),
            "overlay": overlay_digest,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def transit_cache_dir(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / "transit" / key


def write_cached_feed(feed: LoadedGTFS, root: Path) -> None:
    """Persist the derived tables + a provenance manifest."""
    root.mkdir(parents=True, exist_ok=True)
    feed.stops.to_parquet(root / "stops.parquet")
    feed.patterns.to_parquet(root / "patterns.parquet")
    feed.pattern_stops.to_parquet(root / "pattern_stops.parquet")
    (root / "manifest.json").write_text(json.dumps({**feed.metadata(), "crs": feed.crs}, indent=2), encoding="utf-8")


def read_cached_feed(root: Path) -> LoadedGTFS:
    """Reconstruct a :class:`LoadedGTFS` written by :func:`write_cached_feed`."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    stops = gpd.read_parquet(root / "stops.parquet")
    return LoadedGTFS(
        stops=stops,
        patterns=pd.read_parquet(root / "patterns.parquet"),
        pattern_stops=pd.read_parquet(root / "pattern_stops.parquet"),
        source=manifest["transit_source"],
        feed_hash=manifest["transit_feed_hash"],
        service_day=manifest["transit_service_day"],
        service_window=tuple(manifest["transit_service_window"]),  # type: ignore[arg-type]
        crs=manifest.get("crs", str(stops.crs)),
    )


# --------------------------------------------------------------------------- #
# Top-level loader                                                             #
# --------------------------------------------------------------------------- #


def load_gtfs(
    boundary: LoadedBoundary,
    settings: Settings,
    *,
    cache_dir: Path | None = None,
) -> LoadedGTFS:
    """Load, derive, clip, overlay and cache a GTFS feed for the study area.

    Cache-first: a hit reads the derived parquet tables and never re-parses the
    feed. A miss reads ``transit.gtfs_path`` (or downloads ``transit.gtfs_url``
    once into the cache), derives the patterns, reprojects the stops to the
    working CRS, clips to the boundary, and applies ``transit.overlay_path`` if
    set — in that order, so an overlay edits a study-area feed rather than a
    world-scale one.
    """
    transit = settings.transit
    cache_dir = Path(cache_dir or settings.cache_dir)

    if transit.gtfs_path is not None:
        path = Path(transit.gtfs_path)
    elif transit.gtfs_url is not None:
        path = download_feed(transit.gtfs_url, cache_dir)
    else:
        raise TransitDataMissingError(
            "GTFS requested but neither transit.gtfs_path nor transit.gtfs_url is set",
            details={"source": transit.source},
        )

    digest = feed_hash(path)
    overlay_digest = overlay_hash(transit.overlay_path)
    key = transit_cache_key(boundary.bbox_wgs84, digest, transit.service_day, transit.service_window, overlay_digest)
    root = transit_cache_dir(cache_dir, key)
    if (root / "manifest.json").exists():
        return read_cached_feed(root)

    tables = read_feed_tables(path)
    patterns, pattern_stops = build_patterns(tables, settings)
    stops = build_stops_gdf(tables["stops.txt"], working_crs=boundary.crs, geographic_crs=settings.crs.geographic_crs)
    stops, patterns, pattern_stops = clip_patterns_to_boundary(stops, patterns, pattern_stops, boundary.geometry)
    pattern_stops = fill_missing_hop_times(stops, pattern_stops, speed_kmh=settings.network.drive_speed_fallback_kmh)

    feed = LoadedGTFS(
        stops=stops,
        patterns=patterns,
        pattern_stops=pattern_stops,
        source="gtfs",
        feed_hash=digest,
        service_day=transit.service_day,
        service_window=transit.service_window,
        crs=boundary.crs,
    )
    if transit.overlay_path is not None:
        feed = apply_overlay(feed, Path(transit.overlay_path), working_crs=boundary.crs)

    if feed.n_patterns == 0:
        raise TransitDataMissingError(
            f"GTFS feed {path} yields no transit patterns inside the study area for "
            f"{transit.service_day} {transit.service_window[0]}-{transit.service_window[1]}",
            details={
                "path": str(path),
                "service_day": transit.service_day,
                "service_window": list(transit.service_window),
                "n_stops_in_feed": int(len(tables["stops.txt"])),
            },
        )

    write_cached_feed(feed, root)
    return feed


__all__ = [
    "LoadedGTFS",
    "OPTIONAL_FILES",
    "REQUIRED_FILES",
    "TransitSource",
    "active_service_ids",
    "apply_overlay",
    "build_patterns",
    "build_stops_gdf",
    "clip_patterns_to_boundary",
    "download_feed",
    "feed_hash",
    "fill_missing_hop_times",
    "gtfs_time_to_minutes",
    "load_gtfs",
    "normalize_route_type",
    "overlay_hash",
    "read_cached_feed",
    "read_feed_tables",
    "transit_cache_dir",
    "transit_cache_key",
    "write_cached_feed",
]
