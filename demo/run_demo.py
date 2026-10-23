"""SegX end-to-end demo on a real US city using LODES home→work flows.

This is a *showcase* script, not part of the engine. It runs the real pipeline as
far as P3 (ingest → population → routing) on real data and renders a map, so the
work done so far can be **shown** rather than described:

    real OSM street network + real OSM POIs + real LODES commute flows
        → located, income-labelled origins
        → routed trajectories (nearest POI, 30-min catchment, home→work commute)

It deliberately calls the importable orchestrator stages directly (``ingest`` /
``build_population`` / ``route``) instead of ``run_pipeline``, because the full
pipeline still raises at P4 (aggregation). That's expected — P3 is the current
frontier.

Inputs you provide (see demo/README.md):
  --geometry  TIGER/Line census-block polygons (GPKG/SHP/GeoJSON) with the GEOID
              column named in the config, matching the LODES geocodes.
  --od        A LODES-style OD CSV (home/work block ids + per-class worker counts).

The column names are read from the config (demo/segx_lodes.toml), so they need
not match standard LODES. Everything else (street graph, POIs, the study
boundary) is derived/fetched.

Usage:
    uv sync --group demo            # once, to get matplotlib
    uv run python demo/run_demo.py \
        --geometry philadelphia_blocks.shp \
        --od       philadelphia_OD.csv \
        --bbox="-75.165,39.945,-75.150,39.957"   # clip to a neighbourhood

Note the ``--bbox=`` form (with the equals sign): a leading ``-`` would otherwise
be parsed as a flag. Add ``--set key=value`` (repeatable) to override any config
value, exactly like the CLI (e.g. ``--set poi.catchment.budget_minutes=45``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import geopandas as gpd

# Make `segx` importable when run as `python demo/run_demo.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segx.config import Settings  # noqa: E402
from segx.metrics.registry import available_metrics  # noqa: E402
from segx.orchestrator import aggregate, analyse_od, build_population, ingest, route, write  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DEMO_DIR / "segx_lodes.toml"

#: output sub-folders, created under --out; keeps maps/charts/tables/inputs tidy
OUTPUT_DIRS = ("inputs", "cache", "maps", "segments", "charts", "tables", "deliverables")


def _make_layout(out_dir: Path) -> dict[str, Path]:
    """Create and return the output folder layout under *out_dir*."""
    layout = {name: out_dir / name for name in OUTPUT_DIRS}
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _parse_overrides(items: list[str]) -> dict[str, object]:
    """Same `dotted.key=value` (JSON-parsed) override syntax as the CLI."""
    out: dict[str, object] = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


# --------------------------------------------------------------------------- #
# Travel mode on the OD file (what makes `cross_mode_exposure` non-trivial)     #
# --------------------------------------------------------------------------- #
# LODES is a *flow* table: it says how many workers go from block A to block B,
# never how they get there. The engine's `od_columns.mode` column is the seam
# where real modal data (a household travel survey, a smart-card OD matrix)
# enters, and `cross_mode_exposure` (P7.1) is trivial without it — with a single
# mode it reduces exactly to `exposure`. So the demo *synthesises* the column.
#
# It is synthesised, so it is LABELLED (invariant #7's discipline): the shares
# and the seed are written next to the CSV in `od_clip_modes.json`, printed to
# the console, and the numbers that come out are a demonstration of the
# machinery, NOT a finding about Philadelphia. A source OD file that already
# carries the mode column is passed through untouched and nothing is invented.

#: Default synthetic modal split — roughly ACS journey-to-work for Philadelphia
#: (car ≈ drive alone + carpool, transit ≈ 25 % split bus/metro the way SEPTA's
#: ridership is, walk high because the demo bbox is Center City). Override with
#: --mode-split. The *labels* are the ones a survey would use, not graph names:
#: `segx.orchestrator._OD_MODE_ALIAS` maps car→drive and bus/subway→transit, so
#: the OD file reads like transport data and the engine does the translation.
DEFAULT_MODE_SPLIT: dict[str, float] = {"car": 0.45, "bus": 0.20, "subway": 0.15, "walk": 0.20}


def _parse_mode_split(raw: str) -> dict[str, float]:
    """Parse ``"car=0.5,transit=0.3,walk=0.2"`` into normalised shares."""
    split: dict[str, float] = {}
    for item in raw.split(","):
        name, _, value = item.partition("=")
        name = name.strip()
        if not name:
            continue
        try:
            share = float(value)
        except ValueError as exc:
            raise SystemExit(f"--mode-split: {item!r} is not 'mode=share'") from exc
        if share < 0:
            raise SystemExit(f"--mode-split: negative share for {name!r}")
        split[name] = split.get(name, 0.0) + share
    total = sum(split.values())
    if total <= 0:
        raise SystemExit("--mode-split: shares sum to zero")

    # Fail here, not 20 minutes into routing: a mode label the engine cannot
    # resolve to a graph raises ModeNotRegisteredError deep inside P3.
    from segx.network.registry import KNOWN_MODES
    from segx.orchestrator import _OD_MODE_ALIAS

    known = set(KNOWN_MODES) | set(_OD_MODE_ALIAS)
    unknown = sorted(set(split) - known)
    if unknown:
        raise SystemExit(f"--mode-split: unknown mode(s) {unknown}; known labels: {sorted(known)}")
    return {k: v / total for k, v in split.items()}


def _assign_modes(od, *, origin_col: str, dest_col: str, split: dict[str, float], seed: int):
    """A deterministic, seeded travel mode per OD flow.

    Assignment is a **hash** of ``(seed, origin, dest)`` bucketed by the
    cumulative shares — not a row-order draw — so a given home→work pair keeps
    the same mode no matter which ``--bbox`` produced the clip, which row it
    landed on, or how many flows came along with it. Two runs of the demo are
    therefore comparable, and so are a clipped run and a whole-city one.
    """
    import pandas as pd

    modes = list(split)
    edges, acc = [], 0.0
    for name in modes:
        acc += split[name]
        edges.append(acc)

    def _mode_of(origin: str, dest: str) -> str:
        digest = hashlib.blake2b(f"{seed}|{origin}|{dest}".encode(), digest_size=8).digest()
        u = int.from_bytes(digest, "big") / 2**64
        for name, edge in zip(modes, edges, strict=True):
            if u < edge:
                return name
        return modes[-1]

    return pd.Series(
        [_mode_of(o, d) for o, d in zip(od[origin_col].astype(str), od[dest_col].astype(str), strict=True)],
        index=od.index,
        dtype=object,
    )


def _required_graph_modes(od_clip_path: Path, settings: Settings) -> set[str]:
    """The per-mode graphs the OD file's ``mode`` column actually asks for.

    The column holds survey labels (``car``, ``bus``, ``subway``); the engine
    resolves them to graph names through ``_OD_MODE_ALIAS``. Reading them here
    lets the demo build exactly the graphs the run needs — and refuse *before*
    fetching OSM when it has been asked to route a mode it was not set up for,
    instead of raising from inside P3 twenty minutes later.
    """
    import pandas as pd

    from segx.orchestrator import _OD_MODE_ALIAS

    mode_col = settings.od_columns.mode
    header = pd.read_csv(od_clip_path, nrows=0).columns
    if mode_col in header:
        labels = pd.read_csv(od_clip_path, usecols=[mode_col], dtype=str)[mode_col].dropna().unique()
    else:
        # No column: every flow takes the category default, which is a survey
        # label too (`commute = "car"`) and needs the same translation.
        labels = [settings.od_columns.mode_for(settings.od_columns.default_category)]
    return {_OD_MODE_ALIAS.get(str(m), str(m)) for m in labels}


def _prepare_inputs(
    geometry_path: Path,
    od_path: Path,
    bbox: tuple[float, float, float, float] | None,
    settings: Settings,
    inputs_dir: Path,
    *,
    mode_split: dict[str, float] | None = None,
) -> tuple[Path, Path, Path]:
    """Clip the study area and produce a matched (blocks, OD, boundary) triple.

    A whole-city block file + a million-row OD table is far too much to route
    live, so an optional WGS84 ``bbox`` clips the blocks to a neighbourhood. The
    OD loader requires **every** endpoint id to resolve to a block geometry, so
    we also keep only OD flows whose *both* endpoints fall inside the clipped
    blocks (intra-area commutes). Finally we synthesise the trajectory-weight
    column (``S000`` = SE01+SE02+SE03, total workers in the flow) since the file
    has no total column, and dissolve the blocks into one boundary polygon.

    When ``mode_split`` is given and the source file has no ``od_columns.mode``
    column, a synthetic travel mode is added (see :func:`_assign_modes`) and its
    provenance written to ``od_clip_modes.json`` beside the CSV.

    Returns ``(blocks_path, od_path, boundary_path)``.
    """
    import pandas as pd

    inputs_dir.mkdir(parents=True, exist_ok=True)
    blocks = gpd.read_file(geometry_path)
    id_col = settings.demographics.polygon_id_column
    blocks[id_col] = blocks[id_col].astype(str)

    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        blocks = blocks.loc[blocks.to_crs("EPSG:4326").cx[minx:maxx, miny:maxy].index]
        if blocks.empty:
            raise SystemExit(f"--bbox {bbox} clipped out every block; widen it.")
    kept_ids = set(blocks[id_col])
    print(f"  blocks in study area: {len(kept_ids)}")

    # OD: keep only intra-area flows; synthesise the worker-count weight column.
    h_col = settings.od_columns.origin_id_ref
    w_col = settings.od_columns.dest_id_ref
    weight_col = settings.od_columns.weight
    mode_col = settings.od_columns.mode
    od = pd.read_csv(od_path, dtype={h_col: str, w_col: str})
    in_area = od[h_col].isin(kept_ids) & od[w_col].isin(kept_ids)
    od = od.loc[in_area].copy()
    if od.empty:
        raise SystemExit("no OD flow has both endpoints inside --bbox; widen it.")
    class_cols = settings.demographics.attributes[0].class_columns or []
    if weight_col not in od.columns and class_cols:
        od[weight_col] = od[class_cols].sum(axis=1)
    print(f"  intra-area OD flows : {len(od)} (of the full file)")

    blocks_path = inputs_dir / "blocks_clip.gpkg"
    od_clip_path = inputs_dir / "od_clip.csv"
    boundary_path = inputs_dir / "boundary.gpkg"

    # ---- the travel-mode column -------------------------------------------- #
    provenance_path = inputs_dir / "od_clip_modes.json"
    if mode_col in od.columns:
        print(f"  travel mode         : column {mode_col!r} present in the source OD — used as-is (real data)")
        provenance_path.unlink(missing_ok=True)
    elif mode_split:
        od[mode_col] = _assign_modes(od, origin_col=h_col, dest_col=w_col, split=mode_split, seed=settings.random_seed)
        observed = od[mode_col].value_counts()
        shares = "  ".join(f"{m}={int(observed.get(m, 0))}" for m in mode_split)
        print(f"  travel mode         : SYNTHETIC column {mode_col!r} (LODES carries none) — {shares} flows")
        print(f"                        target split {mode_split}, seed {settings.random_seed}")
        provenance_path.write_text(
            json.dumps(
                {
                    "column": mode_col,
                    "source": "synthetic_hashed_split",
                    "why": (
                        "LODES is a flow table with no travel mode; the column is invented so the "
                        "multi-mode machinery (transit routing, cross_mode_exposure) has something "
                        "to work on. Modal numbers from this run describe the engine, not the city."
                    ),
                    "target_split": mode_split,
                    "observed_counts": {str(k): int(v) for k, v in observed.items()},
                    "seed": settings.random_seed,
                    "assignment": "blake2b(f'{seed}|{origin_id_ref}|{dest_id_ref}') bucketed by cumulative share",
                    "origin_column": h_col,
                    "dest_column": w_col,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"                        provenance written to {provenance_path}")
    else:
        # No column and no split asked for: the engine falls back to
        # `od_columns.default_mode_by_category` (commute -> car), as before.
        provenance_path.unlink(missing_ok=True)

    blocks.to_file(blocks_path, driver="GPKG")
    od.to_csv(od_clip_path, index=False)
    gpd.GeoDataFrame(geometry=[blocks.union_all()], crs=blocks.crs).to_file(boundary_path, driver="GPKG")
    return blocks_path, od_clip_path, boundary_path


# --------------------------------------------------------------------------- #
# GTFS preparation (P7) — agency packaging -> one feed the engine can read      #
# --------------------------------------------------------------------------- #
# `segx.io.gtfs.read_feed_tables` reads *one* GTFS zip (or an extracted
# directory). Agencies do not always ship one: SEPTA's `gtfs_public.zip` is a
# zip **of two zips** (`google_bus.zip` + `google_rail.zip`), which between them
# hold the metro, the trolleys, the buses and the Regional Rail — i.e. exactly
# the surface/rail split P7 cares about. Rather than teach the engine every
# agency's packaging, the demo normalises here: unpack, namespace the ids per
# sub-feed so `route_id` "1" from the bus feed never collides with the rail
# feed's, and write one plain feed directory the loader reads unchanged.
#
# It also *thins* the feed to the study area, because SEPTA's bus
# `stop_times.txt` is 103 MB and the demo bbox is a few blocks. Thinning keeps
# whole **trips** (every trip touching the bbox, with its complete stop
# sequence), never individual stops — so the engine still sees a line leaving
# and re-entering the area and can split it, which is the whole point of
# `clip_patterns_to_boundary`.

#: The files `segx.io.gtfs` reads (REQUIRED_FILES + OPTIONAL_FILES); everything
#: else in an agency zip (shapes, fares, pathways…) is dropped.
_GTFS_FILES: tuple[str, ...] = (
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
    "calendar_dates.txt",
    "frequencies.txt",
)

#: Columns holding an id that must be namespaced when sub-feeds are merged.
_GTFS_ID_COLUMNS: dict[str, tuple[str, ...]] = {
    "stops.txt": ("stop_id", "parent_station"),
    "routes.txt": ("route_id",),
    "trips.txt": ("route_id", "service_id", "trip_id"),
    "stop_times.txt": ("trip_id", "stop_id"),
    "calendar.txt": ("service_id",),
    "calendar_dates.txt": ("service_id",),
    "frequencies.txt": ("trip_id",),
}

#: rows per chunk when streaming a large `stop_times.txt`
_GTFS_CHUNK = 500_000


def _file_sha(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _gtfs_subfeeds(gtfs_path: Path, work_dir: Path) -> dict[str, Path]:
    """Return ``{feed_name: directory}`` for *gtfs_path*, unnesting zip-of-zips."""
    import shutil
    import zipfile

    if gtfs_path.is_dir():
        return {gtfs_path.name: gtfs_path}

    with zipfile.ZipFile(gtfs_path) as zf:
        names = zf.namelist()
        nested = [n for n in names if n.lower().endswith(".zip")]
        has_txt = any(Path(n).name in _GTFS_FILES for n in names)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        if has_txt or not nested:
            zf.extractall(work_dir)
            return {gtfs_path.stem: work_dir}
        print(f"  GTFS: {gtfs_path.name} is a zip of {len(nested)} feed(s) — unpacking {', '.join(nested)}")
        out: dict[str, Path] = {}
        for member in nested:
            name = Path(member).stem
            target = work_dir / name
            target.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as inner, zipfile.ZipFile(io.BytesIO(inner.read())) as sub:
                sub.extractall(target)
            out[name] = target
        return out


def _gtfs_trips_in_bbox(feed_dir: Path, bbox: tuple[float, float, float, float] | None) -> set[str] | None:
    """Trip ids of every trip that calls at a stop inside *bbox* (``None`` = all)."""
    import pandas as pd

    if bbox is None:
        return None
    minx, miny, maxx, maxy = bbox
    stops = pd.read_csv(feed_dir / "stops.txt", dtype=str, keep_default_na=False, na_values=[""])
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    inside = set(stops.loc[lon.between(minx, maxx) & lat.between(miny, maxy), "stop_id"])
    if not inside:
        return set()
    trips: set[str] = set()
    for chunk in pd.read_csv(
        feed_dir / "stop_times.txt", dtype=str, keep_default_na=False, na_values=[""], chunksize=_GTFS_CHUNK
    ):
        trips.update(chunk.loc[chunk["stop_id"].isin(inside), "trip_id"])
    return trips


def _gtfs_prefix(df, file: str, prefix: str):
    """Namespace the id columns of one GTFS table (no-op when *prefix* is empty)."""
    if not prefix:
        return df
    for col in _GTFS_ID_COLUMNS.get(file, ()):
        if col in df.columns:
            values = df[col].astype(str)
            df[col] = values.where(values.isin(("", "nan")), prefix + values)
    return df


def _prepare_gtfs(
    gtfs_path: Path,
    bbox: tuple[float, float, float, float] | None,
    inputs_dir: Path,
) -> Path:
    """Normalise an agency GTFS package into one feed directory the engine reads.

    Returns the directory to hand to ``transit.gtfs_path``. The result is cached:
    a stamp records the source hash and the bbox, so a re-run with the same
    inputs skips the (minutes-long) rebuild — same discipline as the OSM cache
    (invariant #4), and the *feed* hash the engine folds into its own transit
    cache key is computed over what this writes.
    """
    import pandas as pd

    if not gtfs_path.exists():
        raise SystemExit(f"--gtfs {gtfs_path} does not exist.")

    out_dir = inputs_dir / "gtfs_feed"
    stamp_path = out_dir / "_segx_source.json"
    stamp = {
        "source": str(gtfs_path),
        "source_sha": _file_sha(gtfs_path) if gtfs_path.is_file() else None,
        "bbox": list(bbox) if bbox else None,
    }
    if stamp_path.exists():
        try:
            if json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
                print(f"  GTFS: reusing prepared feed at {out_dir}")
                return out_dir
        except (OSError, json.JSONDecodeError):
            pass

    import shutil

    work_dir = inputs_dir / "_gtfs_unpack"
    subfeeds = _gtfs_subfeeds(gtfs_path, work_dir)
    prefixes = {name: (f"{name}:" if len(subfeeds) > 1 else "") for name in subfeeds}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    written: set[str] = set()

    def _append(file: str, frame) -> None:
        path = out_dir / file
        frame.to_csv(path, index=False, mode="a" if file in written else "w", header=file not in written)
        written.add(file)

    for name, feed_dir in subfeeds.items():
        prefix = prefixes[name]
        missing = [f for f in ("stops.txt", "routes.txt", "trips.txt", "stop_times.txt") if not (feed_dir / f).exists()]
        if missing:
            print(f"  GTFS: sub-feed {name} is missing {', '.join(missing)} — skipped")
            continue

        keep_trips = _gtfs_trips_in_bbox(feed_dir, bbox)
        trips = pd.read_csv(feed_dir / "trips.txt", dtype=str, keep_default_na=False, na_values=[""])
        if keep_trips is not None:
            trips = trips[trips["trip_id"].isin(keep_trips)]
        if trips.empty:
            print(f"  GTFS: sub-feed {name} has no trip inside the study area — skipped")
            continue
        trip_ids = set(trips["trip_id"])

        # stop_times: streamed, because a city bus feed's is ~100 MB.
        stop_ids: set[str] = set()
        n_stop_times = 0
        for chunk in pd.read_csv(
            feed_dir / "stop_times.txt", dtype=str, keep_default_na=False, na_values=[""], chunksize=_GTFS_CHUNK
        ):
            chunk = chunk[chunk["trip_id"].isin(trip_ids)]
            if chunk.empty:
                continue
            stop_ids.update(chunk["stop_id"])
            n_stop_times += len(chunk)
            _append("stop_times.txt", _gtfs_prefix(chunk.copy(), "stop_times.txt", prefix))

        stops = pd.read_csv(feed_dir / "stops.txt", dtype=str, keep_default_na=False, na_values=[""])
        # keep the called stops *and* their parent stations (transfers use them)
        parents = (
            set(stops.loc[stops["stop_id"].isin(stop_ids), "parent_station"]) if "parent_station" in stops else set()
        )
        stops = stops[stops["stop_id"].isin(stop_ids | {p for p in parents if p and p == p})]
        _append("stops.txt", _gtfs_prefix(stops.copy(), "stops.txt", prefix))

        routes = pd.read_csv(feed_dir / "routes.txt", dtype=str, keep_default_na=False, na_values=[""])
        routes = routes[routes["route_id"].isin(set(trips["route_id"]))]
        _append("routes.txt", _gtfs_prefix(routes.copy(), "routes.txt", prefix))
        _append("trips.txt", _gtfs_prefix(trips.copy(), "trips.txt", prefix))

        services = set(trips["service_id"])
        for file, id_col in (("calendar.txt", "service_id"), ("calendar_dates.txt", "service_id")):
            src = feed_dir / file
            if not src.exists():
                continue
            df = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[""])
            df = df[df[id_col].isin(services)]
            if not df.empty:
                _append(file, _gtfs_prefix(df.copy(), file, prefix))
        freq_src = feed_dir / "frequencies.txt"
        if freq_src.exists():
            df = pd.read_csv(freq_src, dtype=str, keep_default_na=False, na_values=[""])
            df = df[df["trip_id"].isin(trip_ids)]
            if not df.empty:
                _append("frequencies.txt", _gtfs_prefix(df.copy(), "frequencies.txt", prefix))

        print(
            f"  GTFS: {name} -> {len(routes)} routes, {len(trips)} trips, "
            f"{len(stops)} stops, {n_stop_times} stop_times"
            + ("" if keep_trips is None else "  (trips touching the bbox, full sequences kept)")
        )

    if "stop_times.txt" not in written:
        raise SystemExit(
            f"{gtfs_path}: no GTFS trip survived preparation. "
            "Widen --bbox, or check that the zip really contains a feed."
        )
    shutil.rmtree(work_dir, ignore_errors=True)
    stamp_path.write_text(json.dumps(stamp, indent=2), encoding="utf-8")
    print(f"  GTFS: prepared feed written to {out_dir}")
    return out_dir


def _print_summary(state, population, routing) -> None:
    """Console summary — the numbers behind the map."""
    origins = population.origins
    unresolved = int((~origins["demo_resolved"]).sum())
    print("\n" + "=" * 64)
    print("SegX demo - pipeline ran through P3 (routing)")
    print("=" * 64)
    print(f"  working CRS        : {state.boundary.crs}")
    print(f"  demographic rows   : {len(state.demographics.gdf)} ({state.demographics.kind})")
    print(f"  network modes      : {sorted(state.graphs.modes())}")
    print(f"  POIs               : {len(state.pois.gdf)}")
    print(f"  OD pairs           : {len(state.od_pairs.gdf) if state.od_pairs else 0}")
    if state.od_pairs is not None and "mode" in state.od_pairs.gdf.columns:
        by_mode = state.od_pairs.gdf["mode"].value_counts()
        print("    by travel mode   : " + "  ".join(f"{m}={int(n)}" for m, n in by_mode.items()))
    print(f"  origins            : {len(origins)} ({unresolved} without demographics)")
    print(f"  trajectories       : {len(routing.trajectories)}")
    print("\n  reachability by binding (unreachable is recorded, never zeroed):")
    for binding, s in routing.stats.items():
        if binding == "snapping":
            continue
        reasons = ", ".join(f"{k}={v}" for k, v in s["reasons"].items()) or "none"
        print(f"    {binding:10s}: {s['reachable']}/{s['total']} reachable   (unreachable: {reasons})")
    snap = routing.stats.get("snapping", {})
    print(
        f"\n  snapping over {snap.get('tolerance_m')} m: "
        f"origins={snap.get('origins_over_tolerance')}, pois={snap.get('pois_over_tolerance')}"
    )
    print("=" * 64 + "\n")


def _print_transit_summary(state, settings: Settings) -> None:
    """Console block for the P7 transit mode — and its provenance stamp.

    Prints nothing when transit is off. When it is on, the **first** line is the
    source stamp: an ``osm_assumed`` run's headways were invented from config,
    not measured, and that distinction has to be impossible to miss (invariant
    #7). The service window comes next, because ``headway = window_minutes /
    n_trips`` — widening the window *raises* the average headway and makes
    transit look worse, so the number is only readable next to the window that
    produced it (P7.1).
    """
    if "transit" not in state.graphs:
        return
    graph = state.graphs.get("transit")
    meta = graph.graph.get("transit", {})
    source = meta.get("transit_source", "?")
    window = meta.get("transit_service_window", list(settings.transit.service_window))
    bounds = meta.get("transit_service_window_bounds_min", [])

    print("\n" + "=" * 64)
    print("P7 - public transport (transit as a third routing mode)")
    print("=" * 64)
    if source == "osm_assumed":
        print("  source             : OSM  ** HEADWAYS ARE ASSUMED, NOT MEASURED **")
        print(f"                       from transit.osm_assumed_headway_min={settings.transit.osm_assumed_headway_min}")
    else:
        print(f"  source             : GTFS (measured timetable)   feed={meta.get('transit_feed_hash', '?')}")
    span = f"  [{bounds[0]:.0f}-{bounds[1]:.0f} min from midnight]" if len(bounds) == 2 else ""
    day = meta.get("transit_service_day", settings.transit.service_day)
    print(f"  service            : {day}  {window[0]}-{window[1]}{span}")
    print("                       headway = window_minutes / n_trips — a WIDER window raises the")
    print("                       average headway. Run two windows and compare, never read one alone.")
    n_stops, n_patterns = meta.get("transit_n_stops", 0), meta.get("transit_n_patterns", 0)
    print(f"  stops / patterns   : {n_stops} stops, {n_patterns} route-directions")
    print(
        f"  graph              : {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges "
        f"({meta.get('n_platforms', 0)} platforms, {meta.get('n_rides', 0)} rides, "
        f"{meta.get('n_transfers', 0)} transfers)"
    )
    unreachable = meta.get("n_stops_unreachable", 0)
    if unreachable:
        print(f"  stops with no street connector : {unreachable} (beyond transit.max_access_distance_m)")
    if meta.get("map_matched"):
        unmatched = meta.get("n_surface_rides_unmatched", 0)
        note = "" if not unmatched else "  <- these rides deposit no street mass"
        print(f"  surface rides map-matched onto street segments; unmatched: {unmatched}{note}")
    else:
        print("  surface rides NOT map-matched (transit.map_match_surface_rides=false) — bus mass")
        print("  stays on the rail_line layer and never meets pedestrians on the avenue.")
    print("=" * 64 + "\n")


def _dominant_class(vec, attr_name: str, classes: list[str]) -> str:
    """Dominant class label of an origin's demographic vector (for colouring).

    ``demo_vector`` is ``dict[attribute_name, dict[class_label, mass]]``; an
    unresolved origin has NaN masses. Returns the max-mass class, or ``"n/a"``.
    """
    import math

    inner = vec.get(attr_name) if isinstance(vec, dict) else None
    if not inner:
        return "n/a"
    best, best_v = "n/a", None
    for c in classes:
        v = inner.get(c)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if best_v is None or v > best_v:
            best, best_v = c, v
    return best if best_v is not None and best_v > 0 else "n/a"


#: whether to draw the dark web-tile basemap behind maps (set from --no-basemap)
_USE_BASEMAP = True
#: lazily-resolved contextily module (None once we know it's unavailable)
_basemap = {"checked": False, "cx": None}


def _add_basemap(ax, crs) -> None:
    """Draw a dark city basemap (streets + building footprints) behind the data.

    Uses Esri's Dark Gray Canvas tiles (the engine's ``segx.figures`` basemap;
    CartoDB DarkMatter now needs an API key), reprojected to the working CRS so
    they line up with the routes drawn in projected metres. Call it *after* the
    data is plotted (contextily reads the axis extent). Degrades gracefully:
    skipped if contextily is missing or tiles can't be fetched (e.g. offline) —
    the map still renders on the dark background.
    """
    if not _USE_BASEMAP:
        return
    if not _basemap["checked"]:
        _basemap["checked"] = True
        try:
            import contextily as cx

            _basemap["cx"] = cx
        except ImportError:
            print("  (contextily not installed — maps render without a basemap. `uv sync --group demo`.)")
    cx = _basemap["cx"]
    if cx is None:
        return
    try:
        from segx.figures import basemap_provider

        cx.add_basemap(ax, crs=str(crs), source=basemap_provider(), attribution_size=5, zorder=0)
    except Exception as exc:  # tile fetch / network errors must not kill the demo
        print(f"  (basemap skipped — {exc})")


def _draw_network(ax, graph) -> None:
    """Draw a mode graph's edges as a faint backdrop over the dark basemap."""
    xs, ys = [], []
    for u, v in graph.edges():
        xs += [graph.nodes[u]["x"], graph.nodes[v]["x"], None]
        ys += [graph.nodes[u]["y"], graph.nodes[v]["y"], None]
    ax.plot(xs, ys, color="#6a6a6a", linewidth=0.4, alpha=0.45, zorder=1)


def _path_xy(graph, path) -> tuple[list[float], list[float]]:
    """The (x, y) coordinate lists of a trajectory's edge path."""
    nodes = [u for u, _, _ in path] + [path[-1][1]]  # edge sources + final target
    return [graph.nodes[n]["x"] for n in nodes], [graph.nodes[n]["y"] for n in nodes]


def _path_xy_any(state, mode: str, path) -> tuple[list[float], list[float]]:
    """Coordinates of a trajectory's path, tolerating edges from several graphs.

    Needed because ``orchestrator.aggregate`` rewrites transit paths in place
    (``expand_transit_rides``): after aggregation a transit trajectory is a mix
    of walk-graph street edges, drive-graph edges under a map-matched bus ride,
    and the merged graph's own ``rail_line`` edges — no single graph holds them
    all. Edges no graph knows become a ``None`` break in the line rather than a
    ``KeyError``, so a partly-renderable route still draws.
    """
    xs: list[float | None] = []
    ys: list[float | None] = []
    for u, v, key in path:
        xy = _edge_xy_any(state, mode, u, v, key)
        if xy is None:
            xs.append(None)
            ys.append(None)
            continue
        ex, ey = xy
        if xs and xs[-1] is not None and (abs(xs[-1] - ex[0]) > 1e-6 or abs(ys[-1] - ey[0]) > 1e-6):
            xs.append(None)  # discontinuity: don't draw a chord across the city
            ys.append(None)
        xs.extend(ex)
        ys.extend(ey)
    return xs, ys  # type: ignore[return-value]


#: per-income-class line/marker colours, shared across maps
CLASS_COLOUR = {"low": "#4575b4", "mid": "#f4a000", "high": "#d73027", "n/a": "0.6"}

#: per-travel-mode colours, for the multi-mode OD maps (P7)
MODE_COLOUR = {"walk": "#1adb6d", "drive": "#f4a000", "transit": "#00c2d1"}
_MODE_FALLBACK = "#b07aa1"


def _mode_colour(mode: str) -> str:
    return MODE_COLOUR.get(mode, _MODE_FALLBACK)


#: per-POI-category colours (parks/supermarkets/health/bus), with a fallback cycle
CATEGORY_COLOUR = {
    "parks": "#2ca02c",
    "supermarkets": "#ff7f0e",
    "health_centers": "#d62728",
    "bus_stops": "#1f77b4",
}
_CATEGORY_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _category_colour(category: str, i: int) -> str:
    return CATEGORY_COLOUR.get(category, _CATEGORY_FALLBACK[i % len(_CATEGORY_FALLBACK)])


#: home (origin) and work (destination) marker styling, shared across OD maps.
#: colours chosen to read on the dark basemap (work marker is near-white, not black)
_ORIGIN_COLOUR = "#1adb6d"
_DEST_COLOUR = "#f5f5f5"
#: dark edge for light markers so they keep an outline on the basemap
_MARKER_EDGE = "#111111"


def _collect_od_flows(state, routing, settings: Settings):
    """Reachable OD routes as ``(xs, ys, per-class worker counts, mode)``.

    The per-class counts (SE01/SE02/SE03) live on the OD rows; each OD trajectory
    references its row via ``destination_id`` (``"od<idx>"``), so we join back here.
    Each route is drawn on **its own** mode graph (P7: a mode-mixed OD file routes
    walk on the walk graph, car on the drive graph and transit on the merged one),
    and the trajectory's mode is carried out so the maps can colour by it.
    """
    od_gdf = state.od_pairs.gdf
    class_cols = settings.demographics.attributes[0].class_columns or []
    flows: list[tuple[list[float], list[float], list[float], str]] = []
    for traj in routing.trajectories:
        if traj.binding != "od" or not traj.reachable or not traj.path:
            continue
        try:
            idx = int(str(traj.destination_id)[2:])  # "od<idx>"
        except ValueError:
            continue
        if idx not in od_gdf.index:
            continue
        xs, ys = _path_xy_any(state, str(traj.mode), traj.path)
        if not any(x is not None for x in xs):
            continue
        counts = [float(od_gdf.at[idx, c]) if c in od_gdf.columns else 0.0 for c in class_cols]
        flows.append((xs, ys, counts, str(traj.mode)))
    return flows, _backdrop_graph(state, settings)


def _unique_points(coords, decimals: int = 1):
    """De-duplicate (x, y) points (rounded) to keep origin/dest markers legible."""
    seen, out = set(), []
    for x, y in coords:
        key = (round(x, decimals), round(y, decimals))
        if key not in seen:
            seen.add(key)
            out.append((x, y))
    return out


def _endpoints(xs, ys) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """First and last drawable point of a (possibly gapped) polyline."""
    pts = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    return (pts[0], pts[-1]) if pts else None


def _plot_one_od_map(
    flows,
    drive,
    *,
    weights,
    label: str,
    sublabel: str,
    colour: str,
    out_path: Path,
    colour_by_mode: bool = False,
) -> None:
    """Draw one OD map: routes (opacity ∝ *weights*) + home/work markers + legend.

    With ``colour_by_mode`` each route takes its travel mode's colour instead of
    the single *colour* — the picture of a mode-mixed OD file (P7)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vmax = max(weights, default=0.0)
    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, drive)

    origins, dests, total, n_flows = [], [], 0.0, 0
    modes_drawn: dict[str, int] = {}
    for j in sorted(range(len(flows)), key=lambda k: weights[k]):  # heaviest drawn last
        w = weights[j]
        if w <= 0:
            continue
        xs, ys, _, mode = flows[j]
        ends = _endpoints(xs, ys)
        if ends is None:
            continue
        alpha = min(0.04 + 0.66 * (w / vmax if vmax else 0.0), 0.75)
        ax.plot(xs, ys, color=_mode_colour(mode) if colour_by_mode else colour, linewidth=1.0, alpha=alpha, zorder=2)
        origins.append(ends[0])
        dests.append(ends[1])
        modes_drawn[mode] = modes_drawn.get(mode, 0) + 1
        total += w
        n_flows += 1

    if origins:
        ox, oy = zip(*_unique_points(origins), strict=False)
        ax.scatter(ox, oy, marker="o", s=24, facecolor=_ORIGIN_COLOUR, edgecolor=_MARKER_EDGE, linewidth=0.3, zorder=4)
    if dests:
        dx, dy = zip(*_unique_points(dests), strict=False)
        ax.scatter(dx, dy, marker="^", s=32, facecolor=_DEST_COLOUR, edgecolor=_MARKER_EDGE, linewidth=0.3, zorder=5)

    route_handles = (
        [
            Line2D([0], [0], color=_mode_colour(m), lw=2, label=f"{m} ({n} routes)")
            for m, n in sorted(modes_drawn.items())
        ]
        if colour_by_mode
        else [Line2D([0], [0], color=colour, lw=2, label=f"{label} commute route")]
    )
    handles = [
        *route_handles,
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_ORIGIN_COLOUR,
            markeredgecolor="black",
            markersize=8,
            label="home (origin)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=_DEST_COLOUR,
            markeredgecolor=_MARKER_EDGE,
            markersize=9,
            label="work (destination)",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(
        f"SegX — home→work commutes, {label} ({sublabel})\n"
        f"{int(total)} workers across {n_flows} flows (opacity ∝ workers on the route)"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, drive.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _plot_od_maps(state, routing, settings: Settings, out_dir: Path) -> None:
    """One OD map per income class + a combined 'all incomes' map, each showing
    only the home→work commute routes with home/work markers."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not installed — skipping maps. Run `uv sync --group demo`.")
        return
    if state.od_pairs is None:
        print("no OD pairs — skipping OD maps.")
        return

    flows, drive = _collect_od_flows(state, routing, settings)
    class_cols = settings.demographics.attributes[0].class_columns or []
    class_labels = settings.demographics.attributes[0].class_labels

    for i, (col, label) in enumerate(zip(class_cols, class_labels, strict=False)):
        _plot_one_od_map(
            flows,
            drive,
            weights=[f[2][i] for f in flows],
            label=f"{label} income",
            sublabel=col,
            colour=CLASS_COLOUR.get(label, "#333333"),
            out_path=out_dir / f"segx_od_{label}_income.png",
        )

    # combined: all income classes together (opacity ∝ total workers on the route)
    _plot_one_od_map(
        flows,
        drive,
        weights=[sum(f[2]) for f in flows],
        label="all incomes",
        sublabel="+".join(class_cols),
        colour="#5e3c99",
        out_path=out_dir / "segx_od_all_income.png",
    )

    # P7: the same commutes coloured by *travel mode* — only interesting once the
    # OD file carries a mode column, which is also the precondition for
    # `cross_mode_exposure` being anything other than a copy of `exposure`.
    modes = {f[3] for f in flows}
    if len(modes) > 1:
        _plot_one_od_map(
            flows,
            drive,
            weights=[sum(f[2]) for f in flows],
            label="by travel mode",
            sublabel="+".join(sorted(modes)),
            colour="#5e3c99",
            out_path=out_dir / "segx_od_by_mode.png",
            colour_by_mode=True,
        )


def _plot_nearest(state, routing, settings: Settings, out_path: Path) -> None:
    """One map with **all** nearest-POI routes, coloured by POI category.

    Each origin's single closest POI of each category, on the walk graph — the
    full 552 in the Philadelphia demo (not the sampled handful of the overview).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return
    from matplotlib.lines import Line2D

    graph = state.graphs.get(settings.routing.access_mode)
    cats = settings.poi.categories_for("nearest")
    cat_index = {c: i for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)

    counts: dict[str, int] = {}
    for traj in routing.trajectories:
        if traj.binding != "nearest" or not traj.reachable or not traj.path:
            continue
        colour = _category_colour(traj.category, cat_index.get(traj.category, 0))
        xs, ys = _path_xy(graph, traj.path)
        ax.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.55, zorder=2)
        counts[traj.category] = counts.get(traj.category, 0) + 1

    pois = state.pois.gdf.to_crs(state.boundary.crs)
    for i, cat in enumerate(cats):
        sub = pois[pois["category"] == cat]
        if not sub.empty:
            ax.scatter(
                sub.geometry.x,
                sub.geometry.y,
                marker="*",
                s=55,
                color=_category_colour(cat, i),
                edgecolor="black",
                linewidth=0.3,
                zorder=4,
            )

    handles = [
        Line2D([0], [0], color=_category_colour(c, i), lw=2, label=f"{c} ({counts.get(c, 0)} routes)")
        for i, c in enumerate(cats)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, title="nearest route + POIs (★)")
    ax.set_title(
        f"SegX — nearest-POI routes (all {sum(counts.values())}), by category\n"
        f"walk access; each origin's closest POI of each category"
    )
    ax.set_aspect("equal")
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _plot_catchment_by_category(state, routing, settings: Settings, out_dir: Path) -> None:
    """One catchment map per POI category: **all** routes from origins to every POI
    of that category reachable within the budget, plus that category's POIs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping maps. Run `uv sync --group demo`.")
        return
    from matplotlib.lines import Line2D

    graph = state.graphs.get(settings.routing.access_mode)
    budget = settings.poi.catchment.budget_minutes
    pois = state.pois.gdf.to_crs(state.boundary.crs)

    by_cat: dict[str, list] = {}
    for traj in routing.trajectories:
        if traj.binding == "catchment" and traj.reachable and traj.path:
            by_cat.setdefault(traj.category, []).append(traj)

    for i, cat in enumerate(settings.poi.categories_for("catchment")):
        trajs = by_cat.get(cat, [])
        colour = _category_colour(cat, i)
        fig, ax = plt.subplots(figsize=(11, 11))
        _draw_network(ax, graph)

        origins = []
        for traj in trajs:
            xs, ys = _path_xy(graph, traj.path)
            ax.plot(xs, ys, color=colour, linewidth=0.6, alpha=0.18, zorder=2)
            origins.append((xs[0], ys[0]))
        if origins:
            ox, oy = zip(*_unique_points(origins), strict=False)
            ax.scatter(ox, oy, marker="o", s=16, facecolor=_ORIGIN_COLOUR, edgecolor="black", linewidth=0.25, zorder=3)

        sub = pois[pois["category"] == cat]
        ax.scatter(
            sub.geometry.x, sub.geometry.y, marker="*", s=70, color=colour, edgecolor="black", linewidth=0.3, zorder=4
        )

        handles = [
            Line2D([0], [0], color=colour, lw=2, label=f"reach route ({len(trajs)})"),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=colour,
                markeredgecolor="black",
                markersize=11,
                label=f"{cat} POIs ({len(sub)})",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_ORIGIN_COLOUR,
                markeredgecolor="black",
                markersize=8,
                label="home (origin)",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title(
            f"SegX — {budget:g}-min walk catchment: {cat}\n"
            f"{len(trajs)} reachable (origin, POI) pairs to {len(sub)} {cat} POIs"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("metres (working CRS)")
        ax.set_ylabel("metres (working CRS)")
        _add_basemap(ax, graph.graph.get("crs"))
        fig.tight_layout()
        out_path = out_dir / f"segx_catchment_{cat}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"map written to {out_path}")


def _plot(state, population, routing, settings: Settings, out_path: Path, *, max_paths: int = 60) -> None:
    """Render network + origins (by dominant income class) + POIs + sample routes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return

    attr_name = settings.demographics.attributes[0].name
    classes = settings.demographics.attributes[0].class_labels
    fig, ax = plt.subplots(figsize=(11, 11))

    # 1. street network as a faint backdrop
    _draw_network(ax, _backdrop_graph(state, settings))

    # 2. sample trajectory paths, coloured by binding
    colours = {"nearest": "#1f77b4", "catchment": "#2ca02c", "od": "#d62728"}
    bindings = sorted({t.binding for t in routing.trajectories}) or list(colours)
    drawn = {k: 0 for k in colours}
    per_binding_cap = max(1, max_paths // len(bindings))
    for traj in routing.trajectories:
        if not traj.reachable or not traj.path or drawn.get(traj.binding, per_binding_cap) >= per_binding_cap:
            continue
        xs, ys = _path_xy_any(state, str(traj.mode), traj.path)
        ax.plot(xs, ys, color=colours[traj.binding], linewidth=1.0, alpha=0.6, zorder=2)
        drawn[traj.binding] += 1

    # 3. POIs (none in an --od-only run: the access bindings are off)
    pois = state.pois.gdf.to_crs(state.boundary.crs) if len(state.pois.gdf) else state.pois.gdf
    if len(pois):
        ax.scatter(
            pois.geometry.x,
            pois.geometry.y,
            marker="*",
            s=70,
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.3,
            zorder=4,
            label="POIs",
        )

    # 4. origins, coloured by dominant income class
    origins = population.origins.to_crs(state.boundary.crs)
    dom = [_dominant_class(v, attr_name, classes) for v in origins["demo_vector"]]
    ax.scatter(
        origins.geometry.x,
        origins.geometry.y,
        c=[CLASS_COLOUR.get(d, "0.6") for d in dom],
        s=18,
        edgecolor="black",
        linewidth=0.2,
        zorder=3,
    )

    # legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([0], [0], color=c, lw=2, label=f"{b} route") for b, c in colours.items() if drawn.get(b)]
    handles += [
        Patch(facecolor=CLASS_COLOUR[c], edgecolor="black", label=f"origin: {c} income") for c in ("low", "mid", "high")
    ]
    if len(pois):
        handles += [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#ff7f0e",
                markeredgecolor="black",
                markersize=12,
                label="POI",
            )
        ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_title("SegX — routed trajectories on a real street network (LODES demo)")
    ax.set_aspect("equal")
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, state.boundary.crs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"map written to {out_path}")


# --------------------------------------------------------------------------- #
# P4 — per-segment aggregation + diversity (Netto 2018, figs. 3 & 5)           #
# --------------------------------------------------------------------------- #


def _edge_line_xy(graph, u, v, key) -> tuple[list[float], list[float]]:
    """Coordinates of one edge — its OSM geometry if present, else a straight u→v."""
    data = graph.get_edge_data(u, v, key) or {}
    geom = data.get("geometry")
    if geom is not None:
        xs, ys = geom.xy
        return list(xs), list(ys)
    return [graph.nodes[u]["x"], graph.nodes[v]["x"]], [graph.nodes[u]["y"], graph.nodes[v]["y"]]


def _edge_xy_any(state, mode: str, u, v, key) -> tuple[list[float], list[float]] | None:
    """Coordinates of one aggregated edge, whichever mode graph actually holds it.

    A transit trajectory's edges are **not** all in one graph after P7's
    ``expand_transit_rides``: its walking legs are walk-graph ids, its metro
    rides keep the merged graph's own ``rail_line`` edges, and its *surface*
    rides were map-matched onto the **drive** graph. So the renderer tries the
    row's own mode first and then the street graphs, and returns ``None`` for an
    edge no graph knows — a skipped line, never a crash mid-figure.
    """
    for candidate in (mode, "transit", "drive", "walk"):
        if candidate not in state.graphs:
            continue
        graph = state.graphs.get(candidate)
        if graph.has_node(u) and graph.has_node(v):
            return _edge_line_xy(graph, u, v, key)
    return None


def _backdrop_graph(state, settings: Settings):
    """The graph drawn as the faint street backdrop behind every segment map.

    Always a plain street graph (never the merged transit one, whose synthetic
    platform/route-stop edges would draw as spurious straight lines across the
    city)."""
    for mode in ("walk", "drive"):
        if mode in state.graphs:
            return state.graphs.get(mode)
    return state.graphs.get(sorted(state.graphs.modes())[0])


def _print_p4_summary(aggregation) -> None:
    """Console summary of the P4 aggregation + diversity + conservation."""
    dv = aggregation.metrics["diversity"]
    print("\n" + "=" * 64)
    print("P4 - per-segment aggregation + diversity (Netto 2018)")
    print("=" * 64)
    print(f"  segments carrying mass : {aggregation.stats['n_segments']}")
    print(f"  partitions             : {len(aggregation.stats['partitions'])}")
    led = aggregation.ledger
    ok = "OK" if abs(led.emitted - led.deposited - led.unreachable) < 1e-6 else "FAIL"
    print(
        f"  weight conservation    : emitted={led.emitted:.0f} deposited={led.deposited:.0f} "
        f"unreachable={led.unreachable:.0f}  -> {ok}"
    )
    if not dv.empty:
        od = dv[dv["binding"] == "od"]["diversity"].dropna()
        if len(od):
            print(f"  OD segment diversity   : min={od.min():.2f} mean={od.mean():.2f} max={od.max():.2f}")
    print("=" * 64 + "\n")


def _segment_partition(aggregation, settings: Settings, binding: str):
    """Return ``(attr_name, modes, wide_masses_df, diversity_df)`` for one binding.

    ``wide_masses_df`` is indexed by ``(mode, u, v, key)`` with one column per
    class; ``diversity_df`` is the metric output filtered to the same binding.

    ``mode`` stays in the index because it is a **partition key** (invariant #6):
    once the OD file mixes modes, one binding spans two or three edge-id spaces
    and collapsing them would silently add a walk-graph segment's mass to a
    drive-graph one that happens to share ``(u, v, key)``. ``modes`` is the list
    present, most-travelled first.
    """
    attr = settings.demographics.attributes[0].name
    et = aggregation.edge_table
    sub = et[(et["binding"] == binding) & (et["attribute"] == attr)]
    modes = list(sub["mode"].value_counts().index) if not sub.empty else []
    wide = sub.pivot_table(
        index=["mode", "u", "v", "key"], columns="class", values="mass", aggfunc="sum", fill_value=0.0
    )
    # `None` when `diversity` is switched off — the dominant-class map only needs
    # the masses, so it must not fail on a metric it never reads.
    dv = aggregation.metrics.get("diversity")
    if dv is not None:
        dv = dv[(dv["binding"] == binding) & (dv["attribute"] == attr)]
    return attr, modes, wide, dv


def _plot_dominant_segments(state, aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> dict:
    """Map: each street segment coloured by its **dominant income group** — the
    'dominant class network' of Netto 2018 fig. 3. Returns per-class segment counts."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return {}

    attr, modes, wide, _ = _segment_partition(aggregation, settings, binding)
    classes = settings.demographics.attributes[0].class_labels
    if wide.empty:
        print(f"no {binding} segments — skipping dominant map.")
        return {}
    graph = _backdrop_graph(state, settings)

    lines, colours, counts = [], [], {c: 0 for c in classes}
    for (mode, u, v, key), row in wide.iterrows():
        masses = [float(row.get(c, 0.0)) for c in classes]
        if sum(masses) <= 0:
            continue
        xy = _edge_xy_any(state, mode, u, v, key)
        if xy is None:
            continue
        dom = classes[int(np.argmax(masses))]
        lines.append(list(zip(*xy, strict=False)))
        colours.append(CLASS_COLOUR[dom])
        counts[dom] += 1

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)
    ax.add_collection(LineCollection(lines, colors=colours, linewidths=1.3, zorder=2))
    handles = [Line2D([0], [0], color=CLASS_COLOUR[c], lw=3, label=f"{c} dominant ({counts[c]})") for c in classes]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, title=f"dominant {attr} on segment")
    ax.set_title(
        f"SegX — dominant income group per street segment ({binding}; modes: {', '.join(modes) or 'n/a'})\n"
        f"the group with the most agents colours the segment"
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")
    return counts


def _plot_diversity_segments(state, aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Map: each segment coloured by its **diversity** ``D = -Σ Pᵢ log₂(Pᵢ/Pₜ)`` —
    the entropy map of Netto 2018 fig. 5 (yellow = low, red = high)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return

    attr, _modes, _, dv = _segment_partition(aggregation, settings, binding)
    if dv.empty:
        print(f"no {binding} segments — skipping diversity map.")
        return
    graph = _backdrop_graph(state, settings)
    values = dv["diversity"].to_numpy(dtype=float)
    vmax = float(np.nanpercentile(values, 97)) or 1.0  # clip a few outliers for contrast
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")

    lines, colours = [], []
    for _, r in dv.iterrows():
        xy = _edge_xy_any(state, str(r["mode"]), r["u"], r["v"], r["key"])
        if xy is None:
            continue
        lines.append(list(zip(*xy, strict=False)))
        colours.append(cmap(norm(min(float(r["diversity"]), vmax))))

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)
    ax.add_collection(LineCollection(lines, colors=colours, linewidths=1.3, zorder=2))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"segment diversity  D = -Σ Pᵢ log₂(Pᵢ/Pₜ)   (clipped at p97={vmax:.0f})")
    ax.set_title(f"SegX — social diversity per street segment ({binding}, {attr})\nred = most income-mixed streets")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _chart_diversity_distribution(aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Chart: histogram of per-segment diversity values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, _, _, dv = _segment_partition(aggregation, settings, binding)
    vals = dv["diversity"].dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vals, bins=40, color="#d73027", edgecolor="white")
    ax.axvline(vals.mean(), color="black", linestyle="--", linewidth=1, label=f"mean = {vals.mean():.1f}")
    ax.set_xlabel("segment diversity  D = -Σ Pᵢ log₂(Pᵢ/Pₜ)")
    ax.set_ylabel("number of street segments")
    ax.set_title(f"SegX — distribution of per-segment social diversity ({binding})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart written to {out_path}")


def _chart_dominant_counts(counts: dict, settings: Settings, out_path: Path) -> None:
    """Chart: how many street segments each income group dominates."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not counts:
        return
    classes = settings.demographics.attributes[0].class_labels
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(classes, [counts.get(c, 0) for c in classes], color=[CLASS_COLOUR[c] for c in classes], edgecolor="black")
    total = sum(counts.values()) or 1
    for i, c in enumerate(classes):
        n = counts.get(c, 0)
        ax.text(i, n, f"{n}\n{100 * n / total:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("number of segments dominated")
    ax.set_title("SegX — street segments by dominant income group (OD commutes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart written to {out_path}")


def _export_tables(aggregation, settings: Settings, tables_dir: Path) -> None:
    """Write the P4 metric outputs as CSV for downstream inspection."""
    aggregation.metrics["diversity"].to_csv(tables_dir / "diversity_by_segment.csv", index=False)
    # per-partition summary: segment count + mean diversity
    dv = aggregation.metrics["diversity"]
    if not dv.empty:
        summary = (
            dv.groupby(["mode", "category", "binding"])
            .agg(n_segments=("diversity", "size"), mean_diversity=("diversity", "mean"))
            .reset_index()
        )
        summary.to_csv(tables_dir / "partition_summary.csv", index=False)
    print(f"tables written to {tables_dir}")


# --------------------------------------------------------------------------- #
# P5 — segregation metrics (the five P5 measures)                              #
# --------------------------------------------------------------------------- #
# The orchestrator already runs every registered metric; these renderers just  #
# surface the P5 outputs (`aggregation.metrics` + `aggregation.summaries`) that #
# the P4 renderers above ignore. All default to the OD/commute binding.        #


def _metric_frame(aggregation, settings: Settings, name: str, binding: str = "od"):
    """A P5 metric's frame filtered to one binding + the primary attribute.

    Returns ``(attr_name, modes, filtered_df)``. The frame keeps its ``mode``
    column: with a mode-mixed OD file one binding spans several edge-id spaces,
    so every renderer resolves an edge's geometry per row (:func:`_edge_xy_any`)
    instead of assuming one graph."""
    attr = settings.demographics.attributes[0].name
    df = aggregation.metrics[name]
    df = df[(df["binding"] == binding) & (df["attribute"] == attr)]
    modes = list(df["mode"].value_counts().index) if not df.empty else []
    return attr, modes, df


def _plot_segment_scalar(
    state,
    aggregation,
    settings: Settings,
    name: str,
    value_col: str,
    out_path: Path,
    *,
    cmap_name: str,
    label: str,
    title: str,
    binding: str = "od",
    vmax: float | None = None,
) -> None:
    """Generic per-segment colourmap map for a scalar metric (``value_col`` per
    segment). Segments with ``NaN`` (no mass — invariant #2) are left undrawn."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return

    _, _modes, df = _metric_frame(aggregation, settings, name, binding)
    if df.empty or df[value_col].dropna().empty:
        print(f"no {binding} segments for {name} — skipping map.")
        return
    graph = _backdrop_graph(state, settings)
    vals = df[value_col].to_numpy(dtype=float)
    top = vmax if vmax is not None else (float(np.nanpercentile(vals, 97)) or 1.0)
    norm = Normalize(vmin=0.0, vmax=top)
    cmap = plt.get_cmap(cmap_name)

    lines, colours = [], []
    for _, r in df.iterrows():
        v = float(r[value_col])
        if np.isnan(v):
            continue
        xy = _edge_xy_any(state, str(r["mode"]), r["u"], r["v"], r["key"])
        if xy is None:
            continue
        lines.append(list(zip(*xy, strict=False)))
        colours.append(cmap(norm(min(v, top))))

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)
    ax.add_collection(LineCollection(lines, colors=colours, linewidths=1.3, zorder=2))
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _plot_dominant_trajectories(state, aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Map: dominant group per segment **weighted by dominance** ``1 − h_norm`` —
    the P5 `dominant_trajectories` metric ("Dynamic segregation").
    Unlike the P4 `dominant_income.png` (which just colours by the winner), here the
    line's **opacity** encodes how sharply that group dominates: a 33/33/34 segment
    is near-invisible, a 0/0/100 one is solid."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return

    attr, _modes, df = _metric_frame(aggregation, settings, "dominant_trajectories", binding)
    if df.empty:
        print(f"no {binding} segments for dominant_trajectories — skipping map.")
        return
    graph = _backdrop_graph(state, settings)
    classes = settings.demographics.attributes[0].class_labels

    lines, colours = [], []
    for _, r in df.iterrows():
        dom = r["dominant_class"]
        w = float(r["dominance"])
        if dom is None or np.isnan(w):
            continue
        xy = _edge_xy_any(state, str(r["mode"]), r["u"], r["v"], r["key"])
        if xy is None:
            continue
        base = CLASS_COLOUR.get(str(dom), "0.6")
        colours.append(to_rgba(base, alpha=max(0.12, min(1.0, w))))  # dominance-weighted opacity
        lines.append(list(zip(*xy, strict=False)))

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)
    ax.add_collection(LineCollection(lines, colors=colours, linewidths=1.4, zorder=2))
    handles = [Line2D([0], [0], color=CLASS_COLOUR[c], lw=3, label=c) for c in classes]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, title=f"dominant {attr}")
    ax.set_title(
        f"SegX — dominant trajectories per segment ({binding}, {attr})\n"
        "P5 dynamic segregation: colour = dominant group, opacity = dominance (1 − entropia norm.)"
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _plot_segregated_segments(state, aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Map: the **most exclusive segments** — the P5 `segregated_trajectories`
    metric. Non-exclusive segments are drawn faint; segments whose dominant group's
    location quotient ``LQ ≥ exclusive_lq_threshold`` are highlighted, coloured by
    that group ("most exclusive trajectories")."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — skipping map. Run `uv sync --group demo`.")
        return

    attr, _modes, df = _metric_frame(aggregation, settings, "segregated_trajectories", binding)
    if df.empty:
        print(f"no {binding} segments for segregated_trajectories — skipping map.")
        return
    graph = _backdrop_graph(state, settings)
    classes = settings.demographics.attributes[0].class_labels
    threshold = settings.metrics.exclusive_lq_threshold

    faint, hot, hot_colours, counts = [], [], [], {c: 0 for c in classes}
    for _, r in df.iterrows():
        xy = _edge_xy_any(state, str(r["mode"]), r["u"], r["v"], r["key"])
        if xy is None:
            continue
        seg = list(zip(*xy, strict=False))
        if bool(r["exclusive"]) and r["dominant_class"] in CLASS_COLOUR:
            hot.append(seg)
            hot_colours.append(CLASS_COLOUR[str(r["dominant_class"])])
            counts[str(r["dominant_class"])] += 1
        else:
            faint.append(seg)

    fig, ax = plt.subplots(figsize=(11, 11))
    _draw_network(ax, graph)
    ax.add_collection(LineCollection(faint, colors="#888888", linewidths=0.7, alpha=0.5, zorder=2))
    ax.add_collection(LineCollection(hot, colors=hot_colours, linewidths=2.2, zorder=3))
    n_exclusive = sum(counts.values())
    handles = [Line2D([0], [0], color=CLASS_COLOUR[c], lw=3, label=f"{c} ({counts[c]})") for c in classes]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, title=f"exclusive to {attr} group")
    ax.set_title(
        f"SegX — most exclusive segments ({binding}, {attr})\n"
        f"P5: {n_exclusive} segments with dominant-group LQ ≥ {threshold:g} (over-represented vs. the city)"
    )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("metres (working CRS)")
    ax.set_ylabel("metres (working CRS)")
    _add_basemap(ax, graph.graph.get("crs"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"map written to {out_path}")


def _chart_exposure_by_group(aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Chart: **social exposure per group** — the P5 `exposure` metric. Exposure is
    a per-group value (not a segment map): the mean segment diversity each group
    encounters along its trajectories, weighted by presence. One bar per group,
    and one **panel per travel mode**: `mode` is a partition key, so a mode-mixed
    OD file gives one exposure value per (mode, group) and flattening them would
    silently plot whichever row happened to come last."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    attr = settings.demographics.attributes[0].name
    classes = settings.demographics.attributes[0].class_labels
    df = aggregation.metrics["exposure"]
    df = df[(df["binding"] == binding) & (df["attribute"] == attr)]
    if df.empty:
        print(f"no {binding} exposure values — skipping chart.")
        return
    modes = list(dict.fromkeys(df["mode"]))

    fig, axes = plt.subplots(1, len(modes), figsize=(3.6 * len(modes) + 2, 5), squeeze=False, sharey=True)
    for ax, mode in zip(axes[0], modes, strict=True):
        values = {str(r["class"]): float(r["exposure"]) for _, r in df[df["mode"] == mode].iterrows()}
        heights = [values.get(c, float("nan")) for c in classes]
        ax.bar(classes, heights, color=[CLASS_COLOUR[c] for c in classes], edgecolor="black")
        for i, h in enumerate(heights):
            if h == h:  # not NaN
                ax.text(i, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"mode: {mode}", fontsize=10)
    axes[0][0].set_ylabel("mean segment diversity encountered (bits)")
    fig.suptitle(
        f"SegX — social exposure per {attr} group ({binding})\nP5: diversity a group meets on its trajectories",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart written to {out_path}")


def _chart_cross_mode_exposure(aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Chart: `exposure` vs `cross_mode_exposure` per group, one panel per mode (P7.1).

    `exposure` measures the diversity a group meets among **its own mode's**
    travellers; `cross_mode_exposure` re-reads the same edge table counting the
    other modes on the same street at `metrics.cross_mode.factor` (default 0.5).
    Read the *pair*, never the second number alone. Three things to expect and
    not mistake for bugs:

    * the two bars being equal is what a **failed join** looks like as well as
      what "the modes never meet" looks like — check the cross-mode audit block
      printed above (``stats["cross_mode"]``) before believing either;
    * cross-mode exposure can land *below* plain exposure, when the mode joining
      in is more concentrated than the one observing;
    * the effect is strongly asymmetric — a handful of bus riders barely move
      the pedestrians' number while the pedestrians dominate theirs.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    if "cross_mode_exposure" not in aggregation.metrics:
        return
    attr = settings.demographics.attributes[0].name
    classes = settings.demographics.attributes[0].class_labels

    def _prep(name: str):
        df = aggregation.metrics[name]
        return df[(df["binding"] == binding) & (df["attribute"] == attr)]

    base, cross = _prep("exposure"), _prep("cross_mode_exposure")
    if cross.empty:
        print(f"no {binding} cross_mode_exposure values — skipping chart.")
        return
    modes = list(dict.fromkeys(cross["mode"]))

    fig, axes = plt.subplots(1, len(modes), figsize=(4.2 * len(modes) + 1.5, 5), squeeze=False, sharey=True)
    x = np.arange(len(classes))
    for ax, mode in zip(axes[0], modes, strict=True):
        b = {str(r["class"]): float(r["exposure"]) for _, r in base[base["mode"] == mode].iterrows()}
        c = {str(r["class"]): float(r["cross_mode_exposure"]) for _, r in cross[cross["mode"] == mode].iterrows()}
        ax.bar(x - 0.2, [b.get(k, np.nan) for k in classes], 0.4, label="exposure (own mode)", color="#4c72b0")
        ax.bar(x + 0.2, [c.get(k, np.nan) for k in classes], 0.4, label="cross-mode", color="#00c2d1")
        ax.set_xticks(x, classes)
        ax.set_title(f"observer mode: {mode}", fontsize=10)
    axes[0][0].set_ylabel("mean segment diversity encountered (bits)")
    axes[0][0].legend(fontsize=8, framealpha=0.9)
    factor = settings.metrics.cross_mode.factor
    fig.suptitle(
        f"SegX — social exposure with and without cross-mode co-presence ({binding}, {attr})\n"
        f"P7.1: other modes on the same street counted at factor={factor:g}  "
        "(equal bars = the modes never share a segment — check the audit)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart written to {out_path}")


def _print_cross_mode_summary(aggregation, settings: Settings, *, binding: str = "od") -> None:
    """Console block for `cross_mode_exposure` + the audit that makes it readable.

    The audit matters more than the metric: a cross-mode join that found nothing
    does not surface as a zero, it surfaces as ``cross_mode_exposure ==
    exposure``, which reads like the finding "the modes never meet" (P7.1).
    So the join rate is printed first, and a zero rate is called out.
    """
    audit = aggregation.stats.get("cross_mode", {})
    print("\n" + "=" * 64)
    print("P7.1 - cross-mode exposure (co-presence between travel modes)")
    print("=" * 64)
    n_modes = audit.get("n_modes_present", 0)
    rate = audit.get("cross_mode_join_rate", 0.0)
    print(f"  modes in the edge table       : {n_modes}")
    print(
        f"  street segments               : {audit.get('n_segments_total', 0)}"
        f"   of which multi-mode: {audit.get('n_segments_multimode', 0)}"
    )
    print(f"  cross-mode join rate          : {rate:.1%}")
    # P7.2: the scalar above is exactly what hid the `walk x drive = 0` result —
    # it read 64% while the pair that mattered was zero. Print the pairs.
    for pair, info in sorted((audit.get("pair_join_rates") or {}).items()):
        flag = "  <- NEVER MEET" if not info.get("n_shared") else ""
        print(f"    {pair:<24}: {info.get('n_shared', 0)} shared pools ({info.get('rate', 0.0):.1%}){flag}")
    space = audit.get("street_space")
    if space:
        params = space.get("params", {})
        print(
            f"  street-space corridors        : {space.get('n_corridors', 0)}"
            f"   multi-mode: {space.get('n_corridors_multimode', 0)}"
            f"   match rate: {space.get('corridor_match_rate', 0.0):.1%}"
        )
        print(
            f"    anchored on {space.get('anchor_mode')!r}, tolerance {params.get('tolerance_m')} m, "
            f"bearing <= {params.get('max_bearing_deg')} deg"
        )
        print("    (a matched corridor is an INFERENCE, not data - these parameters ship in the metadata)")
    print(
        f"  factor                        : {settings.metrics.cross_mode.factor:g}"
        + (f"   overrides={settings.metrics.cross_mode.overrides}" if settings.metrics.cross_mode.overrides else "")
    )
    if "n_surface_rides_unmatched" in audit:
        print(f"  surface rides not map-matched : {audit['n_surface_rides_unmatched']}")
    if n_modes <= 1:
        print("  -> one mode only: cross_mode_exposure is identical to exposure by construction.")
        print("     Give the OD file a `mode` column (--mode-split) to make it non-trivial.")
    elif rate == 0.0:
        print("  -> WARNING: several modes, but no street segment carries more than one of them.")
        print("     Every value below equals plain `exposure`. That may be the finding, or it may")
        print("     mean the per-mode edge ids never matched (walk/drive graphs are simplified")
        print("     independently, so their (u, v, key) spaces can diverge). Do not report it as")
        print("     'the modes never meet' without checking.")

    # The join rate broken out by *which* modes met. A single percentage cannot
    # tell "36 % of streets are pedestrian side-streets" from "36 % of streets
    # are where the id spaces failed to match"; this table can.
    from segx.metrics.cross_mode_exposure import cross_mode_combination_counts

    combos = cross_mode_combination_counts(aggregation.edge_table)
    if not combos.empty:
        total = int(combos["n_segments"].sum())
        print("\n  segments by exact mode combination (a segment is counted once):")
        print(f"    {'layer':<10}{'modes':<26}{'segments':>9}   {'share':>6}")
        for _, r in combos.iterrows():
            n = int(r["n_segments"])
            print(f"    {str(r['layer']):<10}{str(r['modes']):<26}{n:>9}   {n / total:>5.1%}")
        print(f"    {'':<10}{'TOTAL':<26}{total:>9}")

    attr = settings.demographics.attributes[0].name
    cme = aggregation.metrics.get("cross_mode_exposure")
    exp = aggregation.metrics.get("exposure")
    if cme is None or cme.empty:
        print("=" * 64 + "\n")
        return
    cme = cme[(cme["binding"] == binding) & (cme["attribute"] == attr)]
    exp = exp[(exp["binding"] == binding) & (exp["attribute"] == attr)]
    lookup = {(str(r["mode"]), str(r["class"])): float(r["exposure"]) for _, r in exp.iterrows()}
    print(f"\n  per group ({binding}, {attr}) — exposure -> cross-mode exposure:")
    for mode in dict.fromkeys(cme["mode"]):
        parts = []
        for _, r in cme[cme["mode"] == mode].iterrows():
            klass = str(r["class"])
            before = lookup.get((str(mode), klass), float("nan"))
            parts.append(f"{klass}: {before:.2f}->{float(r['cross_mode_exposure']):.2f}")
        print(f"    {str(mode):8s}  " + "   ".join(parts))
    print("=" * 64 + "\n")


def _print_p5_summary(aggregation, settings: Settings, *, binding: str = "od") -> None:
    """Console summary of the five P5 measures: the network-wide indicators plus
    per-group exposure and the exclusive-segment tally.

    Everything is broken out **per travel mode**, because `mode` is a partition
    key (invariant #6): with a mode-mixed OD file each metric has one row per
    mode, and printing them in a flat list repeats ``low/mid/high`` three times
    with no way to tell which is which — or, worse, quietly shows only the first.
    """
    s = aggregation.summaries
    attr = settings.demographics.attributes[0].name
    print("\n" + "=" * 64)
    print("P5 - segregation metrics (network-wide indicators)")
    print("=" * 64)

    def _filtered(frame):
        if frame is None or frame.empty:
            return None
        out = frame[(frame["binding"] == binding) & (frame["attribute"] == attr)]
        return None if out.empty else out

    exposure = _filtered(aggregation.metrics.get("exposure"))
    modes = list(dict.fromkeys(exposure["mode"])) if exposure is not None else []
    if len(modes) > 1:
        print(f"  one row per travel mode ({', '.join(modes)}) — they are separate partitions,")
        print("  not a breakdown of one number; the OD file's `mode` column is what splits them.\n")

    rows = [
        ("experienced segregation (1 - h) ", "experienced_segregation", "experienced_segregation"),
        ("intergroup contact (1 - Sum pi^2)", "intergroup_contact", "intergroup_contact"),
        ("dominance (1 - h_norm)          ", "dominant_trajectories", "dominance"),
    ]
    for label, name, col in rows:
        df = _filtered(s.get(name))
        if df is None:
            print(f"  {label}: n/a")
            continue
        parts = "   ".join(f"{str(r['mode'])}={float(r[col]):.3f}" for _, r in df.iterrows())
        print(f"  {label}: {parts}")

    # exposure — one value per social group, per mode
    if exposure is not None:
        print("  social exposure by group          :")
        for mode in modes:
            parts = "  ".join(
                f"{row['class']}={float(row['exposure']):.2f}"
                for _, row in exposure[exposure["mode"] == mode].iterrows()
            )
            print(f"    {str(mode):10s} {parts}")

    # segregated — count of exclusive segments per dominant group, per mode
    seg = _filtered(s.get("segregated_trajectories"))
    if seg is not None:
        thr = settings.metrics.exclusive_lq_threshold
        total = int(seg["n_exclusive"].sum())
        print(f"  exclusive segments (LQ>={thr:g})         : {total}")
        for mode in dict.fromkeys(seg["mode"]):
            by_group = "  ".join(
                f"{row['dominant_class']}={int(row['n_exclusive'])}"
                for _, row in seg[seg["mode"] == mode].iterrows()
                if int(row["n_exclusive"]) > 0
            )
            if by_group:
                print(f"    {str(mode):10s} {by_group}")
    print("=" * 64 + "\n")


def _export_p5_tables(aggregation, settings: Settings, tables_dir: Path) -> None:
    """Write every P5 metric's per-segment frame + its network-wide summary as CSV."""
    p5 = [
        "experienced_segregation",
        "intergroup_contact",
        "dominant_trajectories",
        "segregated_trajectories",
        "exposure",
        "cross_mode_exposure",  # P7.1
    ]
    for name in p5:
        if name in aggregation.metrics:
            aggregation.metrics[name].to_csv(tables_dir / f"{name}_by_segment.csv", index=False)
        if name in aggregation.summaries and not aggregation.summaries[name].empty:
            aggregation.summaries[name].to_csv(tables_dir / f"{name}_summary.csv", index=False)
    # The cross-mode join audit travels with the numbers it qualifies: a reader
    # who has `cross_mode_exposure_by_segment.csv` and not this file cannot tell
    # a real "the modes don't meet" from an id-space mismatch (P7.1).
    audit = aggregation.stats.get("cross_mode")
    if audit:
        (tables_dir / "cross_mode_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    from segx.metrics.cross_mode_exposure import cross_mode_combination_counts

    combos = cross_mode_combination_counts(aggregation.edge_table)
    if not combos.empty:
        combos.to_csv(tables_dir / "cross_mode_segments_by_combination.csv", index=False)
    print(f"P5 tables written to {tables_dir}")


# --------------------------------------------------------------------------- #
# P7.3 — path fragmentation (the one measure that is per-path, not per-segment) #
# --------------------------------------------------------------------------- #
def _fragmentation_frame(aggregation, settings: Settings, binding: str):
    """The `path_fragmentation` rows for one binding and the first attribute."""
    df = aggregation.metrics.get("path_fragmentation")
    if df is None or df.empty:
        return None
    attr = settings.demographics.attributes[0].name
    df = df[(df["binding"] == binding) & (df["attribute"] == attr)]
    return None if df.empty else df


def _print_fragmentation_summary(aggregation, settings: Settings, *, binding: str = "od") -> None:
    """Console block for `path_fragmentation` + the coverage audit that qualifies it.

    The audit comes first for the same reason the cross-mode one does: the
    failure mode of this metric is not a zero, it is a column of ``NaN`` — or a
    mean taken over a small fraction of the paths — and only the counts say so.
    The plain readings (`mean_segments`, `detour_ratio`) are printed beside `D`
    because `D` is *how the winding changes with scale*, not how winding a path
    is: a grid staircase scores about the same as a straight line.
    """
    audit = aggregation.stats.get("fragmentation", {})
    cfg = settings.metrics.fragmentation
    print("\n" + "=" * 64)
    print("P7.3 - path fragmentation (fractal dimension of each trajectory)")
    print("=" * 64)
    if cfg.method == "box_counting":
        sizes = ", ".join(f"{cfg.pixel_m * cfg.pixel_factor**i:g}" for i in range(cfg.n_pixel_scales))
        print(f"  method                        : box counting, pixels of {sizes} m")
        convention = "1 - slope (as written: straight line = 2)" if cfg.as_written else "-slope (straight line ~ 1)"
        print(f"  D convention                  : {convention}")
        saturated = "dropped from the fit" if cfg.drop_saturated else "kept in the fit"
        print(f"  single-pixel resolutions      : {saturated}")
        print(f"  street geometry               : {'traced' if cfg.use_edge_geometry else 'node-to-node chords'}")
    else:
        print(
            f"  method                        : divider, {cfg.n_scales} rulers from "
            f"{cfg.largest_fraction:g} x path length, ratio {cfg.scale_ratio:g}"
        )
    n_measured = audit.get("n_measured", 0)
    not_measured = audit.get("n_saturated", 0) + audit.get("n_too_short", 0)
    n_geometry = n_measured + not_measured
    share = f" ({n_measured / n_geometry:.1%})" if n_geometry else ""
    print(f"  paths measured                : {n_measured} of {n_geometry}{share}")
    print(
        f"    too small for the ladder: {not_measured}   no geometry: {audit.get('n_no_geometry', 0)}"
        f"   unreachable: {audit.get('n_unreachable', 0)}   no demographics: {audit.get('n_unresolved', 0)}"
    )
    if n_geometry and n_measured == 0:
        print("  -> WARNING: no path carried a fractal dimension; every D below is NaN.")

    df = _fragmentation_frame(aggregation, settings, binding)
    if df is None:
        print(f"  no {binding} paths.")
        print("=" * 64 + "\n")
        return
    print(f"\n  per group ({binding}) — presence-weighted means over the group's paths:")
    print(f"    {'mode':<9}{'class':<8}{'D':>7}{'sd':>7}{'segments':>10}{'length m':>10}{'detour':>8}{'paths':>8}")
    for _, r in df.iterrows():
        print(
            f"    {str(r['mode']):<9}{str(r['class']):<8}{float(r['fragmentation']):>7.3f}"
            f"{float(r['fragmentation_sd']):>7.3f}{float(r['mean_segments']):>10.1f}"
            f"{float(r['mean_length_m']):>10.0f}{float(r['detour_ratio']):>8.2f}{int(r['n_paths']):>8}"
        )
    print("  (D = how the winding changes with scale; `segments` and `detour` are the plain readings)")
    print("=" * 64 + "\n")


def _chart_path_fragmentation(aggregation, settings: Settings, out_path: Path, *, binding: str = "od") -> None:
    """Chart: `D` and mean street segments per group, one column per travel mode (P7.3).

    Two rows on purpose — the fractal dimension on top, Netto's literal "number
    of street segments used" below — because the two can disagree and neither
    should be shown alone.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    df = _fragmentation_frame(aggregation, settings, binding)
    if df is None:
        print(f"no {binding} path_fragmentation values — skipping chart.")
        return
    attr = settings.demographics.attributes[0].name
    classes = settings.demographics.attributes[0].class_labels
    modes = list(dict.fromkeys(df["mode"]))
    rows = [
        ("fragmentation", "fractal dimension D", "{:.3f}"),
        ("mean_segments", "mean street segments per path", "{:.1f}"),
    ]

    fig, axes = plt.subplots(2, len(modes), figsize=(3.6 * len(modes) + 2, 8), squeeze=False, sharey="row")
    for j, mode in enumerate(modes):
        sub = df[df["mode"] == mode].set_index(df[df["mode"] == mode]["class"].astype(str))
        for i, (col, _label, fmt) in enumerate(rows):
            ax = axes[i][j]
            heights = [float(sub[col].get(c, float("nan"))) for c in classes]
            ax.bar(classes, heights, color=[CLASS_COLOUR[c] for c in classes], edgecolor="black")
            for k, h in enumerate(heights):
                if h == h:  # not NaN
                    ax.text(k, h, fmt.format(h), ha="center", va="bottom", fontsize=9)
            if i == 0:
                ax.set_title(f"mode: {mode}", fontsize=10)
                # D differs between groups in the second or third decimal: a
                # zero-based axis would draw three identical bars
                finite = [h for h in df["fragmentation"].astype(float) if h == h]
                if finite:
                    pad = max(0.02, 0.25 * (max(finite) - min(finite)))
                    ax.set_ylim(min(finite) - pad, max(finite) + pad)
    for i, (_col, label, _fmt) in enumerate(rows):
        axes[i][0].set_ylabel(label)
    cfg = settings.metrics.fragmentation
    how = f"box counting, {cfg.pixel_m:g} m x {cfg.pixel_factor}^k" if cfg.method == "box_counting" else "divider"
    fig.suptitle(
        f"SegX — path fragmentation per {attr} group ({binding})\nP7.3: {how}  (the D axis does not start at zero)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"chart written to {out_path}")


def _export_fragmentation_tables(aggregation, tables_dir: Path) -> None:
    """Write the per-group table and the coverage audit + ladder parameters.

    The audit travels with the numbers it qualifies (invariant #7): two `D`s
    measured under different ladders are indistinguishable once they are rows.
    """
    df = aggregation.metrics.get("path_fragmentation")
    if df is None:
        return
    df.to_csv(tables_dir / "path_fragmentation_by_group.csv", index=False)
    audit = aggregation.stats.get("fragmentation")
    if audit:
        (tables_dir / "path_fragmentation_audit.json").write_text(
            json.dumps(audit, indent=2, default=str), encoding="utf-8"
        )
    print(f"P7.3 tables written to {tables_dir}")


# --------------------------------------------------------------------------- #
# P4.5 — route-set comparison: link-penalty vs. exact k-shortest (Yen)         #
# --------------------------------------------------------------------------- #


def _canon_edge(e: tuple[int, int, int]) -> tuple[int, int, int]:
    u, v, k = e
    return (u, v, k) if u <= v else (v, u, k)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _routeset_stats(routes: list) -> dict:
    """Distinctness / detour stats for one method's route-set on one OD pair."""
    n = len(routes)
    edge_sets = [{_canon_edge(e) for e in r.edges} for r in routes]
    union: set = set().union(*edge_sets) if edge_sets else set()
    cost1 = routes[0].cost if n else 0.0
    detours = [routes[i].cost / cost1 - 1.0 for i in range(1, n) if cost1 > 0]
    mean_detour = sum(detours) / len(detours) if detours else 0.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    overlaps = [_jaccard(edge_sets[i], edge_sets[j]) for i, j in pairs]
    internal = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {"n": n, "cost1": cost1, "mean_detour": mean_detour, "internal": internal, "union": union, "sets": edge_sets}


def _compare_route_sets(state, settings: Settings, out, *, sample: int, k: int, penalty: float, seed: int) -> None:
    """Route both algorithms on a random sample of OD pairs and compare them, so
    the user can see the trade-off and choose. Writes a per-pair CSV, a
    summary CSV, and a side-by-side figure of a few example pairs."""
    import csv
    import random
    import time

    from segx.orchestrator import _routable_od_pairs
    from segx.routing.route_set import build_route_set

    if state.od_pairs is None:
        print("no OD pairs — skipping route-set comparison.")
        return

    scheme = settings.routing.route_set.weight_scheme
    beta = settings.routing.route_set.logit_beta
    # Sample OD *rows* first, then snap only those — snapping all N flows (can be
    # tens of thousands) just to route a handful is wasteful. Oversample (×4) so
    # unreachable pairs don't starve the target sample.
    gdf = state.od_pairs.gdf
    n_take = min(len(gdf), max(sample * 4, sample))
    sample_gdf = gdf.sample(n=n_take, random_state=seed) if len(gdf) > n_take else gdf
    pairs = _routable_od_pairs(sample_gdf, state.graphs, settings, working_crs=state.boundary.crs)
    random.Random(seed).shuffle(pairs)

    methods = ("link_penalty", "yen")
    per_pair: list[dict] = []
    examples: list[tuple] = []
    runtime = {m: 0.0 for m in methods}
    cross_union_jac: list[float] = []
    route1_identical = 0
    n_used = 0

    print(f"\ncomparing route-sets on up to {sample} OD pairs (k={k}, penalty={penalty}, weight={scheme})…")
    for pair in pairs:
        if n_used >= sample:
            break
        graph = state.graphs.get(pair.mode)
        routes_by_method = {}
        reachable = True
        for method in methods:
            t0 = time.perf_counter()
            routes = build_route_set(
                graph,
                pair.origin_node,
                pair.dest_node,
                weight="travel_time_min",
                method=method,
                k=k,
                penalty=penalty,
                weight_scheme=scheme,
                logit_beta=beta,
            )
            runtime[method] += time.perf_counter() - t0
            if not routes:
                reachable = False
                break
            routes_by_method[method] = routes
        if not reachable:
            continue
        n_used += 1
        stats = {m: _routeset_stats(routes_by_method[m]) for m in methods}
        for m in methods:
            s = stats[m]
            per_pair.append(
                {
                    "pair": f"{pair.origin_id}->{pair.dest_id}",
                    "method": m,
                    "n_routes": s["n"],
                    "cost1_min": round(s["cost1"], 3),
                    "mean_detour_pct": round(100 * s["mean_detour"], 2),
                    "internal_overlap": round(s["internal"], 3),
                }
            )
        cross_union_jac.append(_jaccard(stats["link_penalty"]["union"], stats["yen"]["union"]))
        if stats["link_penalty"]["sets"] and stats["link_penalty"]["sets"][0] == stats["yen"]["sets"][0]:
            route1_identical += 1
        examples.append(
            (pair, routes_by_method["link_penalty"], routes_by_method["yen"], stats["link_penalty"]["cost1"])
        )

    if n_used == 0:
        print("no reachable sampled OD pairs — widen --bbox or lower --sample.")
        return

    lp, yen = "link_penalty", "yen"

    def _avg(method: str, key: str) -> float:
        vals = [r[key] for r in per_pair if r["method"] == method]
        return sum(vals) / len(vals) if vals else 0.0

    # ---- console summary table ----
    def _row(label: str, a: float, b: float, fmt: str = "{:>16.2f}") -> None:
        print(f"{label:<36}" + fmt.format(a) + fmt.format(b))

    cross = sum(cross_union_jac) / len(cross_union_jac)
    print("\n" + "=" * 68)
    print(f"ROUTE-SET COMPARISON — {n_used} OD pairs, k={k}")
    print("=" * 68)
    print(f"{'metric':<36}{'link_penalty':>16}{'yen (exact)':>16}")
    print("-" * 68)
    _row("routes per pair (mean)", _avg(lp, "n_routes"), _avg(yen, "n_routes"))
    _row("mean detour of alternatives (%)", _avg(lp, "mean_detour_pct"), _avg(yen, "mean_detour_pct"))
    _row(
        "internal overlap (lower = more distinct)",
        _avg(lp, "internal_overlap"),
        _avg(yen, "internal_overlap"),
        "{:>16.3f}",
    )
    _row("total time (s)", runtime[lp], runtime[yen], "{:>16.3f}")
    _row("time per pair (ms)", 1000 * runtime[lp] / n_used, 1000 * runtime[yen] / n_used)
    print("-" * 68)
    print(f"agreement between methods (Jaccard of route unions): {cross:.3f}")
    print(
        f"route 1 identical across methods: {route1_identical}/{n_used} pairs ({100 * route1_identical / n_used:.0f}%)"
    )
    print("=" * 68 + "\n")

    # ---- CSV outputs ----
    tables_dir = out["tables"]
    fieldnames = ["pair", "method", "n_routes", "cost1_min", "mean_detour_pct", "internal_overlap"]
    with (tables_dir / "routeset_comparison_by_pair.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_pair)
    with (tables_dir / "routeset_comparison_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "link_penalty", "yen"])
        writer.writerow(["n_pairs", n_used, n_used])
        for metric, key in [
            ("mean_routes_per_pair", "n_routes"),
            ("mean_detour_pct", "mean_detour_pct"),
            ("mean_internal_overlap", "internal_overlap"),
        ]:
            writer.writerow([metric, round(_avg(lp, key), 3), round(_avg(yen, key), 3)])
        writer.writerow(["total_runtime_s", round(runtime[lp], 4), round(runtime[yen], 4)])
        writer.writerow(["cross_method_union_jaccard", round(cross, 4), ""])
        writer.writerow(["route1_identical_frac", round(route1_identical / n_used, 4), ""])
    print(f"comparison tables written to {tables_dir}")

    # ---- side-by-side figure of the longest example pairs (most visible spread) ----
    examples.sort(key=lambda e: e[3], reverse=True)
    _plot_routeset_comparison(state, examples[:4], out["maps"] / "routeset_comparison.png", k=k)


#: per-route colours (segments used by exactly one route)
_RANK_COLOURS = ["#00e5ff", "#ffb000", "#ff5da2", "#7cff6b", "#c77dff"]
#: overlap colours — segments SHARED by several routes, keyed by how many share
#: them; deliberately outside the neon per-route palette so they read as "shared".
_SHARED_COLOURS = {2: "#ffffff", 3: "#ff3b30", 4: "#ffe000", 5: "#b0ff00"}


def _shared_colour(count: int) -> str:
    return _SHARED_COLOURS.get(count, _SHARED_COLOURS[max(_SHARED_COLOURS)])


def _classify_route_edges(graph, routes) -> tuple[dict[int, list], dict[int, list], set[int]]:
    """Split a route-set's edges into per-route-unique vs. shared-by-N.

    Returns ``(unique_lines_by_rank, shared_lines_by_count, shared_counts)`` where
    each value list holds edge polylines (lists of (x, y)) ready for a
    ``LineCollection``. An edge on several routes is drawn once, in the shared
    colour — so overlap is explicit instead of one route hiding another."""
    edge_ranks: dict[tuple[int, int, int], tuple[set[int], tuple[int, int, int]]] = {}
    for rank, r in enumerate(routes):
        for u, v, key in r.edges:
            ce = _canon_edge((u, v, key))
            entry = edge_ranks.setdefault(ce, (set(), (u, v, key)))
            entry[0].add(rank)
    unique: dict[int, list] = {}
    shared: dict[int, list] = {}
    counts: set[int] = set()
    for ranks, rep in edge_ranks.values():
        xs, ys = _edge_line_xy(graph, *rep)
        line = list(zip(xs, ys, strict=False))
        if len(ranks) == 1:
            unique.setdefault(next(iter(ranks)), []).append(line)
        else:
            shared.setdefault(len(ranks), []).append(line)
            counts.add(len(ranks))
    return unique, shared, counts


def _plot_routeset_comparison(state, examples: list, out_path: Path, *, k: int) -> None:
    """Rows = example OD pairs; columns = link-penalty vs. Yen. Segments used by a
    single route take that route's colour; segments **shared** by several routes
    take a distinct overlap colour (white → red → …, by how many share them). Makes
    the 'distinct vs. near-duplicate' difference explicit."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed — skipping route-set figure. Run `uv sync --group demo`.")
        return
    if not examples:
        return

    # round caps/joins so a route's consecutive segments read as one continuous line
    line_kw = {"capstyle": "round", "joinstyle": "round", "alpha": 1.0}
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows), squeeze=False)
    for row, (pair, lp_routes, yen_routes, cost1) in enumerate(examples):
        graph = state.graphs.get(pair.mode)
        for col, (title, routes) in enumerate((("link-penalty", lp_routes), ("Yen (exact)", yen_routes))):
            ax = axes[row][col]
            _draw_network(ax, graph)
            unique, shared, shared_counts = _classify_route_edges(graph, routes)
            # per-route-unique segments (thinner), then the shared trunk on top (thicker)
            for rank, lines in sorted(unique.items()):
                colour = _RANK_COLOURS[rank % len(_RANK_COLOURS)]
                ax.add_collection(LineCollection(lines, colors=[colour], linewidths=2.4, zorder=3, **line_kw))
            for count, lines in sorted(shared.items()):
                lw = 3.2 + 1.6 * (count - 1)  # more routes sharing → thicker trunk
                ax.add_collection(
                    LineCollection(lines, colors=[_shared_colour(count)], linewidths=lw, zorder=4, **line_kw)
                )
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=_RANK_COLOURS[i % len(_RANK_COLOURS)],
                    lw=3,
                    label=f"route {i + 1} (w={routes[i].weight:.2f}, +{100 * (routes[i].cost / cost1 - 1):.0f}%)",
                )
                for i in range(len(routes))
            ]
            handles += [
                Line2D([0], [0], color=_shared_colour(c), lw=5, label=f"shared by {c} routes")
                for c in sorted(shared_counts)
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)
            ax.set_title(f"{pair.origin_id}→{pair.dest_id} — {title}\n{len(routes)} routes, geodesic={cost1:.1f} min")
            ax.set_aspect("equal")
            ax.autoscale()
            _add_basemap(ax, state.boundary.crs)
    fig.suptitle(
        f"SegX — route set per OD pair: link-penalty vs. exact k-shortest (k={k})\n"
        "colour per route; shared stretches in white/red — link-penalty spreads, Yen near-duplicates",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"route-set comparison figure written to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="SegX LODES end-to-end demo (through P4).")
    p.add_argument("--geometry", required=True, type=Path, help="Census-block polygons (TIGER GPKG/SHP/GeoJSON).")
    p.add_argument("--od", required=True, type=Path, help="LODES OD CSV.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML config (default: demo/segx_lodes.toml).")
    p.add_argument("--out", type=Path, default=DEMO_DIR / "out", help="Output directory.")
    p.add_argument("--bbox", type=str, default=None, help="Optional WGS84 clip 'minx,miny,maxx,maxy'.")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Override config: dotted.key=value.")
    p.add_argument(
        "--no-basemap",
        action="store_true",
        help="Skip the dark web-tile city basemap (use for offline runs; maps still render on a dark background).",
    )
    # --- P4.5 route-set comparison (link-penalty vs. exact Yen k-shortest) ---
    p.add_argument(
        "--compare-routesets",
        action="store_true",
        help="Route a random sample of OD pairs with BOTH algorithms and write a comparison (table + CSV + figure).",
    )
    p.add_argument("--sample", type=int, default=40, help="Route-set comparison: number of OD pairs to sample.")
    p.add_argument("--k", type=int, default=3, help="Route-set comparison: routes per OD pair (k).")
    p.add_argument("--penalty", type=float, default=1.5, help="Route-set comparison: link-penalty factor (>1).")
    p.add_argument("--seed", type=int, default=None, help="Route-set comparison: sampling seed (default: config seed).")
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Run ONLY the route-set comparison (skip the full routing + maps). Implies --compare-routesets.",
    )
    # --- binding selection: OD pairs only (no POI access bindings) ---
    p.add_argument(
        "--od-only",
        action="store_true",
        help=(
            "Route ONLY the home→work OD pairs: turns off the nearest-POI and catchment bindings "
            "and fetches no POIs at all. Faster, and the only binding for which cross_mode_exposure "
            "is non-trivial (the access bindings all use one routing.access_mode)."
        ),
    )
    # --- travel mode on the OD file (P7 / P7.1) ---
    p.add_argument(
        "--mode-split",
        type=str,
        default=None,
        help=(
            "Give the OD file a SYNTHETIC travel-mode column, e.g. 'car=0.45,transit=0.35,walk=0.20' "
            "(LODES carries none). Shares are normalised; assignment is a seeded hash of the OD pair, "
            "so it is stable across runs and bboxes. Ignored if the source file already has the column. "
            "Implied by --transit. Provenance is written to inputs/od_clip_modes.json."
        ),
    )
    # --- public transport (P7) ---
    p.add_argument(
        "--transit",
        action="store_true",
        help=(
            "Enable public transport as a third routing mode (P7): builds the merged walk + transit"
            "graph and routes OD pairs whose mode is transit/bus/metro on it. Implies --mode-split "
            "(so there is transit demand to route) unless the OD file already has a mode column."
        ),
    )
    p.add_argument(
        "--gtfs",
        type=Path,
        default=None,
        help=(
            "GTFS feed for --transit: a zip, an extracted directory, or an agency zip-of-zips "
            "(e.g. SEPTA's gtfs_public.zip). Without it the run falls back to OSM, whose headways "
            "are ASSUMED, not measured, and is stamped transit_source=osm_assumed."
        ),
    )
    p.add_argument(
        "--service-window",
        type=str,
        default=None,
        help=(
            "Headway measurement window as 'HH:MM-HH:MM' (default from config: 06:00-09:00). "
            "headway = window_minutes / n_trips, so a WIDER window raises the average headway — "
            "run two windows and compare. 'end <= start' crosses midnight."
        ),
    )
    p.add_argument(
        "--cross-mode-factor",
        type=float,
        default=None,
        help=(
            "Weight given to another mode's mass in cross_mode_exposure (default 0.5). "
            "0 reproduces `exposure` exactly; 1 pools every mode as one crowd."
        ),
    )
    p.add_argument(
        "--disable-metric",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Switch one metric off (repeatable), e.g. --disable-metric path_fragmentation. "
            "Shorthand for --set metrics.disabled=[...]; its outputs are skipped."
        ),
    )
    # The console summary is the demo's main output and is often piped to a
    # file. On Windows a
    # redirected stdout defaults to cp1252, which raises UnicodeEncodeError on the
    # very characters these summaries are written in (arrows, Σ, ≥). Force UTF-8
    # before anything prints, so `> out.txt` behaves the same everywhere.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = p.parse_args()

    import warnings

    # Pandana is noisy about unconnected node pairs (a real, handled signal here —
    # those routes are simply recorded unreachable). Quieten it for a clean demo.
    warnings.filterwarnings("ignore", message="Unsigned integer: shortest path")

    # Dark theme for every figure (chart + map), and the city basemap toggle.
    global _USE_BASEMAP
    _USE_BASEMAP = not args.no_basemap
    try:
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")
    except ImportError:
        pass  # maps/charts are skipped later with a message if matplotlib is absent

    bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
    out = _make_layout(args.out)

    # Flag-derived config overrides are applied *first* so an explicit --set
    # always wins: the flags are shorthand for exactly these keys, nothing more.
    flag_overrides: dict[str, object] = {}
    if args.od_only:
        # Off at the binding *and* at the fetch: no POI category is requested, so
        # `io.pois.load_pois` has nothing to pull from Overpass either.
        flag_overrides |= {
            "poi.nearest.enabled": False,
            "poi.catchment.enabled": False,
            "poi.nearest.categories": [],
            "poi.catchment.categories": [],
        }
    if args.transit:
        flag_overrides["transit.enabled"] = True
        if args.gtfs is None:
            print("WARNING: --transit without --gtfs — falling back to OSM. Headways will be ASSUMED,")
            print("         not measured, and the run is stamped transit_source=osm_assumed.")
    if args.service_window:
        start, _, end = args.service_window.partition("-")
        if not end:
            raise SystemExit("--service-window: expected 'HH:MM-HH:MM'")
        flag_overrides["transit.service_window"] = [start.strip(), end.strip()]
    if args.cross_mode_factor is not None:
        flag_overrides["metrics.cross_mode.factor"] = args.cross_mode_factor
    if args.disable_metric:
        flag_overrides["metrics.disabled"] = list(args.disable_metric)

    settings = Settings.load(config_path=args.config, overrides={**flag_overrides, **_parse_overrides(args.overrides)})

    # --transit needs transit *demand*: without a mode column every OD flow keeps
    # the config's default mode (commute -> car) and the transit graph is built
    # and never used, which looks like a bug and is only a missing column.
    mode_split = _parse_mode_split(args.mode_split) if args.mode_split else None
    if mode_split is None and args.transit:
        mode_split = dict(DEFAULT_MODE_SPLIT)

    print("preparing inputs (clip area + match OD flows)…")
    blocks_path, od_path, boundary_path = _prepare_inputs(
        args.geometry, args.od, bbox, settings, out["inputs"], mode_split=mode_split
    )

    # Build exactly the graphs the OD file asks for — and refuse *now*, with the
    # fix in the message, if it asks for transit and transit is off. Discovering
    # that inside P3 (after the OSM fetch) is the same error twenty minutes late.
    needed = _required_graph_modes(od_path, settings)
    if "transit" in needed and not settings.transit.enabled:
        raise SystemExit(
            f"the OD file routes {sorted(needed)} but transit is disabled.\n"
            "  Either add --transit (with --gtfs <feed> for measured headways, or without it for\n"
            "  OSM-assumed ones), or keep the OD file to street modes, e.g.\n"
            "  --mode-split 'car=0.7,walk=0.3'."
        )
    settings = settings._with_overrides({"network.modes": sorted({"walk", "drive"} | needed)})

    if args.transit and args.gtfs is not None:
        settings = settings._with_overrides({"transit.gtfs_path": str(_prepare_gtfs(args.gtfs, bbox, out["inputs"]))})

    # ---- pipeline: P1 ingest -> P2 population -> P3 routing -> P4 aggregation ----
    what = "street networks" if args.od_only else "street networks + POIs"
    if args.transit:
        what += " + the merged walk+transit graph"
    print(f"building {what} (first run for a bbox fetches live OSM — can take a few minutes)…")
    state = ingest(
        settings,
        boundary_path=boundary_path,
        demographics_path=blocks_path,
        od_pairs_path=od_path,
        cache_dir=out["cache"],
    )
    # ---- P4.5 route-set comparison (optional; only needs P1 ingest) ----
    if args.compare_routesets or args.compare_only:
        _compare_route_sets(
            state,
            settings,
            out,
            sample=args.sample,
            k=args.k,
            penalty=args.penalty,
            seed=args.seed if args.seed is not None else settings.random_seed,
        )
        if args.compare_only:
            print("--compare-only: skipping the full routing + maps.")
            return

    population = build_population(state, settings)
    routing = route(state, population, settings)
    aggregation = aggregate(state, population, routing, settings)

    # ---- P8 deliverables: the same files `segx run` writes, from the same frames
    # every map/chart/table below reads ----
    outputs = write(state, routing, aggregation, settings, out["deliverables"], od_spread=analyse_od(state, settings))
    print(
        f"deliverables: {outputs.segments_path.name} ({', '.join(outputs.layers) or 'no layers'}), "
        f"{outputs.workbook_path.name} ({len(outputs.sheets)} sheets) in {outputs.output_dir}"
    )
    if outputs.stats["n_segments_no_geometry"]:
        print(f"  warning: {outputs.stats['n_segments_no_geometry']} segment(s) written without geometry")

    _print_summary(state, population, routing)
    _print_transit_summary(state, settings)

    # A metric switched off in the config (`metrics.disabled` / `metrics.enabled`)
    # is absent from `aggregation.metrics`; its console block, maps, charts and
    # tables are skipped rather than failing the run.
    def on(*names: str) -> bool:
        return all(name in aggregation.metrics for name in names)

    off = sorted(set(available_metrics()) - set(aggregation.metrics))
    if off:
        print(f"metrics switched off for this run: {', '.join(off)} — their outputs are skipped.")
    if on("diversity"):
        _print_p4_summary(aggregation)
    _print_p5_summary(aggregation, settings)
    if on("cross_mode_exposure"):
        _print_cross_mode_summary(aggregation, settings)
    if on("path_fragmentation"):
        _print_fragmentation_summary(aggregation, settings)

    # ---- P3 routing maps ----
    _plot(state, population, routing, settings, out["maps"] / "overview.png")
    if not args.od_only:
        _plot_nearest(state, routing, settings, out["maps"] / "nearest.png")
        _plot_catchment_by_category(state, routing, settings, out["maps"])
    _plot_od_maps(state, routing, settings, out["maps"])

    # ---- P4 metric maps + charts + tables ----
    counts = _plot_dominant_segments(state, aggregation, settings, out["segments"] / "dominant_income.png")
    if on("diversity"):
        _plot_diversity_segments(state, aggregation, settings, out["segments"] / "diversity.png")
        _chart_diversity_distribution(aggregation, settings, out["charts"] / "diversity_distribution.png")
        _export_tables(aggregation, settings, out["tables"])
    _chart_dominant_counts(counts, settings, out["charts"] / "dominant_class_counts.png")

    # ---- P5 segregation-metric maps + chart + tables ----
    if on("experienced_segregation"):
        _plot_segment_scalar(
            state,
            aggregation,
            settings,
            "experienced_segregation",
            "experienced_segregation",
            out["segments"] / "experienced_segregation.png",
            cmap_name="magma",
            vmax=1.0,
            label="experienced segregation  1 − h  (1 = homogeneous)",
            title=(
                "SegX — experienced segregation per segment (od, income)\n"
                "P5: social homogeneity along trajectories (1 − entropia norm.)"
            ),
        )
    if on("intergroup_contact"):
        _plot_segment_scalar(
            state,
            aggregation,
            settings,
            "intergroup_contact",
            "intergroup_contact",
            out["segments"] / "intergroup_contact.png",
            cmap_name="viridis",
            vmax=1.0,
            label="intergroup contact  1 − Σ pᵢ²  (Gini–Simpson)",
            title=(
                "SegX — intergroup contact probability per segment (od, income)\n"
                "P5: chance two people on the segment differ in group"
            ),
        )
    if on("dominant_trajectories"):
        _plot_dominant_trajectories(state, aggregation, settings, out["segments"] / "dominant_trajectories.png")
    if on("segregated_trajectories"):
        _plot_segregated_segments(state, aggregation, settings, out["segments"] / "segregated_segments.png")
    if on("exposure"):
        _chart_exposure_by_group(aggregation, settings, out["charts"] / "exposure_by_group.png")
    # ---- P7.1 cross-mode exposure ----
    _chart_cross_mode_exposure(aggregation, settings, out["charts"] / "cross_mode_exposure.png")
    _export_p5_tables(aggregation, settings, out["tables"])
    # ---- P7.3 path fragmentation ----
    _chart_path_fragmentation(aggregation, settings, out["charts"] / "path_fragmentation.png")
    _export_fragmentation_tables(aggregation, out["tables"])


if __name__ == "__main__":
    main()
