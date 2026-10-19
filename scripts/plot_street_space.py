"""Visual validation of the P7.2 canonical street space (corridor matching).

Renders four figures, each answering one question a reader will actually ask
before trusting `cross_mode_exposure`:

1. ``01_three_graphs.png``   — are ``walk`` and ``drive`` really different
   geometries? (side by side, then overlaid)
2. ``02_corridors.png``      — which edges ended up in the same corridor?
   (one colour per corridor, plus a zoom where individual sidewalks are legible)
3. ``03_attachment.png``     — *what got attached to what*: a hairline from every
   matched edge's midpoint to its anchor. This is the figure that catches a
   mis-tuned tolerance, because a wrong attachment is visibly a long diagonal.
4. ``04_unmatched.png``      — the residual, coloured by ``highway``. The misses
   have to be legible as legitimate (interior alleys, plazas, steps) rather than
   as failures.

Runs entirely off the on-disk OSM cache — no Overpass call, no routing. Point it
at a cache directory and it picks the ``walk`` / ``drive`` entries out of it.

Usage::

    uv run python scripts/plot_street_space.py
    uv run python scripts/plot_street_space.py --tolerance-m 30 --max-bearing-deg 45
    uv run python scripts/plot_street_space.py --cache-dir demo/out/cache --out-dir demo/out/street_space
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/plot_street_space.py`
    sys.path.insert(0, str(REPO_ROOT))

from segx.config import Settings  # noqa: E402
from segx.network.builder import build_mode_graph  # noqa: E402
from segx.network.cache import read_graph  # noqa: E402
from segx.network.registry import GraphRegistry  # noqa: E402
from segx.network.street_space import (  # noqa: E402
    edges_as_gdf,
    resolve_street_space,
)

WALK_COLOUR = "#2b6cb0"
DRIVE_COLOUR = "#c53030"
MISS_COLOUR = "#b8b8b8"


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def _cache_entries(cache_dir: Path) -> dict[str, Path]:
    """``network_type -> cache entry dir``, read from each entry's manifest."""
    found: dict[str, Path] = {}
    for manifest_path in sorted((cache_dir / "osm").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        found.setdefault(str(manifest.get("network_type")), manifest_path.parent)
    return found


def load_graphs(cache_dir: Path, settings: Settings, working_crs: str) -> GraphRegistry:
    entries = _cache_entries(cache_dir)
    missing = [m for m in ("walk", "drive") if m not in entries]
    if missing:
        raise SystemExit(
            f"cache at {cache_dir} has no entry for {missing}; found {sorted(entries)}. "
            "Run the demo once so the OSM extracts are cached."
        )
    graphs = GraphRegistry()
    for mode in ("walk", "drive"):
        entry = entries[mode]
        manifest = read_manifest_dir(entry)
        raw = read_graph(_entry_obj(entry, manifest))
        graphs.register(mode, build_mode_graph(raw, mode, settings, working_crs=working_crs))
    return graphs


def read_manifest_dir(entry_dir: Path) -> dict:
    return json.loads((entry_dir / "manifest.json").read_text())


def _entry_obj(entry_dir: Path, manifest: dict):
    """Adapt a bare directory to whatever shape :func:`read_graph` expects."""
    from segx.network.cache import cache_entry

    return cache_entry(entry_dir.parent.parent, entry_dir.name)


# --------------------------------------------------------------------------- #
# Drawing helpers                                                              #
# --------------------------------------------------------------------------- #


def _segments(gdf) -> list[np.ndarray]:
    return [np.asarray(geom.coords) for geom in gdf.geometry]


def _draw(ax, gdf, *, color, lw=0.6, alpha=1.0, zorder=1):
    if len(gdf) == 0:
        return
    ax.add_collection(LineCollection(_segments(gdf), colors=color, linewidths=lw, alpha=alpha, zorder=zorder))


def _draw_coloured(ax, gdf, colours, *, lw=0.9, alpha=1.0, zorder=2):
    if len(gdf) == 0:
        return
    ax.add_collection(LineCollection(_segments(gdf), colors=colours, linewidths=lw, alpha=alpha, zorder=zorder))


def _finish(ax, bounds, title, subtitle=""):
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#dddddd")
    # Title above the subtitle, both above the frame. `pad` has to clear the
    # subtitle's own line box or the two overprint each other.
    ax.set_title(title, fontsize=11, loc="left", pad=20)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=8, color="#666666", va="bottom", ha="left")


def _corridor_colours(corridor_ids, cmap_name="tab20"):
    """A stable pseudo-random colour per corridor id.

    Deliberately *not* a sequential colormap: the question these figures answer is
    "are these two edges in the same group", which is categorical.

    ``crc32``, not ``hash()`` — Python randomises string hashing per process, so
    ``hash()`` would repaint the map on every run and two renderings of the same
    data could not be compared side by side.
    """
    cmap = matplotlib.colormaps[cmap_name]
    return [cmap(zlib.crc32(str(cid).encode()) % cmap.N) for cid in corridor_ids]


def _zoom_bounds(bounds, *, frac=0.16, cx=0.34, cy=0.52):
    """A small window inside *bounds*, big enough to hold a few blocks."""
    minx, miny, maxx, maxy = bounds
    w, h = (maxx - minx), (maxy - miny)
    zx, zy = minx + cx * w, miny + cy * h
    return (zx, zy, zx + frac * w, zy + frac * h * (h / w) / (h / w))


# --------------------------------------------------------------------------- #
# Figures                                                                      #
# --------------------------------------------------------------------------- #


def fig_three_graphs(walk, drive, bounds, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    _draw(axes[0], walk, color=WALK_COLOUR, lw=0.5)
    _finish(axes[0], bounds, f"walk — {len(walk)} edges", "footways, service roads, walkable streets")
    _draw(axes[1], drive, color=DRIVE_COLOUR, lw=0.7)
    _finish(axes[1], bounds, f"drive — {len(drive)} edges", "roadway centrelines only")
    _draw(axes[2], walk, color=WALK_COLOUR, lw=0.5, alpha=0.75)
    _draw(axes[2], drive, color=DRIVE_COLOUR, lw=0.9, alpha=0.85, zorder=3)
    _finish(
        axes[2],
        bounds,
        "overlaid",
        "blue rarely sits ON red — the sidewalk is its own way. No id remap can join them.",
    )
    fig.suptitle(
        "P7.2 validation 1/4 — walk and drive are different geometries, not two ids for one way",
        fontsize=13,
        x=0.012,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def fig_corridors(edges, bounds, out: Path) -> None:
    matched = edges[edges["matched"]]
    unmatched = edges[~edges["matched"]]
    zoom = _zoom_bounds(bounds)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    _draw(axes[0], unmatched, color=MISS_COLOUR, lw=0.5, alpha=0.8)
    _draw_coloured(axes[0], matched, _corridor_colours(matched["corridor"]), lw=0.8)
    _finish(
        axes[0],
        bounds,
        f"corridors — {edges['corridor'].nunique()} total",
        "colour = corridor; grey = its own corridor. NB the palette has 20 colours and repeats — "
        "same colour is only meaningful for lines that touch.",
    )

    zoom_edges = edges.cx[zoom[0] : zoom[2], zoom[1] : zoom[3]]
    zm, zu = zoom_edges[zoom_edges["matched"]], zoom_edges[~zoom_edges["matched"]]
    _draw(axes[1], zu, color=MISS_COLOUR, lw=1.4, alpha=0.9)
    _draw_coloured(axes[1], zm, _corridor_colours(zm["corridor"]), lw=2.6)
    _finish(
        axes[1],
        zoom,
        "zoom",
        "a roadway and the sidewalks on both sides share one colour = one street space",
    )
    for ax in axes:
        ax.add_patch(
            plt.Rectangle(
                (zoom[0], zoom[1]),
                zoom[2] - zoom[0],
                zoom[3] - zoom[1],
                fill=False,
                edgecolor="#111111",
                lw=0.9,
                zorder=9,
            )
        )
    fig.suptitle("P7.2 validation 2/4 — what shares a corridor", fontsize=13, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def fig_attachment(edges, anchors, bounds, out: Path) -> None:
    """The strongest check: draw what attached to what.

    A correct attachment is a short hairline perpendicular to the street. A
    mis-tuned tolerance shows up immediately as long diagonals leaping across
    blocks — which no summary statistic makes visible.
    """
    import shapely

    matched = edges[edges["matched"] & ~edges["is_anchor"]]
    anchor_geom = dict(zip(anchors["corridor"], anchors.geometry, strict=True))
    links: list[np.ndarray] = []
    lengths: list[float] = []
    for corridor, geom in zip(matched["corridor"], matched.geometry, strict=True):
        target = anchor_geom.get(corridor)
        if target is None:
            continue
        mid = geom.interpolate(0.5, normalized=True)
        near = shapely.ops.nearest_points(mid, target)[1]
        links.append(np.array([[mid.x, mid.y], [near.x, near.y]]))
        lengths.append(float(mid.distance(target)))

    zoom = _zoom_bounds(bounds)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    for ax, lw in ((axes[0], 0.4), (axes[1], 1.6)):
        _draw(ax, edges[edges["mode"] == "walk"], color="#cfd8e3", lw=lw, alpha=0.9)
        _draw(ax, anchors, color=DRIVE_COLOUR, lw=lw * 1.6, alpha=0.9, zorder=3)
        if links:
            ax.add_collection(LineCollection(links, colors="#2f855a", linewidths=lw * 1.1, alpha=0.85, zorder=4))
        ax.add_collection(LineCollection([], colors="none"))
    arr = np.array(lengths) if lengths else np.array([0.0])
    _finish(
        axes[0],
        bounds,
        f"attachment — {len(links)} edges attached to an anchor",
        f"green = edge midpoint -> its corridor anchor; "
        f"median {np.median(arr):.1f} m, p95 {np.percentile(arr, 95):.1f} m",
    )
    _finish(axes[1], zoom, "zoom", "every green tick should be short and cross the street, never run along it")
    fig.suptitle("P7.2 validation 3/4 — what got attached to what", fontsize=13, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def fig_unmatched(edges, bounds, out: Path) -> None:
    walk = edges[edges["mode"] == "walk"].copy()
    walk["hw"] = walk["highway"].fillna("(none)")
    top = walk.groupby("hw")["matched"].agg(["size", "sum"]).sort_values("size", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), width_ratios=[1.35, 1])
    _draw(axes[0], walk[walk["matched"]], color="#a7c7e7", lw=0.5, alpha=0.9)
    _draw(axes[0], walk[~walk["matched"]], color="#d53f8c", lw=0.9, zorder=3)
    _finish(
        axes[0],
        bounds,
        "unmatched walk edges (pink)",
        "these must read as interior alleys, plazas and steps — not as missed sidewalks",
    )

    labels = list(top.index)[:9]
    rate = [top.loc[label, "sum"] / max(top.loc[label, "size"], 1) for label in labels]
    sizes = [int(top.loc[label, "size"]) for label in labels]
    y = np.arange(len(labels))
    axes[1].barh(y, rate, color=["#2b6cb0" if r >= 0.5 else "#d53f8c" for r in rate])
    axes[1].set_yticks(y, [f"{label}  (n={n})" for label, n in zip(labels, sizes, strict=True)], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("share matched to a roadway corridor", fontsize=9)
    axes[1].set_title("match rate by OSM highway class", fontsize=11, loc="left", pad=8)
    axes[1].grid(axis="x", color="#eeeeee")
    axes[1].set_axisbelow(True)
    for spine in ("top", "right"):
        axes[1].spines[spine].set_visible(False)
    fig.suptitle("P7.2 validation 4/4 — read the residual before calling it a miss", fontsize=13, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=Path("demo/out/cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("demo/out/street_space"))
    parser.add_argument("--crs", default="EPSG:32618", help="working CRS the cached graphs are projected to")
    parser.add_argument("--tolerance-m", type=float, default=None)
    parser.add_argument("--max-bearing-deg", type=float, default=None)
    args = parser.parse_args()

    overrides: dict = {}
    if args.tolerance_m is not None:
        overrides["tolerance_m"] = args.tolerance_m
    if args.max_bearing_deg is not None:
        overrides["max_bearing_deg"] = args.max_bearing_deg
    settings = Settings.model_validate({"network": {"street_space": overrides}}) if overrides else Settings()

    graphs = load_graphs(args.cache_dir, settings, args.crs)
    space = resolve_street_space(graphs, settings, working_crs=args.crs)
    print(json.dumps(space.stats, indent=2, default=str))

    import pandas as pd

    frames = [edges_as_gdf(graphs.get(mode), mode, args.crs) for mode in graphs.modes()]
    import geopandas as gpd

    edges = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=args.crs)
    info = [space.corridors.get((r.u, r.v, r.key)) for r in edges.itertuples()]
    edges["corridor"] = [i.corridor_id if i else "" for i in info]
    edges["matched"] = [bool(i and i.matched) for i in info]
    edges["is_anchor"] = [
        bool(i and i.corridor_id == f"e:{r.u}:{r.v}:{r.key}") for i, r in zip(info, edges.itertuples(), strict=True)
    ]
    anchors = edges[edges["is_anchor"]]

    walk = edges[edges["mode"] == "walk"]
    drive = edges[edges["mode"] == "drive"]
    bounds = tuple(edges.total_bounds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_three_graphs(walk, drive, bounds, args.out_dir / "01_three_graphs.png")
    fig_corridors(edges, bounds, args.out_dir / "02_corridors.png")
    fig_attachment(edges, anchors, bounds, args.out_dir / "03_attachment.png")
    fig_unmatched(edges, bounds, args.out_dir / "04_unmatched.png")


if __name__ == "__main__":
    main()
