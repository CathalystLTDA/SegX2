"""Figures for a finished run: route maps, segment maps, charts and tables.

The ``render_figures`` stage. It draws what the engine demo (``demo/run_demo.py``)
drew, from the same in-memory stage results, generalised so any run can use it:

* every **attribute** of the demographic schema, with its own class labels
  (the demo assumed one ``income`` attribute with ``low/mid/high``);
* every **partition** ``(binding, category)`` present — ``od/commute``,
  ``nearest/parks``, ``catchment/parks``, … (the demo drew ``od`` only);
* drawn in bulk: each edge is one stroke of a ``LineCollection`` and route maps
  draw at most ``figures.max_routes`` trajectories (a seeded sample, stated in the
  title), since a city-wide catchment holds millions of paths.

Layout under the run's output folder::

    maps/                            routes: overview, nearest, catchment per category, OD
    segments/<binding>-<category>/   per-segment metric maps
    charts/<binding>-<category>/     per-group / distribution charts
    tables/                          CSV + JSON audits of the metric frames

Figures are derived output. A figure that fails to draw is logged and listed in
:attr:`RenderedFigures.failed`; it never fails the run. Rendering is headless
(matplotlib's ``Figure`` API, Agg) so it is safe on the RPC worker thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from segx.io.demographics import _slug
from segx.io.writers import _edge_line, _find_edge
from segx.progress import report

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings
    from segx.progress import CancelCheck, ProgressCallback

log = logging.getLogger(__name__)

STAGE = "render_figures"
FOLDERS = ("maps", "segments", "charts", "tables")

#: The demo's income palette, used when an attribute's classes are exactly these.
CLASS_COLOUR = {"low": "#4575b4", "mid": "#f4a000", "high": "#d73027"}
_NA_COLOUR = "#999999"
MODE_COLOUR = {"walk": "#1adb6d", "drive": "#f4a000", "transit": "#00c2d1"}
CATEGORY_COLOUR = {"parks": "#2ca02c", "supermarkets": "#ff7f0e", "health_centers": "#d62728", "bus_stops": "#1f77b4"}
BINDING_COLOUR = {"nearest": "#1f77b4", "catchment": "#2ca02c", "od": "#d62728"}
_FALLBACK = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]
_ORIGIN_COLOUR = "#1adb6d"
_DEST_COLOUR = "#f5f5f5"
_MARKER_EDGE = "#111111"
_BACKDROP = "#6a6a6a"
_BASEMAP_TIMEOUT_S = 20

#: Esri's Dark Gray Canvas base layer (no labels, no API key). CartoDB DarkMatter
#: — the demo's original choice — now answers every tile with an "API KEY
#: REQUIRED" watermark, which would draw over the data as if it were a basemap.
BASEMAP_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
BASEMAP_ATTRIBUTION = "Basemap: Esri, HERE, Garmin, © OpenStreetMap contributors"


def basemap_provider():
    """The basemap tile source (an ``xyzservices.TileProvider``)."""
    import xyzservices

    return xyzservices.TileProvider(
        name="Esri.WorldDarkGrayBase", url=BASEMAP_URL, attribution=BASEMAP_ATTRIBUTION, max_zoom=16
    )


@dataclass
class RenderedFigures:
    """What :func:`render_figures` wrote: files per folder (relative to ``root``)."""

    root: Path
    files: dict[str, list[str]] = field(default_factory=dict)
    failed: list[dict[str, str]] = field(default_factory=list)
    basemap: str = "off"  # "on" | "off" | "unavailable"

    def artifacts(self) -> list[dict[str, Any]]:
        """JSON-native listing for the RPC result: one entry per folder."""
        return [
            {"kind": f"figures:{folder}", "path": str(self.root / folder), "files": names}
            for folder, names in self.files.items()
            if names
        ]


# --------------------------------------------------------------------------- #
# Colours                                                                      #
# --------------------------------------------------------------------------- #


def class_colours(labels: list[str]) -> dict[str, str]:
    """One colour per class: the demo's income palette when it fits, else a cycle."""
    if labels and set(labels) <= set(CLASS_COLOUR):
        out = {c: CLASS_COLOUR[c] for c in labels}
    else:
        palette = ["#4575b4", "#d73027", "#f4a000", "#1a9850", "#984ea3", "#00c2d1", "#a6761d", "#e7298a"]
        out = {c: palette[i % len(palette)] for i, c in enumerate(labels)}
    out["n/a"] = _NA_COLOUR
    return out


def _cycle(name: str, i: int, fixed: dict[str, str]) -> str:
    return fixed.get(name, _FALLBACK[i % len(_FALLBACK)])


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #


class _Geometry:
    """Edge and node coordinates looked up across the mode graphs, cached.

    Same lookup as the writers (:func:`segx.io.writers.resolve_geometries`): the
    row's own mode graph first, then the others — after ``expand_transit_rides`` a
    transit path mixes walk, drive and rail edges.
    """

    def __init__(self, graphs) -> None:
        self.graphs = graphs
        self.modes = list(graphs.modes())
        self._edges: dict[tuple, np.ndarray | None] = {}

    def _order(self, mode: str) -> list[str]:
        return [m for m in (mode, *self.modes) if m in self.graphs]

    def edge(self, mode: str, u, v, key) -> np.ndarray | None:
        token = (mode, u, v, key)
        if token not in self._edges:
            coords = None
            for candidate in dict.fromkeys(self._order(mode)):
                graph = self.graphs.get(candidate)
                found = _find_edge(graph, u, v, key)
                if found is not None:
                    coords = np.asarray(_edge_line(graph, *found).coords, dtype=float)[:, :2]
                    break
            self._edges[token] = coords
        return self._edges[token]

    def node(self, mode: str, n) -> tuple[float, float] | None:
        for candidate in dict.fromkeys(self._order(mode)):
            graph = self.graphs.get(candidate)
            if graph.has_node(n):
                data = graph.nodes[n]
                return float(data["x"]), float(data["y"])
        return None

    def path(self, mode: str, path) -> list[np.ndarray]:
        """A trajectory as edge polylines (one stroke each; direction does not matter)."""
        out = []
        for u, v, key in path:
            coords = self.edge(mode, u, v, key)
            if coords is not None:
                out.append(coords)
        return out

    def ends(self, mode: str, path) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        return self.node(mode, path[0][0]), self.node(mode, path[-1][1])


def _backdrop_graph(graphs):
    """A plain street graph for the faint backdrop (never the merged transit one)."""
    for mode in ("walk", "drive"):
        if mode in graphs:
            return graphs.get(mode)
    modes = sorted(graphs.modes())
    return graphs.get(modes[0]) if modes else None


def _backdrop_segments(graph) -> np.ndarray:
    if graph is None or graph.number_of_edges() == 0:
        return np.empty((0, 2, 2))
    xs = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}
    return np.array([[xs[u], xs[v]] for u, v in graph.edges()], dtype=float)


# --------------------------------------------------------------------------- #
# Drawing helpers                                                              #
# --------------------------------------------------------------------------- #


class _Canvas:
    """Shared state for one run's figures: style, extent, backdrop, basemap."""

    def __init__(self, state, settings: Settings, cache_dir: Path | None) -> None:
        self.settings = settings
        self.cfg = settings.figures
        self.crs = state.boundary.crs
        minx, miny, maxx, maxy = state.boundary.geometry.bounds
        pad = 0.03 * max(maxx - minx, maxy - miny)
        self.extent = (minx - pad, maxx + pad, miny - pad, maxy + pad)
        self.boundary = state.boundary.geometry
        self.backdrop = _backdrop_segments(_backdrop_graph(state.graphs))
        self.basemap_state = "on" if self.cfg.basemap else "off"
        self._tiles_dir = (cache_dir or settings.cache_dir) / "tiles"
        self._cx = None

    @contextlib.contextmanager
    def figure(self, out_path: Path, *, size=(11, 11), map_axes: bool = True) -> Iterator[tuple[Any, Any]]:
        """A dark-styled figure with one axes; saved to *out_path* on exit."""
        import matplotlib.style
        from matplotlib.figure import Figure

        with matplotlib.style.context("dark_background"):
            fig = Figure(figsize=size)
            ax = fig.subplots()
            if map_axes:
                self._map_frame(ax)
            yield fig, ax
            if map_axes:
                ax.set_xlim(self.extent[0], self.extent[1])
                ax.set_ylim(self.extent[2], self.extent[3])
                self._basemap(ax)
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=self.cfg.dpi)

    def _map_frame(self, ax) -> None:
        from matplotlib.collections import LineCollection

        ax.set_aspect("equal")
        ax.set_xlabel("metres (working CRS)")
        ax.set_ylabel("metres (working CRS)")
        ax.ticklabel_format(style="plain", useOffset=False)  # UTM northings, not "6.68 × 1e6"
        if len(self.backdrop):
            ax.add_collection(LineCollection(self.backdrop, colors=_BACKDROP, linewidths=0.35, alpha=0.4, zorder=1))
        geoms = getattr(self.boundary, "geoms", [self.boundary])
        for poly in geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="#bbbbbb", linewidth=0.6, alpha=0.6, zorder=1)

    def _basemap(self, ax) -> None:
        """Dark web tiles behind the data; switched off for the run after a failure."""
        if self.basemap_state != "on":
            return
        try:
            if self._cx is None:
                import contextily as cx

                self._tiles_dir.mkdir(parents=True, exist_ok=True)
                cx.set_cache_dir(str(self._tiles_dir))
                self._cx = cx
            previous = socket.getdefaulttimeout()
            socket.setdefaulttimeout(_BASEMAP_TIMEOUT_S)  # contextily's requests carry no timeout
            try:
                # attribution=False: contextily's own attribution calls pyplot.draw(),
                # which opens a GUI-backed pyplot figure per map off the main thread
                self._cx.add_basemap(ax, crs=str(self.crs), source=basemap_provider(), attribution=False, zorder=0)
            finally:
                socket.setdefaulttimeout(previous)
            ax.text(
                0.005,
                0.005,
                BASEMAP_ATTRIBUTION,
                transform=ax.transAxes,
                fontsize=5,
                color="#aaaaaa",
                ha="left",
                va="bottom",
            )
        except Exception as exc:  # noqa: BLE001 — offline / tile server down: draw without it
            self.basemap_state = "unavailable"
            log.warning("figures: basemap unavailable (%s) — maps of this run render without it", exc)


def _lines(segments: list[np.ndarray], colours, *, width: float, zorder: int = 2, alpha: float | None = None):
    from matplotlib.collections import LineCollection

    return LineCollection(segments, colors=colours, linewidths=width, alpha=alpha, zorder=zorder)


def _legend(ax, handles, **kw) -> None:
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85, **kw)


def _line_handle(colour: str, label: str):
    from matplotlib.lines import Line2D

    return Line2D([0], [0], color=colour, lw=2.5, label=label)


def _marker_handle(marker: str, colour: str, label: str, size: float = 9):
    from matplotlib.lines import Line2D

    return Line2D(
        [0],
        [0],
        marker=marker,
        color="none",
        markerfacecolor=colour,
        markeredgecolor=_MARKER_EDGE,
        markersize=size,
        label=label,
    )


def _sample(items: list, cap: int, seed: int) -> tuple[list, bool]:
    if len(items) <= cap:
        return items, False
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=cap, replace=False))
    return [items[i] for i in idx], True


def _dominant(vec, attr: str, classes: list[str]) -> str:
    inner = vec.get(attr) if isinstance(vec, dict) else None
    best, best_v = "n/a", None
    for c in classes:
        v = (inner or {}).get(c)
        if v is None or v != v:  # missing or NaN
            continue
        if best_v is None or v > best_v:
            best, best_v = c, v
    return best if best_v is not None and best_v > 0 else "n/a"


# --------------------------------------------------------------------------- #
# Route maps (maps/)                                                           #
# --------------------------------------------------------------------------- #


def _plot_overview(canvas: _Canvas, geo: _Geometry, state, population, routing, out: Path, *, per_binding: int = 30):
    attr = canvas.settings.demographics.attributes[0]
    colours = class_colours(attr.class_labels)
    with canvas.figure(out) as (_fig, ax):
        drawn: dict[str, int] = {}
        segs, cols = [], []
        for traj in routing.trajectories:
            if not traj.reachable or not traj.path or drawn.get(traj.binding, 0) >= per_binding:
                continue
            parts = geo.path(str(traj.mode), traj.path)
            segs.extend(parts)
            cols.extend([BINDING_COLOUR.get(traj.binding, _FALLBACK[0])] * len(parts))
            drawn[traj.binding] = drawn.get(traj.binding, 0) + 1
        if segs:
            ax.add_collection(_lines(segs, cols, width=1.0, alpha=0.7))
        pois = state.pois.gdf
        if len(pois):
            ax.scatter(
                pois.geometry.x,
                pois.geometry.y,
                marker="*",
                s=30,
                color="#ff7f0e",
                edgecolor=_MARKER_EDGE,
                linewidth=0.2,
                zorder=4,
            )
        origins = population.origins
        dom = [_dominant(v, attr.name, attr.class_labels) for v in origins["demo_vector"]]
        ax.scatter(
            origins.geometry.x,
            origins.geometry.y,
            c=[colours[d] for d in dom],
            s=14,
            edgecolor=_MARKER_EDGE,
            linewidth=0.2,
            zorder=3,
        )
        handles = [
            _line_handle(BINDING_COLOUR.get(b, _FALLBACK[0]), f"{b} route (sample of {n})") for b, n in drawn.items()
        ]
        handles += [_marker_handle("o", colours[c], f"origin: {c} ({attr.name})", 7) for c in attr.class_labels]
        if len(pois):
            handles.append(_marker_handle("*", "#ff7f0e", f"POI ({len(pois)})", 11))
        _legend(ax, handles)
        ax.set_title(f"SegX — origins by dominant {attr.name}, POIs and sample routes\n{len(origins)} origins")


def _plot_nearest(canvas: _Canvas, geo: _Geometry, state, routing, out: Path):
    cats = canvas.settings.poi.categories_for("nearest")
    trajs = [t for t in routing.trajectories if t.binding == "nearest" and t.reachable and t.path]
    if not trajs:
        return False
    trajs, sampled = _sample(trajs, canvas.cfg.max_routes, canvas.settings.random_seed)
    with canvas.figure(out) as (_fig, ax):
        counts: dict[str, int] = {}
        segs, cols = [], []
        for t in trajs:
            colour = _cycle(t.category, cats.index(t.category) if t.category in cats else 0, CATEGORY_COLOUR)
            parts = geo.path(str(t.mode), t.path)
            segs.extend(parts)
            cols.extend([colour] * len(parts))
            counts[t.category] = counts.get(t.category, 0) + 1
        ax.add_collection(_lines(segs, cols, width=0.8, alpha=0.55))
        pois = state.pois.gdf
        for i, cat in enumerate(cats):
            sub = pois[pois["category"] == cat]
            if len(sub):
                ax.scatter(
                    sub.geometry.x,
                    sub.geometry.y,
                    marker="*",
                    s=40,
                    color=_cycle(cat, i, CATEGORY_COLOUR),
                    edgecolor=_MARKER_EDGE,
                    linewidth=0.3,
                    zorder=4,
                )
        _legend(
            ax,
            [
                _line_handle(_cycle(c, i, CATEGORY_COLOUR), f"{c} ({counts.get(c, 0)} routes)")
                for i, c in enumerate(cats)
            ],
            title="nearest route + POIs (★)",
        )
        note = f"random sample of {len(trajs):,}" if sampled else f"all {len(trajs):,}"
        ax.set_title(f"SegX — nearest-POI routes by category ({note})\neach origin's closest POI of each category")
    return True


def _plot_catchment(canvas: _Canvas, geo: _Geometry, state, routing, category: str, i: int, out: Path):
    trajs = [
        t
        for t in routing.trajectories
        if t.binding == "catchment" and t.category == category and t.reachable and t.path
    ]
    if not trajs:
        return False
    trajs, sampled = _sample(trajs, canvas.cfg.max_routes, canvas.settings.random_seed)
    colour = _cycle(category, i, CATEGORY_COLOUR)
    catchment = canvas.settings.poi.catchment
    # a catchment is bounded by time or by distance, whichever the config sets
    budget = (
        f"{catchment.budget_minutes:g}-min"
        if catchment.budget_minutes is not None
        else f"{catchment.distance_budget_m:g} m"
        if getattr(catchment, "distance_budget_m", None) is not None
        else "budget"
    )
    pois = state.pois.gdf
    sub = pois[pois["category"] == category]
    with canvas.figure(out) as (_fig, ax):
        segs, origins = [], []
        for t in trajs:
            segs.extend(geo.path(str(t.mode), t.path))
            start, _ = geo.ends(str(t.mode), t.path)
            if start is not None:
                origins.append(start)
        ax.add_collection(_lines(segs, colour, width=0.6, alpha=0.18))
        if origins:
            ox, oy = zip(*set(origins), strict=False)
            ax.scatter(
                ox, oy, marker="o", s=12, facecolor=_ORIGIN_COLOUR, edgecolor=_MARKER_EDGE, linewidth=0.2, zorder=3
            )
        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            marker="*",
            s=55,
            color=colour,
            edgecolor=_MARKER_EDGE,
            linewidth=0.3,
            zorder=4,
        )
        _legend(
            ax,
            [
                _line_handle(colour, f"reach route ({len(trajs):,}{' sampled' if sampled else ''})"),
                _marker_handle("*", colour, f"{category} POIs ({len(sub)})", 11),
                _marker_handle("o", _ORIGIN_COLOUR, "origin", 7),
            ],
        )
        ax.set_title(f"SegX — {budget} catchment: {category}\n(origin, POI) pairs reachable within the budget")
    return True


def _od_flows(geo: _Geometry, population, routing) -> list[dict]:
    vectors = dict(zip(population.origins["origin_id"].astype(str), population.origins["demo_vector"], strict=False))
    flows = []
    for t in routing.trajectories:
        if t.binding != "od" or not t.reachable or not t.path:
            continue
        parts = geo.path(str(t.mode), t.path)
        if not parts:
            continue
        flows.append(
            {
                "parts": parts,
                "ends": geo.ends(str(t.mode), t.path),
                "weight": float(t.weight) * float(getattr(t, "route_weight", 1.0)),
                "mode": str(t.mode),
                "vector": vectors.get(str(t.origin_id)),
            }
        )
    return flows


def _class_share(vec, attr: str, label: str) -> float:
    inner = vec.get(attr) if isinstance(vec, dict) else None
    if not inner:
        return 0.0
    total = sum(v for v in inner.values() if v == v and v is not None)
    v = inner.get(label)
    return float(v) / total if total and v is not None and v == v else 0.0


def _plot_od(canvas: _Canvas, flows: list[dict], weights: list[float], out: Path, *, label: str, colour: str | None):
    """One OD map: routes (opacity ∝ weight), home/work markers; colour by mode when *colour* is None."""
    from matplotlib.colors import to_rgba

    vmax = max(weights, default=0.0)
    with canvas.figure(out) as (_fig, ax):
        segs, cols, homes, works, by_mode = [], [], set(), set(), {}
        total = 0.0
        for j in sorted(range(len(flows)), key=lambda k: weights[k]):  # heaviest drawn last
            w = weights[j]
            if w <= 0:
                continue
            f = flows[j]
            alpha = min(0.06 + 0.7 * (w / vmax if vmax else 0.0), 0.8)
            rgba = to_rgba(colour or MODE_COLOUR.get(f["mode"], _FALLBACK[0]), alpha)
            segs.extend(f["parts"])
            cols.extend([rgba] * len(f["parts"]))
            if f["ends"][0]:
                homes.add(f["ends"][0])
            if f["ends"][1]:
                works.add(f["ends"][1])
            by_mode[f["mode"]] = by_mode.get(f["mode"], 0) + 1
            total += w
        if segs:
            ax.add_collection(_lines(segs, cols, width=1.0))
        if homes:
            hx, hy = zip(*homes, strict=False)
            ax.scatter(
                hx, hy, marker="o", s=22, facecolor=_ORIGIN_COLOUR, edgecolor=_MARKER_EDGE, linewidth=0.3, zorder=4
            )
        if works:
            wx, wy = zip(*works, strict=False)
            ax.scatter(
                wx, wy, marker="^", s=30, facecolor=_DEST_COLOUR, edgecolor=_MARKER_EDGE, linewidth=0.3, zorder=5
            )
        routes = (
            [_line_handle(MODE_COLOUR.get(m, _FALLBACK[0]), f"{m} ({n} routes)") for m, n in sorted(by_mode.items())]
            if colour is None
            else [_line_handle(colour, "route")]
        )
        _legend(
            ax,
            [
                *routes,
                _marker_handle("o", _ORIGIN_COLOUR, "origin", 7),
                _marker_handle("^", _DEST_COLOUR, "destination", 8),
            ],
        )
        ax.set_title(
            f"SegX — OD routes, {label}\n{total:,.0f} trips across {sum(by_mode.values()):,} flows (opacity ∝ trips)"
        )


# --------------------------------------------------------------------------- #
# Segment maps (segments/<partition>/)                                         #
# --------------------------------------------------------------------------- #


def _partition(frame, binding: str, category: str, attribute: str | None = None):
    if frame is None or frame.empty:
        return frame
    mask = frame["binding"] == binding
    if "category" in frame.columns:
        mask &= frame["category"] == category
    if attribute is not None and "attribute" in frame.columns:
        mask &= frame["attribute"] == attribute
    return frame[mask]


def _segment_lines(geo: _Geometry, df) -> tuple[list[np.ndarray], list[int]]:
    """Polylines of a metric frame's rows, and the row positions that had one."""
    segs, rows = [], []
    for i, (mode, u, v, key) in enumerate(df[["mode", "u", "v", "key"]].itertuples(index=False, name=None)):
        coords = geo.edge(str(mode), u, v, key)
        if coords is not None:
            segs.append(coords)
            rows.append(i)
    return segs, rows


def _plot_scalar(canvas, geo, df, col: str, out: Path, *, cmap: str, label: str, title: str, vmax: float | None = None):
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    df = df[df[col].notna()]
    if df.empty:
        return False
    vals = df[col].to_numpy(dtype=float)
    top = vmax if vmax is not None else (float(np.nanpercentile(vals, 97)) or 1.0)
    norm = Normalize(vmin=0.0, vmax=top)
    cm = matplotlib.colormaps[cmap]
    segs, rows = _segment_lines(geo, df)
    with canvas.figure(out) as (fig, ax):
        ax.add_collection(_lines(segs, cm(norm(np.minimum(vals[rows], top))), width=1.3))
        fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.6, pad=0.02).set_label(label)
        ax.set_title(title)
    return True


def _plot_dominant(canvas, geo, edge_part, attr, out: Path, where: str) -> dict[str, int]:
    """Segments coloured by the class with the most mass (Netto 2018 fig. 3); returns counts."""
    classes = attr.class_labels
    colours = class_colours(classes)
    wide = edge_part.pivot_table(
        index=["mode", "u", "v", "key"], columns="class", values="mass", aggfunc="sum", fill_value=0.0
    )
    if wide.empty:
        return {}
    masses = wide.reindex(columns=classes, fill_value=0.0).to_numpy(dtype=float)
    keep = masses.sum(axis=1) > 0
    wide, masses = wide[keep], masses[keep]
    dom = [classes[i] for i in masses.argmax(axis=1)]
    frame = wide.reset_index()[["mode", "u", "v", "key"]]
    segs, rows = _segment_lines(geo, frame)
    counts = {c: 0 for c in classes}
    for r in rows:
        counts[dom[r]] += 1
    with canvas.figure(out) as (_fig, ax):
        ax.add_collection(_lines(segs, [colours[dom[r]] for r in rows], width=1.3))
        _legend(ax, [_line_handle(colours[c], f"{c} ({counts[c]:,})") for c in classes], title=f"dominant {attr.name}")
        ax.set_title(
            f"SegX — dominant {attr.name} group per street segment ({where})\n"
            "the group with the most people colours the segment"
        )
    return counts


def _plot_dominant_trajectories(canvas, geo, df, attr, out: Path, where: str) -> bool:
    from matplotlib.colors import to_rgba

    df = df[df["dominant_class"].notna() & df["dominance"].notna()]
    if df.empty:
        return False
    colours = class_colours(attr.class_labels)
    segs, rows = _segment_lines(geo, df)
    dom = df["dominant_class"].astype(str).to_numpy()
    w = df["dominance"].to_numpy(dtype=float)
    with canvas.figure(out) as (_fig, ax):
        ax.add_collection(
            _lines(segs, [to_rgba(colours.get(dom[r], _NA_COLOUR), max(0.12, min(1.0, w[r]))) for r in rows], width=1.4)
        )
        _legend(ax, [_line_handle(colours[c], c) for c in attr.class_labels], title=f"dominant {attr.name}")
        ax.set_title(
            f"SegX — dominant trajectories ({where}, {attr.name})\n"
            "colour = dominant group, opacity = dominance (1 − normalised entropy)"
        )
    return True


def _plot_segregated(canvas, geo, df, attr, out: Path, where: str) -> bool:
    if df.empty:
        return False
    colours = class_colours(attr.class_labels)
    segs, rows = _segment_lines(geo, df)
    exclusive = df["exclusive"].fillna(False).astype(bool).to_numpy()
    dom = df["dominant_class"].astype(str).to_numpy()
    faint = [segs[k] for k, r in enumerate(rows) if not exclusive[r]]
    hot = [(segs[k], dom[r]) for k, r in enumerate(rows) if exclusive[r]]
    counts = {c: sum(1 for _, d in hot if d == c) for c in attr.class_labels}
    with canvas.figure(out) as (_fig, ax):
        ax.add_collection(_lines(faint, "#888888", width=0.7, alpha=0.45))
        if hot:
            ax.add_collection(
                _lines([s for s, _ in hot], [colours.get(d, _NA_COLOUR) for _, d in hot], width=2.2, zorder=3)
            )
        _legend(
            ax,
            [_line_handle(colours[c], f"{c} ({counts[c]:,})") for c in attr.class_labels],
            title=f"exclusive to {attr.name} group",
        )
        thr = canvas.settings.metrics.exclusive_lq_threshold
        ax.set_title(
            f"SegX — most exclusive segments ({where}, {attr.name})\n"
            f"{len(hot):,} segments with dominant-group LQ ≥ {thr:g}"
        )
    return True


# --------------------------------------------------------------------------- #
# Charts (charts/<partition>/)                                                 #
# --------------------------------------------------------------------------- #


def _bars(ax, classes, heights, colours, fmt="{:.2f}") -> None:
    ax.bar(classes, heights, color=[colours[c] for c in classes], edgecolor="black")
    for i, h in enumerate(heights):
        if h == h:  # not NaN
            ax.text(i, h, fmt.format(h), ha="center", va="bottom", fontsize=9)


def _chart_counts(canvas, counts: dict[str, int], attr, out: Path, where: str) -> bool:
    if not counts or not sum(counts.values()):
        return False
    classes = attr.class_labels
    total = sum(counts.values())
    with canvas.figure(out, size=(7, 5), map_axes=False) as (_fig, ax):
        ax.bar(
            classes,
            [counts.get(c, 0) for c in classes],
            color=[class_colours(classes)[c] for c in classes],
            edgecolor="black",
        )
        for i, c in enumerate(classes):
            n = counts.get(c, 0)
            ax.text(i, n, f"{n:,}\n{100 * n / total:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("street segments dominated")
        ax.set_title(f"SegX — segments by dominant {attr.name} group ({where})")
    return True


def _chart_histogram(canvas, df, out: Path, where: str) -> bool:
    vals = df["diversity"].dropna() if df is not None and not df.empty else None
    if vals is None or vals.empty:
        return False
    with canvas.figure(out, size=(9, 5), map_axes=False) as (_fig, ax):
        ax.hist(vals, bins=40, color="#d73027", edgecolor="white")
        ax.axvline(vals.mean(), color="white", linestyle="--", linewidth=1, label=f"mean = {vals.mean():.1f}")
        ax.set_xlabel("segment diversity  D = −Σ Pᵢ log₂(Pᵢ/Pₜ)")
        ax.set_ylabel("street segments")
        ax.set_title(f"SegX — distribution of per-segment diversity ({where})")
        ax.legend()
    return True


def _chart_per_mode(canvas, frames: dict[str, Any], attr, out: Path, *, ylabel: str, title: str, fmt="{:.2f}") -> bool:
    """One panel per travel mode (a partition key): bars per class for each series."""
    first = next(iter(frames.values()))
    if first is None or first.empty:
        return False
    classes = attr.class_labels
    modes = list(dict.fromkeys(first["mode"]))
    colours = class_colours(classes)
    with canvas.figure(out, size=(4 * len(modes) + 2, 5), map_axes=False) as (fig, _ax):
        fig.clear()
        axes = fig.subplots(1, len(modes), squeeze=False, sharey=True)[0]
        x = np.arange(len(classes))
        series = list(frames.items())
        width = 0.8 / len(series)
        for ax, mode in zip(axes, modes, strict=True):
            for k, (col, df) in enumerate(series):
                sub = df[df["mode"] == mode]
                values = dict(zip(sub["class"].astype(str), sub[col].astype(float), strict=False))
                heights = [values.get(c, float("nan")) for c in classes]
                if len(series) == 1:
                    _bars(ax, classes, heights, colours, fmt)
                else:
                    ax.bar(x + (k - (len(series) - 1) / 2) * width, heights, width, label=col, color=_FALLBACK[k])
                    ax.set_xticks(x, classes)
            ax.set_title(f"mode: {mode}", fontsize=10)
        axes[0].set_ylabel(ylabel)
        if len(series) > 1:
            axes[0].legend(fontsize=8)
        fig.suptitle(title, fontsize=11)
    return True


# --------------------------------------------------------------------------- #
# Tables (tables/)                                                             #
# --------------------------------------------------------------------------- #

_P5_METRICS = (
    "experienced_segregation",
    "intergroup_contact",
    "dominant_trajectories",
    "segregated_trajectories",
    "exposure",
    "cross_mode_exposure",
)


def _write_tables(aggregation, tables: Path) -> list[str]:
    tables.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def csv(df, name: str) -> None:
        df.to_csv(tables / name, index=False)
        written.append(name)

    dv = aggregation.metrics.get("diversity")
    if dv is not None and not dv.empty:
        csv(dv, "diversity_by_segment.csv")
        summary = (
            dv.groupby(["mode", "category", "binding", "attribute"])
            .agg(n_segments=("diversity", "size"), mean_diversity=("diversity", "mean"))
            .reset_index()
        )
        csv(summary, "partition_summary.csv")
    for name in _P5_METRICS:
        if name in aggregation.metrics and not aggregation.metrics[name].empty:
            csv(aggregation.metrics[name], f"{name}_by_segment.csv")
        if name in aggregation.summaries and not aggregation.summaries[name].empty:
            csv(aggregation.summaries[name], f"{name}_summary.csv")
    audit = aggregation.stats.get("cross_mode")
    if audit:
        (tables / "cross_mode_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
        written.append("cross_mode_audit.json")
    from segx.metrics.cross_mode_exposure import cross_mode_combination_counts

    combos = cross_mode_combination_counts(aggregation.edge_table)
    if not combos.empty:
        csv(combos, "cross_mode_segments_by_combination.csv")
    frag = aggregation.metrics.get("path_fragmentation")
    if frag is not None and not frag.empty:
        csv(frag, "path_fragmentation_by_group.csv")
        frag_audit = aggregation.stats.get("fragmentation")
        if frag_audit:
            (tables / "path_fragmentation_audit.json").write_text(
                json.dumps(frag_audit, indent=2, default=str), encoding="utf-8"
            )
            written.append("path_fragmentation_audit.json")
    return written


# --------------------------------------------------------------------------- #
# The stage                                                                    #
# --------------------------------------------------------------------------- #


def _partitions(edge_table) -> list[tuple[str, str]]:
    if edge_table is None or edge_table.empty:
        return []
    pairs = edge_table[["binding", "category"]].drop_duplicates().itertuples(index=False, name=None)
    order = {"od": 0, "nearest": 1, "catchment": 2}
    return sorted(((str(b), str(c)) for b, c in pairs), key=lambda p: (order.get(p[0], 9), p[1]))


def render_figures(
    run,
    settings: Settings,
    output_dir: str | Path,
    *,
    cache_dir: Path | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> RenderedFigures:
    """Draw every figure of a finished run under *output_dir* (see the module docstring).

    *run* is anything with ``state``, ``population``, ``routing`` and
    ``aggregation`` (a :class:`~segx.orchestrator.RunResult`).
    """
    state, population, routing, aggregation = run.state, run.population, run.routing, run.aggregation
    root = Path(output_dir)
    result = RenderedFigures(root=root, files={f: [] for f in FOLDERS})
    canvas = _Canvas(state, settings, cache_dir)
    geo = _Geometry(state.graphs)
    metrics = aggregation.metrics
    attrs = [a for a in settings.demographics.attributes if a.name in set(aggregation.edge_table.get("attribute", []))]
    tasks: list[tuple[str, str, Callable[[Path], Any]]] = []  # (folder, relative file, draw(path))

    def add(folder: str, rel: str, fn: Callable[[Path], Any]) -> None:
        tasks.append((folder, rel, fn))

    # ---- maps/: routes ----
    if routing.trajectories:
        add("maps", "overview.png", lambda p: _plot_overview(canvas, geo, state, population, routing, p))
    if any(t.binding == "nearest" for t in routing.trajectories):
        add("maps", "nearest.png", lambda p: _plot_nearest(canvas, geo, state, routing, p))
    for i, cat in enumerate(settings.poi.categories_for("catchment")):
        if any(t.binding == "catchment" and t.category == cat for t in routing.trajectories):
            add(
                "maps",
                f"catchment_{_slug(cat)}.png",
                lambda p, c=cat, k=i: _plot_catchment(canvas, geo, state, routing, c, k, p),
            )
    flows_box: dict[str, list] = {}

    def flows() -> list[dict]:
        if "f" not in flows_box:
            flows_box["f"] = _od_flows(geo, population, routing)
        return flows_box["f"]

    if any(t.binding == "od" for t in routing.trajectories):
        add(
            "maps",
            "od_all.png",
            lambda p: _plot_od(canvas, flows(), [f["weight"] for f in flows()], p, label="all trips", colour="#a27bd6"),
        )
        for attr in attrs:
            colours = class_colours(attr.class_labels)
            for label in attr.class_labels:
                add(
                    "maps",
                    f"od_{_slug(attr.name)}_{_slug(label)}.png",
                    lambda p, a=attr, c=label, col=colours[label]: _plot_od(
                        canvas,
                        flows(),
                        [f["weight"] * _class_share(f["vector"], a.name, c) for f in flows()],
                        p,
                        label=f"{a.name} = {c} (trips × origin share)",
                        colour=col,
                    ),
                )
        if len({str(t.mode) for t in routing.trajectories if t.binding == "od"}) > 1:
            add(
                "maps",
                "od_by_mode.png",
                lambda p: _plot_od(
                    canvas, flows(), [f["weight"] for f in flows()], p, label="by travel mode", colour=None
                ),
            )

    # ---- segments/ + charts/: one folder per partition ----
    et = aggregation.edge_table
    for binding, category in _partitions(et):
        part = f"{binding}-{_slug(category)}"
        where = f"{binding}, {category}"
        for attr in attrs:
            a = _slug(attr.name)
            edge_part = _partition(et, binding, category, attr.name)
            counts_box: dict[str, dict] = {}

            def dominant(p, ep=edge_part, at=attr, w=where, box=counts_box):
                box["counts"] = _plot_dominant(canvas, geo, ep, at, p, w)
                return bool(box["counts"])

            add("segments", f"{part}/dominant_{a}.png", dominant)
            add(
                "charts",
                f"{part}/dominant_class_counts_{a}.png",
                lambda p, at=attr, w=where, box=counts_box: _chart_counts(canvas, box.get("counts", {}), at, p, w),
            )

            def scalar(metric, col, cmap, label, title, vmax=None, at=attr, b=binding, c=category):
                def draw(p):
                    df = _partition(metrics.get(metric), b, c, at.name)
                    return (
                        df is not None
                        and not df.empty
                        and _plot_scalar(canvas, geo, df, col, p, cmap=cmap, label=label, title=title, vmax=vmax)
                    )

                return draw

            if "diversity" in metrics:
                add(
                    "segments",
                    f"{part}/diversity_{a}.png",
                    scalar(
                        "diversity",
                        "diversity",
                        "YlOrRd",
                        "segment diversity  D = −Σ Pᵢ log₂(Pᵢ/Pₜ)  (clipped at p97)",
                        f"SegX — social diversity per street segment ({where}, {attr.name})\nred = most mixed streets",
                    ),
                )
                add(
                    "charts",
                    f"{part}/diversity_distribution_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _chart_histogram(
                        canvas, _partition(metrics["diversity"], b, c, at.name), p, w
                    ),
                )
            if "experienced_segregation" in metrics:
                add(
                    "segments",
                    f"{part}/experienced_segregation_{a}.png",
                    scalar(
                        "experienced_segregation",
                        "experienced_segregation",
                        "magma",
                        "experienced segregation  1 − h  (1 = homogeneous)",
                        f"SegX — experienced segregation per segment ({where}, {attr.name})",
                        vmax=1.0,
                    ),
                )
            if "intergroup_contact" in metrics:
                add(
                    "segments",
                    f"{part}/intergroup_contact_{a}.png",
                    scalar(
                        "intergroup_contact",
                        "intergroup_contact",
                        "viridis",
                        "intergroup contact  1 − Σ pᵢ²  (Gini–Simpson)",
                        f"SegX — intergroup contact probability per segment ({where}, {attr.name})",
                        vmax=1.0,
                    ),
                )
            if "dominant_trajectories" in metrics:
                add(
                    "segments",
                    f"{part}/dominant_trajectories_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _plot_dominant_trajectories(
                        canvas, geo, _partition(metrics["dominant_trajectories"], b, c, at.name), at, p, w
                    ),
                )
            if "segregated_trajectories" in metrics:
                add(
                    "segments",
                    f"{part}/segregated_segments_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _plot_segregated(
                        canvas, geo, _partition(metrics["segregated_trajectories"], b, c, at.name), at, p, w
                    ),
                )
            if "exposure" in metrics:
                add(
                    "charts",
                    f"{part}/exposure_by_group_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _chart_per_mode(
                        canvas,
                        {"exposure": _partition(metrics["exposure"], b, c, at.name)},
                        at,
                        p,
                        ylabel="mean segment diversity encountered (bits)",
                        title=f"SegX — social exposure per {at.name} group ({w})",
                    ),
                )
            if "cross_mode_exposure" in metrics and "exposure" in metrics:
                add(
                    "charts",
                    f"{part}/cross_mode_exposure_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _chart_per_mode(
                        canvas,
                        {
                            "exposure": _partition(metrics["exposure"], b, c, at.name),
                            "cross_mode_exposure": _partition(metrics["cross_mode_exposure"], b, c, at.name),
                        },
                        at,
                        p,
                        ylabel="mean segment diversity encountered (bits)",
                        title=f"SegX — exposure with and without cross-mode co-presence ({w}, {at.name})\n"
                        f"other modes counted at factor={settings.metrics.cross_mode.factor:g}",
                    ),
                )
            if "path_fragmentation" in metrics:
                add(
                    "charts",
                    f"{part}/path_fragmentation_{a}.png",
                    lambda p, at=attr, b=binding, c=category, w=where: _chart_per_mode(
                        canvas,
                        {"fragmentation": _partition(metrics["path_fragmentation"], b, c, at.name)},
                        at,
                        p,
                        ylabel="fractal dimension D",
                        fmt="{:.3f}",
                        title=f"SegX — path fragmentation per {at.name} group ({w})",
                    ),
                )

    total = len(tasks) + 1
    for index, (folder, rel, draw) in enumerate(tasks):
        report(progress, cancel, stage=STAGE, index=index, total=total, message=f"{folder}/{rel}")
        path = root / folder / rel
        try:
            drawn = draw(path)
        except Exception as exc:  # noqa: BLE001 — derived output: log, list, carry on
            log.warning("figures: %s/%s failed: %s: %s", folder, rel, type(exc).__name__, exc)
            result.failed.append({"file": f"{folder}/{rel}", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if drawn is not False and path.exists():
            result.files[folder].append(rel)

    report(progress, cancel, stage=STAGE, index=total - 1, total=total, message="tables/")
    try:
        result.files["tables"] = _write_tables(aggregation, root / "tables")
    except Exception as exc:  # noqa: BLE001
        log.warning("figures: tables failed: %s: %s", type(exc).__name__, exc)
        result.failed.append({"file": "tables/", "error": f"{type(exc).__name__}: {exc}"})
    result.basemap = canvas.basemap_state
    return result


__all__ = ["FOLDERS", "STAGE", "RenderedFigures", "class_colours", "render_figures"]
