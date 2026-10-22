"""Path fragmentation index — the **linear fractal dimension of an individual
trajectory** (Netto et al. 2015, via Mandelbrot 1983). P7.3.

The third term of the composed mobility measure in Netto et al. (2015): mobility
in personal urban trajectories is read as *(1) extent; (2) number of activities;
(3) fragmentation* — "the number of street segments used by actors, a property of
movement along the street network, also dependent on modes of transport (a high
number of segments indicates the use of many spaces, which in turn tends to
increase knowledge of the urban structure)".

Why this metric is shaped unlike every other one here
-----------------------------------------------------
Every other metric in this package is a function of the **edge table** — the
per-segment aggregate. Fragmentation is not: it is a property of a *path*, and
the edge table has thrown path identity away by construction (invariant #6, no
per-trajectory storage in the hot path). So the measurement happens while the
trajectories stream past, in
:class:`~segx.aggregation.path_accumulator.PathAccumulator`, which keeps only
bounded per-``(partition, attribute, class)`` running sums; this module owns the
*formula* (invariant #5) and the metric plugin that turns those sums into the
reported values.

The method as published: box counting (the default)
---------------------------------------------------
The source text: "Trajectories are converted into pixels in different
resolutions, considering increasing detail according to scale (the 'Richardson
Effect'). We opted for a 0.7m pixel and a scaling factor of 5 times (r = 0.7,
3.5, 17.5, 87.5, 437.5, 2,187.5)", fitted as

    ``log N = log K + (1 - D) log r``

"where N is the number of pixels in each resolution scale". :func:`box_counts`
rasterises the path once at the finest pixel, on a grid anchored at the working
CRS origin — the same grid for every trajectory, so two paths are measured at the
same scales — and derives each coarser resolution by integer division, which the
integer ``pixel_factor`` makes exact. :func:`box_counting_dimension` fits the
slope.

Two things in the published recipe need saying, and both are config switches
rather than silent choices:

* **The exponent.** With ``N`` a *pixel count*, a straight line has
  ``N ≈ L/r`` — slope ``-1`` — and ``1 - slope`` is ``2``. The printed relation is
  Richardson's, which is about measured *length*: ``L(r) = N·r ∝ r^(1-D)``, i.e.
  ``N ∝ r^(-D)``. The default reports that consistent reading, ``D = -slope``
  (straight line = 1); ``as_written = true`` reports ``1 - slope``, exactly one
  higher.
* **Saturated resolutions.** A path shorter than a coarse pixel occupies one
  pixel there; ``log N = 0`` pins the fit and drags ``D`` down in proportion to
  how small the path is, so *fragmentation* stops being separable from *extent*.
  ``drop_saturated`` (default on) leaves resolutions with ``N < 2`` out of the
  fit; fewer than ``min_scales`` left yields ``NaN``, never a two-point fit
  (invariant #2).

Because that grid is the same for every trajectory, the pixels a path occupies
are the **union of the pixels of its edges** — a property of the edge, not of the
path. :func:`measure_path` therefore rasterises each traversed edge once
(:class:`EdgeShapeCache`, bounded by the edge count like every other piece of
aggregation state) and a path costs one concatenation and a handful of
``numpy.unique`` calls. The numbers are identical to rasterising the whole
polyline; only the work is shared.

Box counting on a finite grid carries a small discretisation bias — a straight
line occupies ``⌊L/r⌋ + 1`` pixels, not ``L/r`` — so a straight path scores
slightly *below* ``1.0``, approaching it as the path gets longer.

The cross-check: the divider method
-----------------------------------
``method = "divider"`` is the earlier reading (Richardson's structured walk):
rulers of several lengths ``δ`` *relative to the path's own length*, steps
``N(δ)``, ``D = -slope`` of ``log N`` on ``log δ`` — :func:`divider_dimension`.
A straight path gives ``N(δ) = L/δ`` exactly and therefore ``D = 1.0`` to machine
precision. Its ladder is floored at the path's **mean segment length** (a
polyline has no structure below its straight pieces), so a path with too few
segments to span ``min_scales`` rungs yields ``NaN``. It measures the *node*
polyline — the street segment as the atom — whatever ``use_edge_geometry`` says.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from segx.metrics.base import Metric
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import FragmentationConfig, Settings
    from segx.routing.trajectories import EdgeId

__all__ = [
    "EdgeShapeCache",
    "EdgeXY",
    "PathFragmentationMetric",
    "PathShape",
    "box_counting_dimension",
    "box_counts",
    "divider_dimension",
    "measure_path",
    "path_polyline",
    "polyline_length",
    "structured_walk",
]

#: a point in the working CRS (metres)
Point = tuple[float, float]

#: resolves a traversed edge to its vertex sequence **oriented u→v** (endpoints
#: included), or ``None`` when the edge has no geometry of its own
EdgeXY = Callable[["EdgeId"], "Sequence[Point] | None"]

#: guard against a pathological ruler walk; ``N`` can never legitimately exceed
#: this multiple of the vertex count, since the ladder is floored at the mean
#: segment length (see the module docstring).
_MAX_STEPS_FACTOR = 100


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def path_polyline(
    path: list[EdgeId],
    node_xy: dict[int, Point],
    edge_xy: EdgeXY | None = None,
) -> list[Point] | None:
    """The ordered vertex sequence of a trajectory's path, in the working CRS.

    ``path`` is a list of **traversed** ``(u, v, key)`` edges in order, so the
    node sequence is ``u₀, v₀, v₁, …`` — see
    :func:`segx.routing._paths.node_path_to_edges`, which builds them that way.
    Without ``edge_xy`` the vertices are the *nodes* — one straight piece per
    street segment. With it, each edge that has a geometry of its own (a curved
    street) contributes its interior vertices too, which is what a 0.7 m pixel
    actually sees.

    Returns ``None`` when the path is empty or any node is missing a coordinate —
    an unmeasurable path, which the accumulator books as such rather than
    substituting a straight line for it.
    """
    if not path:
        return None
    poly: list[Point] = []

    def _push(xy: Point) -> None:
        # drop repeated vertices: a zero-length piece carries no direction and
        # would divide by zero in the ruler walk
        if poly and abs(xy[0] - poly[-1][0]) < 1e-9 and abs(xy[1] - poly[-1][1]) < 1e-9:
            return
        poly.append(xy)

    start = node_xy.get(path[0][0])
    if start is None:
        return None
    _push(start)
    for edge in path:
        end = node_xy.get(edge[1])
        if end is None:
            return None
        coords = edge_xy(edge) if edge_xy is not None else None
        if coords is not None:
            # interior vertices only: the end nodes come from `node_xy`, so the
            # polyline stays continuous even if a geometry's ends are a hair off
            for xy in coords[1:-1]:
                _push((float(xy[0]), float(xy[1])))
        _push(end)
    return poly if len(poly) >= 2 else None


def polyline_length(poly: list[Point]) -> float:
    """Total arclength of the polyline, in the units of its coordinates (metres)."""
    return math.fsum(math.dist(a, b) for a, b in zip(poly, poly[1:], strict=False))


def _exit_fraction(a: Point, b: Point, centre: Point, radius: float) -> float | None:
    """Where the segment ``a→b`` leaves the circle of ``radius`` around ``centre``.

    Returns the parameter ``u ∈ (0, 1]`` of the crossing, or ``None`` if the
    segment ends before reaching it. Called only with ``a`` **inside** the circle,
    so the quadratic's larger root is the outward crossing.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - centre[0], a[1] - centre[1]
    qa = dx * dx + dy * dy
    if qa <= 0.0:  # pragma: no cover - deduped by path_polyline
        return None
    qb = 2.0 * (dx * fx + dy * fy)
    qc = fx * fx + fy * fy - radius * radius
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return None
    u = (-qb + math.sqrt(disc)) / (2.0 * qa)
    return u if 0.0 < u <= 1.0 else None


def structured_walk(poly: list[Point], ruler: float) -> float:
    """``N(δ)`` — the number of steps of length ``ruler`` along ``poly``.

    Richardson's structured walk: plant the compass at the start, step to the
    first point of the curve exactly ``ruler`` away, plant it there, repeat. The
    leftover tail is added as the **chord** from the last anchor to the endpoint
    divided by the ruler, which keeps ``N`` continuous in ``δ`` (an integer count
    puts a staircase into the regression) and makes a straight path measure
    exactly ``L/δ`` at every scale — the reason ``D == 1.0`` comes out clean.
    """
    if ruler <= 0.0:  # pragma: no cover - guarded by divider_dimension
        return math.nan
    anchor = poly[0]
    idx, pt = 0, poly[0]
    steps = 0.0
    max_steps = _MAX_STEPS_FACTOR * len(poly)
    while steps < max_steps:
        hit_idx, hit_pt = -1, None
        cursor = pt
        for j in range(idx, len(poly) - 1):
            start = cursor if j == idx else poly[j]
            u = _exit_fraction(start, poly[j + 1], anchor, ruler)
            if u is not None:
                hit_idx = j
                hit_pt = (start[0] + u * (poly[j + 1][0] - start[0]), start[1] + u * (poly[j + 1][1] - start[1]))
                break
        if hit_pt is None:
            break
        idx, pt, anchor = hit_idx, hit_pt, hit_pt
        steps += 1.0
    return steps + math.dist(anchor, poly[-1]) / ruler


def _ruler_ladder(
    length: float,
    n_segments: int,
    cfg: FragmentationConfig,
) -> list[float]:
    """The ruler lengths ``δ`` to measure at, largest first.

    Starts at ``largest_fraction × L`` and shrinks by ``scale_ratio`` each rung,
    stopping at the **mean segment length** — the resolution floor below which a
    polyline stops being fractal (see the module docstring).
    """
    if length <= 0.0 or n_segments <= 0:
        return []
    floor = length / n_segments
    rulers: list[float] = []
    delta = cfg.largest_fraction * length
    for _ in range(cfg.n_scales):
        if delta < floor:
            break
        rulers.append(delta)
        delta /= cfg.scale_ratio
    return rulers


def divider_dimension(poly: list[Point], n_segments: int, cfg: FragmentationConfig) -> float:
    """The linear fractal dimension ``D`` of ``poly`` by the divider method.

    Fits ``log N(δ) = -D·log δ + c`` by ordinary least squares over the ruler
    ladder. Returns ``NaN`` — never a substitute value — when the path cannot
    span ``cfg.min_scales`` rungs above its resolution floor, or when the fitted
    scales are degenerate.
    """
    length = polyline_length(poly)
    rulers = _ruler_ladder(length, n_segments, cfg)
    if len(rulers) < cfg.min_scales:
        return math.nan

    xs: list[float] = []
    ys: list[float] = []
    for delta in rulers:
        n = structured_walk(poly, delta)
        if not math.isfinite(n) or n <= 0.0:  # pragma: no cover - defensive
            continue
        xs.append(math.log(delta))
        ys.append(math.log(n))
    if len(xs) < cfg.min_scales:  # pragma: no cover - defensive
        return math.nan
    return -_slope(xs, ys)


def _slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope of ``ys`` on ``xs``; ``NaN`` if ``xs`` is degenerate."""
    n_pts = len(xs)
    mean_x = math.fsum(xs) / n_pts
    mean_y = math.fsum(ys) / n_pts
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:  # pragma: no cover - scales are distinct by construction
        return math.nan
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx


# --------------------------------------------------------------------------- #
# box counting
# --------------------------------------------------------------------------- #
def _grid_crossings(g0: np.ndarray, g1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each segment crosses the integer grid lines of one axis.

    ``g0``/``g1`` are the segments' start/end coordinates on that axis in grid
    units. Returns ``(segment index, parameter t ∈ (0, 1))`` for every integer
    strictly between the two ends.
    """
    lo = np.floor(np.minimum(g0, g1)).astype(np.int64) + 1
    hi = np.ceil(np.maximum(g0, g1)).astype(np.int64) - 1
    count = np.clip(hi - lo + 1, 0, None)
    seg = np.repeat(np.arange(len(g0)), count)
    first = np.cumsum(count) - count
    k = lo[seg] + (np.arange(int(count.sum())) - first[seg])
    return seg, (k - g0[seg]) / (g1[seg] - g0[seg])


def _finest_cells(poly: list[Point], pixel: float) -> np.ndarray:
    """The distinct pixels of size ``pixel`` that ``poly`` passes through, as sorted keys.

    Exact grid traversal, not sampling: every segment is cut at each grid line it
    crosses, and the midpoint of each resulting piece names the one pixel that
    piece lies in. The grid is anchored at the CRS origin, so it is the same grid
    for every path.
    """
    pts = np.asarray(poly, dtype=float) / pixel
    a, b = pts[:-1], pts[1:]
    n_seg = len(a)
    seg_x, t_x = _grid_crossings(a[:, 0], b[:, 0])
    seg_y, t_y = _grid_crossings(a[:, 1], b[:, 1])
    ends = np.arange(n_seg)
    seg = np.concatenate([ends, seg_x, seg_y, ends])
    t = np.concatenate([np.zeros(n_seg), t_x, t_y, np.ones(n_seg)])
    order = np.lexsort((t, seg))
    seg, t = seg[order], t[order]
    # a piece runs between consecutive cuts of the *same* segment; a zero-length
    # piece (the path through a pixel corner) lies in no pixel of its own
    piece = (seg[1:] == seg[:-1]) & (t[1:] - t[:-1] > 1e-12)
    s = seg[:-1][piece]
    mid = 0.5 * (t[:-1][piece] + t[1:][piece])
    xy = a[s] + mid[:, None] * (b[s] - a[s])
    cells = np.floor(xy).astype(np.int64)
    return _distinct(cells[:, 0] * _KEY_BASE + cells[:, 1])


#: One int64 key per pixel, ``ix * 2**32 + iy`` — injective while ``|iy| < 2**31``,
#: i.e. for any projected coordinate at any sane pixel size.
_KEY_BASE = 4294967296
_KEY_HALF = 2147483648


def _distinct(keys: np.ndarray) -> np.ndarray:
    """The sorted distinct values of *keys* (sorted in place).

    ``numpy.unique`` without its wrapper: this runs half a dozen times per
    trajectory on small arrays, where the call overhead outweighs the sort.
    """
    if len(keys) < 2:
        return keys
    keys.sort()
    keep = np.empty(len(keys), dtype=bool)
    keep[0] = True
    np.not_equal(keys[1:], keys[:-1], out=keep[1:])
    return keys[keep]


def _coarsen(keys: np.ndarray, factor: int) -> np.ndarray:
    """The distinct pixels one resolution up: both indexes floor-divided by *factor*."""
    ix = (keys + _KEY_HALF) >> 32
    iy = keys - (ix << 32)
    return _distinct((ix // factor) * _KEY_BASE + iy // factor)


def _counts_from_keys(keys: np.ndarray, cfg: FragmentationConfig) -> list[tuple[float, int]]:
    """``(pixel size, N)`` per resolution from the finest level's distinct pixel keys."""
    counts: list[tuple[float, int]] = []
    size = cfg.pixel_m
    for level in range(cfg.n_pixel_scales):
        if level:
            keys = _coarsen(keys, cfg.pixel_factor)
            size *= cfg.pixel_factor
        counts.append((size, len(keys)))
    return counts


def box_counts(poly: list[Point], cfg: FragmentationConfig) -> list[tuple[float, int]]:
    """``(pixel size, N)`` per resolution, finest first — ``N`` the pixels occupied.

    The path is rasterised once, at ``cfg.pixel_m``; each coarser resolution is
    the previous one's pixels floor-divided by ``cfg.pixel_factor``, which is
    exact because the factor is an integer and the grids share an origin.
    """
    return _counts_from_keys(_finest_cells(poly, cfg.pixel_m), cfg)


def box_counting_dimension(poly: list[Point], cfg: FragmentationConfig) -> float:
    """The fractal dimension ``D`` of ``poly`` by box counting (the paper's method).

    Fits ``log N`` on ``log r`` over the resolutions of :func:`box_counts` and
    returns ``-slope`` — or ``1 - slope`` under ``cfg.as_written``. Returns
    ``NaN`` when fewer than ``cfg.min_scales`` resolutions are left after
    ``cfg.drop_saturated`` removes the single-pixel ones.
    """
    return _dimension_from_counts(box_counts(poly, cfg), cfg)


def _dimension_from_counts(counts: list[tuple[float, int]], cfg: FragmentationConfig) -> float:
    """The slope fit of :func:`box_counting_dimension`, over ready-made counts."""
    if cfg.drop_saturated:
        counts = [(r, n) for r, n in counts if n >= 2]
    if len(counts) < cfg.min_scales:
        return math.nan
    slope = _slope([math.log(r) for r, _n in counts], [math.log(n) for _r, n in counts])
    return 1.0 - slope if cfg.as_written else -slope


class PathShape:
    """The shape measurements of one trajectory, all four terms at once.

    ``fragmentation`` is the fractal dimension ``D``; ``n_segments`` is Netto's
    literal "number of street segments used"; ``length_m`` and ``extent_m`` (the
    straight-line origin→destination distance) are the *extent* term of the same
    composed measure, kept because they are unambiguous whatever ``D``'s final
    definition turns out to be. ``reason`` says why ``fragmentation`` is ``NaN``
    when it is: ``"too_short"`` (divider: too few segments for the ladder) or
    ``"saturated"`` (box counting: too few resolutions with more than one pixel).
    """

    __slots__ = ("extent_m", "fragmentation", "length_m", "n_segments", "reason")

    def __init__(
        self,
        n_segments: int,
        length_m: float,
        extent_m: float,
        fragmentation: float,
        reason: str | None = None,
    ):
        self.n_segments = n_segments
        self.length_m = length_m
        self.extent_m = extent_m
        self.fragmentation = fragmentation
        self.reason = reason

    @property
    def measured(self) -> bool:
        """True when ``fragmentation`` is a real number (the path spanned the ladder)."""
        return math.isfinite(self.fragmentation)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PathShape(n_segments={self.n_segments}, length_m={self.length_m:.1f}, "
            f"extent_m={self.extent_m:.1f}, fragmentation={self.fragmentation:.4f})"
        )


class EdgeShapeCache:
    """Each traversed edge's contribution to a path's shape, computed once.

    For box counting, the pixels a path occupies are the union of its edges'
    pixels (the grid is anchored at the CRS origin, the same for every path), and
    its length is the sum of its edges' pieces. Both are therefore properties of
    the *edge*, and thousands of trajectories ride the same avenue. An entry is
    ``(end point, piece lengths, pixel keys)`` for one edge **entered at one
    point**: the key includes the point the polyline had reached, not just the
    edge, because :func:`path_polyline` joins consecutive edges end to end
    without assuming they share a node — after ``expand_transit_rides`` drops a
    transit path's access edges they often do not. Keyed that way, the cached
    pieces are exactly the ones the whole-path polyline would contain.

    Bounded by the number of distinct edges traversed (times the directions they
    are traversed in), never by the number of trajectories — the same footing as
    the edge accumulator's own state (invariant #6). At 0.7 m a 100 m block is
    about 150 pixels, i.e. ~1 KB per cached edge.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[tuple[Point, EdgeId], tuple[Point, tuple[float, ...], np.ndarray] | None] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def piece(
        self,
        start: Point,
        edge: EdgeId,
        node_xy: dict[int, Point],
        edge_xy: EdgeXY | None,
        pixel: float,
    ) -> tuple[Point, tuple[float, ...], np.ndarray] | None:
        """``(end point, piece lengths, pixel keys)`` for *edge* entered at *start*.

        ``None`` when the edge's end node has no coordinate. An edge that adds no
        new vertex (zero length) returns *start* unchanged and nothing else.
        """
        key = (start, edge)
        try:
            return self._entries[key]
        except KeyError:
            pass
        end = node_xy.get(edge[1])
        if end is None:
            self._entries[key] = None
            return None
        poly = _extend_polyline([start], edge, end, edge_xy)
        if len(poly) < 2:
            entry = (start, (), _NO_KEYS)
        else:
            lengths = tuple(math.dist(a, b) for a, b in zip(poly, poly[1:], strict=False))
            entry = (poly[-1], lengths, _finest_cells(poly, pixel))
        self._entries[key] = entry
        return entry


_NO_KEYS = np.empty(0, dtype=np.int64)
#: sentinel for "not in the cache yet" — ``None`` is a real entry (no end coordinate)
_UNSEEN = object()


def _extend_polyline(poly: list[Point], edge: EdgeId, end: Point, edge_xy: EdgeXY | None) -> list[Point]:
    """Append one edge's vertices to *poly* exactly as :func:`path_polyline` does."""

    def _push(xy: Point) -> None:
        if abs(xy[0] - poly[-1][0]) < 1e-9 and abs(xy[1] - poly[-1][1]) < 1e-9:
            return
        poly.append(xy)

    coords = edge_xy(edge) if edge_xy is not None else None
    if coords is not None:
        for xy in coords[1:-1]:
            _push((float(xy[0]), float(xy[1])))
    _push(end)
    return poly


def measure_path(
    path: list[EdgeId],
    node_xy: dict[int, Point],
    cfg: FragmentationConfig,
    edge_xy: EdgeXY | None = None,
    cache: EdgeShapeCache | None = None,
) -> PathShape | None:
    """Measure one trajectory's path, or ``None`` if its geometry is unresolvable.

    ``None`` means "this path has no measurable geometry" (empty, or a node with
    no coordinate); a :class:`PathShape` with ``fragmentation = NaN`` means "real
    geometry, too little of it to carry a fractal dimension". The accumulator
    books the two separately, because they are different findings.

    ``n_segments`` always counts **street segments** (node to node), whatever the
    geometry traced; ``length_m`` follows the traced geometry.

    ``cache`` shares the per-edge rasterisation between calls (box counting
    only); pass one :class:`EdgeShapeCache` for a whole stream of trajectories.
    Without it each call rasterises its own edges — same numbers, more work.
    """
    nodes = path_polyline(path, node_xy)
    if nodes is None:
        return None
    n_segments = len(nodes) - 1
    extent = math.dist(nodes[0], nodes[-1])
    geometry = edge_xy if cfg.use_edge_geometry else None

    if cfg.method == "divider":
        traced = (path_polyline(path, node_xy, geometry) or nodes) if geometry is not None else nodes
        length = polyline_length(traced)
        d = math.nan if n_segments < cfg.min_segments else divider_dimension(nodes, n_segments, cfg)
        reason = None if math.isfinite(d) else "too_short"
        return PathShape(n_segments=n_segments, length_m=length, extent_m=extent, fragmentation=d, reason=reason)

    if cache is None:
        cache = EdgeShapeCache()
    last = nodes[0]
    lengths: list[float] = []
    keys: list[np.ndarray] = []
    known = cache._entries
    for edge in path:
        piece = known.get((last, edge), _UNSEEN)
        if piece is _UNSEEN:
            piece = cache.piece(last, edge, node_xy, geometry, cfg.pixel_m)
        if piece is None:  # pragma: no cover - `nodes` resolved, so every end node has a coordinate
            return None
        last, piece_lengths, piece_keys = piece
        if piece_lengths:
            lengths.extend(piece_lengths)
            keys.append(piece_keys)
    if not keys:  # pragma: no cover - every edge collapsed onto the point before it
        d = box_counting_dimension(nodes, cfg)
        reason = None if math.isfinite(d) else "saturated"
        return PathShape(n_segments, polyline_length(nodes), extent, d, reason)
    length = math.fsum(lengths)
    cells = _distinct(np.concatenate(keys)) if len(keys) > 1 else keys[0]
    d = _dimension_from_counts(_counts_from_keys(cells, cfg), cfg)
    reason = None if math.isfinite(d) else "saturated"
    return PathShape(n_segments=n_segments, length_m=length, extent_m=extent, fragmentation=d, reason=reason)


# --------------------------------------------------------------------------- #
# the metric plugin
# --------------------------------------------------------------------------- #
_PARTITION_KEYS: tuple[str, ...] = ("mode", "category", "binding", "attribute")

_OUTPUT_COLUMNS: tuple[str, ...] = (
    *_PARTITION_KEYS,
    "class",
    "fragmentation",
    "fragmentation_sd",
    "mean_segments",
    "mean_length_m",
    "mean_extent_m",
    "detour_ratio",
    "n_paths",
    "n_measured",
)


@register_metric
class PathFragmentationMetric(Metric):
    """Per-group mean path fragmentation ``D``, plus the extent terms. P7.3.

    Unlike every other registered metric this one does not read the edge table:
    it declares ``requires = ("path_table",)`` and the orchestrator hands it the
    bounded per-``(partition, attribute, class)`` sums streamed by
    :class:`~segx.aggregation.path_accumulator.PathAccumulator`. All this class
    does is turn those sums into means — the division of labour that keeps the
    formula in one place (invariant #5) without storing a trajectory (invariant
    #6).

    Each value is weighted by the class's demographic mass on the trajectory, so
    ``fragmentation`` reads "the mean fractal dimension of the paths travelled by
    a member of this group" — which is the comparison the source text is after:
    *"different patterns of appropriation and mobility between social classes"*.
    """

    name = "path_fragmentation"
    description = (
        "Fractal dimension of each routed path (box counting on 0.7 m pixels at 5× coarser "
        "scales): how fragmented a class's trips are. Mean per class, weighted by the class's "
        "presence on each path."
    )
    formula = (
        "N(r) = K · r^(1−D)   r = 0.7, 3.5, 17.5, 87.5, 437.5, 2187.5 m\n"
        "D = 1 − slope of log N(r) against log r\n"
        "N(r) = pixels the path occupies at scale r"
    )
    value_column = "fragmentation"
    requires = ("path_table",)

    def compute(self, edge_table, settings: Settings, path_table=None):  # noqa: ARG002
        import numpy as np
        import pandas as pd

        if path_table is None or path_table.empty:
            return pd.DataFrame(columns=list(_OUTPUT_COLUMNS))

        df = path_table.copy()
        w_all = df["w_total"].to_numpy(dtype=float)
        w_meas = df["w_measured"].to_numpy(dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            # D is averaged over the paths that actually carried one; the extent
            # terms over every deposited path, so a run where most trajectories
            # are too short still reports honest lengths next to a sparse D.
            mean_d = np.where(w_meas > 0.0, df["w_fragmentation"].to_numpy(float) / w_meas, np.nan)
            var = np.where(w_meas > 0.0, df["w_fragmentation_sq"].to_numpy(float) / w_meas - mean_d**2, np.nan)
            mean_len = np.where(w_all > 0.0, df["w_length_m"].to_numpy(float) / w_all, np.nan)
            mean_ext = np.where(w_all > 0.0, df["w_extent_m"].to_numpy(float) / w_all, np.nan)
            mean_seg = np.where(w_all > 0.0, df["w_n_segments"].to_numpy(float) / w_all, np.nan)
            detour = np.where(mean_ext > 0.0, mean_len / mean_ext, np.nan)

        out = df[[*_PARTITION_KEYS, "class"]].copy()
        out["fragmentation"] = mean_d
        # clamp: a float-cancellation negative under a numerically-zero variance
        out["fragmentation_sd"] = np.sqrt(np.clip(var, 0.0, None))
        out["mean_segments"] = mean_seg
        out["mean_length_m"] = mean_len
        out["mean_extent_m"] = mean_ext
        out["detour_ratio"] = detour
        out["n_paths"] = df["n_paths"].to_numpy(dtype=int)
        out["n_measured"] = df["n_measured"].to_numpy(dtype=int)
        return out.reset_index(drop=True)[list(_OUTPUT_COLUMNS)]
