"""Path fragmentation tests (P7.3) — hand-computable throughout.

Two methods are covered. **Box counting** is the default and the paper's: pixels
at 0.7 m × 5ᵏ, whose counts on a straight line can be checked by hand
(``⌊L/r⌋ + 1``). The **divider** method is the cross-check, and its notes follow.

The anchor case is a **straight** path: the structured walk on a straight line
covers exactly ``L/δ`` ruler lengths at *every* scale, so the regression of
``log N`` on ``log δ`` has slope exactly ``-1`` and ``D == 1.0``. That is the
calibration point of the whole measure — anything crookeder must score above it,
and any implementation error in the ruler walk shows up here first.

The counter-case is a **Koch curve**, whose true dimension ``log 4 / log 3 ≈
1.2619`` is known analytically. The estimator is expected to *under*shoot it on a
short ladder (a finite ladder cannot see the whole cascade) but must approach it
monotonically as rungs are added — the property that says the slope is measuring
the cascade rather than an artefact.

The third case is the one that must be read carefully in any report: a **regular
90° staircase** scores ``D ≈ 1``, not ``D ≈ 1.4``. A staircase has the *same*
crookedness at every scale, so the detour is absorbed into the fit's intercept
and never reaches the slope. ``D`` measures how crookedness **changes** with
scale, not how crooked a path is — which is precisely why the metric also reports
``detour_ratio`` and ``mean_segments``, the plain readings, beside it.
"""

from __future__ import annotations

import math

import pytest

from segx.aggregation.path_accumulator import PathAccumulator
from segx.config import (
    DemographicAttribute,
    DemographicSchema,
    FragmentationConfig,
    MetricsConfig,
    Settings,
)
from segx.metrics.fragmentation import (
    PathFragmentationMetric,
    box_counting_dimension,
    box_counts,
    divider_dimension,
    measure_path,
    path_polyline,
    polyline_length,
    structured_walk,
)
from segx.routing.trajectories import Trajectory

KOCH_D = math.log(4) / math.log(3)


def _settings(**frag) -> Settings:
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        ),
        metrics=MetricsConfig(fragmentation=FragmentationConfig(**frag)),
    )


def _straight(n_segments: int, step: float = 100.0) -> dict[int, tuple[float, float]]:
    """``node -> (x, y)`` for a straight east–west chain of ``n_segments`` edges."""
    return {i: (float(i) * step, 0.0) for i in range(n_segments + 1)}


def _chain(n_segments: int) -> list[tuple[int, int, int]]:
    return [(i, i + 1, 0) for i in range(n_segments)]


def _koch(p0, p1, depth):
    if depth == 0:
        return [p0]
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = (x1 - x0) / 3.0, (y1 - y0) / 3.0
    a = (x0 + dx, y0 + dy)
    b = (x0 + 2 * dx, y0 + 2 * dy)
    ang = math.radians(-60)
    px, py = b[0] - a[0], b[1] - a[1]
    peak = (a[0] + px * math.cos(ang) - py * math.sin(ang), a[1] + px * math.sin(ang) + py * math.cos(ang))
    return _koch(p0, a, depth - 1) + _koch(a, peak, depth - 1) + _koch(peak, b, depth - 1) + _koch(b, p1, depth - 1)


def koch_polyline(depth: int = 5) -> list[tuple[float, float]]:
    return [*_koch((0.0, 0.0), (2000.0, 0.0), depth), (2000.0, 0.0)]


# --------------------------------------------------------------------------- #
# the ruler walk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ruler", [1000.0, 500.0, 250.0, 125.0, 64.0])
def test_structured_walk_on_a_straight_line_is_exactly_length_over_ruler(ruler):
    """The calibration identity. The fractional tail is the *chord* from the last
    anchor to the endpoint, which is what makes this exact rather than a staircase."""
    poly = [(float(i) * 100.0, 0.0) for i in range(21)]
    assert structured_walk(poly, ruler) == pytest.approx(polyline_length(poly) / ruler, rel=1e-12)


def test_a_straight_path_has_fractal_dimension_one():
    cfg = FragmentationConfig()
    poly = [(float(i) * 100.0, 0.0) for i in range(21)]
    assert divider_dimension(poly, 20, cfg) == pytest.approx(1.0, abs=1e-9)


def test_koch_curve_scores_above_one_and_approaches_its_true_dimension():
    """Under-shooting a short ladder is expected; drifting *away* is a bug."""
    poly = koch_polyline(depth=5)
    n_segments = len(poly) - 1
    coarse = divider_dimension(poly, n_segments, FragmentationConfig(n_scales=3))
    fine = divider_dimension(poly, n_segments, FragmentationConfig(n_scales=10))
    assert 1.0 < coarse < fine < KOCH_D
    assert fine == pytest.approx(KOCH_D, abs=0.1)


def test_a_regular_staircase_scores_near_one_because_its_detour_is_scale_free():
    """The interpretation guard: `D` is not "how winding", it is "how the winding
    changes with scale". A staircase is equally winding at every scale."""
    poly = []
    x = y = 0.0
    for _ in range(12):
        poly.append((x, y))
        x += 100.0
        poly.append((x, y))
        y += 100.0
    poly.append((x, y))
    d = divider_dimension(poly, len(poly) - 1, FragmentationConfig())
    assert d == pytest.approx(1.0, abs=0.05)
    # ...while the plain detour reading does see it: L/chord = sqrt(2).
    assert polyline_length(poly) / math.dist(poly[0], poly[-1]) == pytest.approx(math.sqrt(2), rel=1e-9)


# --------------------------------------------------------------------------- #
# "unmeasurable" is a typed result, never a zero (invariant #2)
# --------------------------------------------------------------------------- #
def test_a_path_too_short_for_the_ladder_reports_nan_not_zero():
    cfg = FragmentationConfig(method="divider")
    shape = measure_path(_chain(4), _straight(4), cfg)
    assert shape is not None  # real geometry ...
    assert math.isnan(shape.fragmentation)  # ... but no dimension
    assert not shape.measured
    assert shape.n_segments == 4
    assert shape.extent_m == pytest.approx(400.0)


def test_the_implied_floor_of_the_default_ladder_is_eight_segments():
    """``scale_ratio**(min_scales-1) / largest_fraction`` = 2**2 / 0.5 = 8 — the
    documented reason ``min_segments`` defaults to 8 rather than something rounder."""
    cfg = FragmentationConfig(method="divider", min_segments=2)  # get the guard out of the way
    assert math.isnan(measure_path(_chain(7), _straight(7), cfg).fragmentation)
    assert measure_path(_chain(8), _straight(8), cfg).measured


def test_a_path_whose_nodes_have_no_coordinates_is_unmeasurable_not_straight():
    """``None`` (no geometry) and ``NaN`` (geometry, no dimension) are different
    findings and the accumulator books them separately."""
    assert measure_path([(0, 1, 0), (1, 99, 0)], _straight(4), FragmentationConfig()) is None
    assert measure_path([], _straight(4), FragmentationConfig()) is None


def test_repeated_vertices_are_dropped_so_a_zero_length_piece_cannot_divide_by_zero():
    node_xy = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (100.0, 0.0)}
    assert path_polyline([(0, 1, 0), (1, 2, 0)], node_xy) == [(0.0, 0.0), (100.0, 0.0)]


def test_config_rejects_a_ladder_that_cannot_produce_a_slope():
    with pytest.raises(ValueError, match="scale_ratio"):
        FragmentationConfig(scale_ratio=1.0)
    with pytest.raises(ValueError, match="min_scales"):
        FragmentationConfig(min_scales=1)
    with pytest.raises(ValueError, match="below min_scales"):
        FragmentationConfig(n_scales=2, min_scales=3)
    with pytest.raises(ValueError, match="largest_fraction"):
        FragmentationConfig(largest_fraction=1.5)


# --------------------------------------------------------------------------- #
# accumulation + the metric
# --------------------------------------------------------------------------- #
def _trajectory(path, mode="walk") -> Trajectory:
    return Trajectory(origin_id="o", destination_id="d", mode=mode, category="commute", binding="od", path=path)


def _vector(low: float, high: float):
    return {"income": {"low": low, "high": high}}


def test_the_accumulator_weights_each_class_by_its_mass_on_the_trajectory():
    """Two groups, two path shapes, hand-checkable means.

    ``low`` travels the crooked path only; ``high`` travels the straight one only.
    So ``low``'s mean D is the Koch value and ``high``'s is exactly 1.0 — the
    between-class comparison the source text is after.
    """
    settings = _settings(method="divider", n_scales=10)
    koch = koch_polyline(depth=5)
    node_xy = {i: pt for i, pt in enumerate(koch)}
    offset = len(koch)
    node_xy.update({offset + i: (float(i) * 100.0, -5000.0) for i in range(21)})

    acc = PathAccumulator(settings, node_xy)
    acc.add(_trajectory([(i, i + 1, 0) for i in range(len(koch) - 1)]), _vector(low=4.0, high=0.0))
    acc.add(_trajectory([(offset + i, offset + i + 1, 0) for i in range(20)]), _vector(low=0.0, high=3.0))

    frame = PathFragmentationMetric().compute(None, settings, path_table=acc.to_frame())
    by_class = frame.set_index("class")

    assert by_class.loc["high", "fragmentation"] == pytest.approx(1.0, abs=1e-9)
    assert by_class.loc["low", "fragmentation"] > 1.1
    assert by_class.loc["high", "n_paths"] == 1
    assert by_class.loc["low", "n_paths"] == 1
    # the straight path's extent is its length, so its detour ratio is exactly 1
    assert by_class.loc["high", "detour_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert by_class.loc["low", "detour_ratio"] > 3.0  # Koch: L/chord = (4/3)**5


def test_a_class_with_no_mass_on_a_trajectory_does_not_inherit_its_shape():
    """The bug this guards: averaging every class over every path would report
    that `high` walks the crooked route, when nobody in `high` travelled it."""
    settings = _settings()
    acc = PathAccumulator(settings, _straight(20))
    acc.add(_trajectory(_chain(20)), _vector(low=5.0, high=0.0))

    frame = PathFragmentationMetric().compute(None, settings, path_table=acc.to_frame()).set_index("class")
    assert frame.loc["low", "n_paths"] == 1
    # present as a row (so the group does not silently vanish) but with no value
    assert frame.loc["high", "n_paths"] == 0
    assert math.isnan(frame.loc["high", "fragmentation"])


def test_unresolved_and_unreachable_trajectories_are_counted_not_deposited():
    settings = _settings(method="divider")
    acc = PathAccumulator(settings, _straight(20))
    acc.add(_trajectory(_chain(20)), _vector(low=math.nan, high=math.nan))
    acc.add(_trajectory(None), _vector(low=1.0, high=1.0))
    acc.add(_trajectory([(0, 1, 0), (1, 999, 0)]), _vector(low=1.0, high=1.0))
    acc.add(_trajectory(_chain(3)), _vector(low=1.0, high=1.0))
    acc.add(_trajectory(_chain(20)), _vector(low=1.0, high=1.0))

    stats = acc.stats()
    assert stats["n_unresolved"] == 1
    assert stats["n_unreachable"] == 1
    assert stats["n_no_geometry"] == 1
    assert stats["n_too_short"] == 1
    assert stats["n_saturated"] == 0
    assert stats["n_measured"] == 1
    # the ruler ladder is an assumption, so it ships with the numbers (invariant #7)
    assert stats["params"]["n_scales"] == FragmentationConfig().n_scales


def test_the_partition_is_the_trajectorys_not_the_legs():
    """A transit trip's walking legs are labelled ``walk`` for *edge* occupancy
    (P7.2), but a fractal dimension is a property of the one continuous curve —
    including the walk to the stop. It must stay filed under ``transit``."""
    settings = _settings()
    acc = PathAccumulator(settings, _straight(20))
    trajectory = _trajectory(_chain(20), mode="transit")
    trajectory.leg_modes = ["walk"] * 5 + ["transit"] * 10 + ["walk"] * 5
    acc.add(trajectory, _vector(low=2.0, high=0.0))

    assert acc.partitions() == [("transit", "commute", "od")]
    assert set(acc.to_frame()["mode"]) == {"transit"}


def test_the_metric_returns_an_empty_frame_when_no_paths_were_measured():
    settings = _settings()
    empty = PathAccumulator(settings, {}).to_frame()
    assert PathFragmentationMetric().compute(None, settings, path_table=empty).empty
    assert PathFragmentationMetric().compute(None, settings, path_table=None).empty


def test_the_metric_is_registered_and_declares_the_frame_it_needs():
    from segx.metrics.registry import available_metrics, get_metric

    assert "path_fragmentation" in available_metrics()
    assert get_metric("path_fragmentation").requires == ("path_table",)


# --------------------------------------------------------------------------- #
# box counting — the paper's method, and the default
# --------------------------------------------------------------------------- #
def _line(length: float, n: int = 3) -> list[tuple[float, float]]:
    return [(length * i / n, 0.0) for i in range(n + 1)]


def test_box_counts_on_a_straight_line_are_floor_of_length_over_pixel_plus_one():
    """300 m at 0.7 m: ``⌊300/0.7⌋ = 428``, pixel indices 0..428 → 429. The two
    coarsest resolutions hold the whole path in one pixel."""
    counts = box_counts(_line(300.0), FragmentationConfig())
    assert [n for _r, n in counts] == [429, 86, 18, 4, 1, 1]


def test_the_default_ladder_is_the_published_one():
    sizes = [r for r, _n in box_counts(_line(300.0), FragmentationConfig())]
    assert sizes == pytest.approx([0.7, 3.5, 17.5, 87.5, 437.5, 2187.5])


def test_nested_resolutions_equal_rasterising_each_one_directly():
    """The coarser counts come from integer division of the finest pixels; that is
    only legitimate if it gives what a fresh rasterisation at that size would."""
    import random

    rng = random.Random(1)
    poly = [(0.0, 0.0)]
    for _ in range(60):
        poly.append((poly[-1][0] + rng.uniform(-150, 200), poly[-1][1] + rng.uniform(-150, 200)))
    for size, n in box_counts(poly, FragmentationConfig()):
        assert box_counts(poly, FragmentationConfig(pixel_m=size))[0][1] == n


def test_a_diagonal_through_pixel_corners_does_not_count_the_corner_pixels():
    """(0,0)→(3,3) on a unit grid passes through two corners and occupies exactly
    the three diagonal pixels — not the off-diagonal ones it only touches."""
    cfg = FragmentationConfig(pixel_m=1.0)
    assert box_counts([(0.0, 0.0), (3.0, 3.0)], cfg)[0] == (1.0, 3)
    assert box_counts([(0.0, 3.0), (3.0, 0.0)], cfg)[0] == (1.0, 3)


def test_a_long_straight_path_scores_just_under_one_by_box_counting():
    """Not exactly 1.0: a line occupies ``⌊L/r⌋ + 1`` pixels, not ``L/r``, and the
    ``+ 1`` weighs more at the coarse end. The bias shrinks as the path grows."""
    d_5km = box_counting_dimension(_line(5000.0), FragmentationConfig())
    d_20km = box_counting_dimension(_line(20000.0), FragmentationConfig())
    assert 0.95 < d_5km < d_20km < 1.0


def test_as_written_is_exactly_one_above_the_consistent_convention():
    """``1 - slope`` against ``-slope``: the paper's equation applied literally to
    pixel counts puts a straight line at 2."""
    poly = _line(5000.0)
    d = box_counting_dimension(poly, FragmentationConfig())
    assert box_counting_dimension(poly, FragmentationConfig(as_written=True)) == pytest.approx(d + 1.0, abs=1e-12)


def test_saturated_resolutions_confound_d_with_extent_unless_dropped():
    """A 300 m straight path: with the two single-pixel resolutions in the fit
    ``D`` falls to 0.80; without them it is 0.97. Same shape, the only difference
    is how much smaller the path is than the coarsest pixel."""
    poly = _line(300.0)
    assert box_counting_dimension(poly, FragmentationConfig()) == pytest.approx(0.969, abs=1e-3)
    assert box_counting_dimension(poly, FragmentationConfig(drop_saturated=False)) == pytest.approx(0.802, abs=1e-3)


def test_a_path_inside_too_few_pixels_reports_nan_with_a_reason():
    """10 m: only 0.7 m and 3.5 m see more than one pixel — two points, and a
    two-point fit is not a measurement (invariant #2)."""
    shape = measure_path(_chain(1), _straight(1, step=10.0), FragmentationConfig())
    assert shape is not None
    assert math.isnan(shape.fragmentation)
    assert shape.reason == "saturated"


def test_box_counting_measures_the_short_paths_the_divider_cannot():
    """Four segments: below the divider's eight-segment floor, an ordinary path
    for box counting."""
    shape = measure_path(_chain(4), _straight(4), FragmentationConfig())
    assert shape.measured
    assert shape.reason is None
    assert math.isnan(measure_path(_chain(4), _straight(4), FragmentationConfig(method="divider")).fragmentation)


def test_koch_curve_scores_above_a_straight_line_of_the_same_span_by_box_counting():
    koch = box_counting_dimension(koch_polyline(depth=5), FragmentationConfig())
    straight = box_counting_dimension(_line(2000.0), FragmentationConfig())
    assert straight < 1.0 < koch < KOCH_D


def test_edge_geometry_is_traced_between_the_end_nodes():
    """A curved street is one edge with interior vertices. The measure follows the
    curve, while the street-segment count stays the node count."""
    node_xy = {0: (0.0, 0.0), 1: (200.0, 0.0), 2: (300.0, 0.0)}
    bend = [(0.0, 0.0), (100.0, 80.0), (200.0, 0.0)]

    def edge_xy(edge):
        return bend if edge == (0, 1, 0) else None

    path = [(0, 1, 0), (1, 2, 0)]
    assert path_polyline(path, node_xy, edge_xy) == [(0.0, 0.0), (100.0, 80.0), (200.0, 0.0), (300.0, 0.0)]

    cfg = FragmentationConfig()
    chord = measure_path(path, node_xy, cfg)
    traced = measure_path(path, node_xy, cfg, edge_xy)
    assert chord.length_m == pytest.approx(300.0)
    assert traced.length_m == pytest.approx(2 * math.hypot(100.0, 80.0) + 100.0)
    assert traced.n_segments == chord.n_segments == 2
    assert traced.extent_m == chord.extent_m
    # ... and the switch turns the tracing off
    off = measure_path(path, node_xy, FragmentationConfig(use_edge_geometry=False), edge_xy)
    assert off.length_m == pytest.approx(300.0)


def test_config_rejects_a_pixel_ladder_that_cannot_produce_a_slope():
    with pytest.raises(ValueError, match="pixel_m"):
        FragmentationConfig(pixel_m=0.0)
    with pytest.raises(ValueError, match="pixel_factor"):
        FragmentationConfig(pixel_factor=1)
    with pytest.raises(ValueError, match="n_pixel_scales"):
        FragmentationConfig(n_pixel_scales=2)


def test_the_accumulator_books_saturated_paths_apart_from_measured_ones():
    settings = _settings()
    node_xy = {**_straight(20), 100: (0.0, 500.0), 101: (10.0, 500.0)}
    acc = PathAccumulator(settings, node_xy)
    acc.add(_trajectory(_chain(3)), _vector(low=1.0, high=1.0))  # 300 m: measured now
    acc.add(_trajectory([(100, 101, 0)]), _vector(low=1.0, high=1.0))  # 10 m: saturated

    stats = acc.stats()
    assert stats["n_measured"] == 1
    assert stats["n_saturated"] == 1
    assert stats["n_too_short"] == 0
    assert stats["params"]["method"] == "box_counting"
    assert stats["params"]["pixel_m"] == 0.7


def test_the_edge_resolver_orients_a_stored_geometry_along_the_traversal():
    """A two-way street's reverse edge (and any edge of an ``undirected`` graph)
    can carry its geometry v→u; the path must still be traced u→v."""
    from types import SimpleNamespace

    import networkx as nx
    from shapely.geometry import LineString

    from segx.orchestrator import _edge_coordinates

    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=200.0, y=0.0)
    graph.add_node(3, x=300.0, y=0.0)
    stored = LineString([(200.0, 0.0), (100.0, 80.0), (0.0, 0.0)])  # runs 2→1
    graph.add_edge(1, 2, key=0, geometry=stored)
    graph.add_edge(2, 3, key=0)  # a straight block: no geometry of its own
    state = SimpleNamespace(graphs=SimpleNamespace(modes=lambda: ["walk"], get=lambda _m: graph))
    node_xy = {1: (0.0, 0.0), 2: (200.0, 0.0), 3: (300.0, 0.0)}

    resolve = _edge_coordinates(state, node_xy)
    assert resolve((1, 2, 0)) == [(0.0, 0.0), (100.0, 80.0), (200.0, 0.0)]
    assert resolve((2, 3, 0)) is None
    assert resolve((9, 9, 0)) is None


def test_sharing_the_per_edge_rasterisation_changes_no_number() -> None:
    """A path's pixels are the union of its edges' pixels, so each edge is
    rasterised once and shared (`EdgeShapeCache`). That is an optimisation, not a
    method: measured with a shared cache, with none, or by rasterising the whole
    polyline at once, every path must report the *same* floats — including a
    curved edge traced along its geometry and a path whose consecutive edges do
    not share a node (what `expand_transit_rides` leaves behind once the access
    edges are dropped)."""
    from segx.metrics.fragmentation import EdgeShapeCache, box_counting_dimension, path_polyline, polyline_length

    cfg = Settings().metrics.fragmentation
    node_xy = {
        1: (500000.3, 4420000.7),
        2: (500310.9, 4420012.2),
        3: (500322.4, 4420405.1),
        4: (500811.6, 4420398.8),
        5: (500905.2, 4421377.0),
        6: (501990.5, 4421420.3),
    }
    curve = {(2, 3, 0): [node_xy[2], (500360.0, 4420150.0), (500290.0, 4420290.0), node_xy[3]]}

    def edge_xy(edge):
        return curve.get(edge)

    paths = [
        [(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0)],
        [(2, 3, 0), (3, 4, 0)],  # shares edges with the first
        [(1, 2, 0), (4, 5, 0), (5, 6, 0)],  # 2 -> 4 is a jump: no shared node
        [(6, 5, 0), (5, 4, 0), (4, 3, 0)],  # the reverse direction
    ]
    shared = EdgeShapeCache()
    for path in paths * 2:  # second pass is served entirely from the cache
        with_cache = measure_path(path, node_xy, cfg, edge_xy, shared)
        without = measure_path(path, node_xy, cfg, edge_xy)
        traced = path_polyline(path, node_xy, edge_xy)
        assert with_cache.fragmentation == without.fragmentation == box_counting_dimension(traced, cfg)
        assert with_cache.length_m == without.length_m == polyline_length(traced)
        assert (with_cache.n_segments, with_cache.extent_m) == (without.n_segments, without.extent_m)
    assert 0 < len(shared) <= sum(len(p) for p in paths)  # bounded by edges, not by calls
