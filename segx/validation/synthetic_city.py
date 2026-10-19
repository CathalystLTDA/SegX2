r"""A tiny synthetic city with **hand-computed ground truth** — the engine's
gating fixture.

Layout: a 4×4 Manhattan grid (16 nodes, all adjacent horizontal/vertical edges
present), every edge ``EDGE_M`` metres long. Node id is ``r * GRID_N + c`` for a
node at grid column ``c`` and row ``r`` (both 0..3); its coordinates are
``(c * EDGE_M, r * EDGE_M)`` in a metric CRS::

    r=3  12 — 13 — 14 — 15
         |    |    |    |
    r=2   8 —  9 — 10 — 11
         |    |    |    |
    r=1   4 —  5 —  6 —  7
         |    |    |    |
    r=0   0 —  1 —  2 —  3
         c=0  c=1  c=2  c=3

Because the grid is complete, the shortest-path length between any two nodes
equals their Manhattan distance × ``EDGE_M`` — which makes the nearest-POI and
catchment ground truth below derivable by hand and stable for the P3 routing
tests.

POIs (note: **two** ``park`` POIs, so the dual binding — nearest vs catchment —
is genuinely exercised):

============  ==========  =======================
POI id        category    node (grid c,r)
============  ==========  =======================
``park_w``    park        12  (0, 3)
``park_e``    park         7  (3, 1)
``clinic``    health_center 15  (3, 3)
``stop_s``    bus_stop      1  (1, 0)
============  ==========  =======================

Origins: eight, each carrying a demographic vector over the classes in
``DEMOGRAPHIC_CLASSES``. The per-origin entropies, the nearest-POI assignments,
and the 200 m catchment sets are all hand-computed and asserted in
``tests/unit/test_synthetic_city.py``.

What this module provides at **P0**: the graph, the POIs, the origin
demographics, and the ground-truth constants. The end-to-end assertions that
need routing/aggregation (per-edge demographic sums, segment entropy) live in the
same test file behind ``@pytest.mark.integration`` and are switched on in P3/P4.
"""

from __future__ import annotations

import math

import networkx as nx

# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #

GRID_N: int = 4  # nodes per side → 16 nodes
EDGE_M: float = 100.0  # length of every grid edge, metres
WALK_SPEED_KMH: float = 5.0
#: minutes to traverse one edge on foot, = EDGE_M / (5000 m / 60 min)
EDGE_WALK_MIN: float = EDGE_M / (WALK_SPEED_KMH * 1000.0 / 60.0)  # 1.2 min for 100 m @ 5 km/h


def node_id(col: int, row: int) -> int:
    """Grid (col, row) → node id."""
    if not (0 <= col < GRID_N and 0 <= row < GRID_N):
        raise ValueError(f"({col}, {row}) outside the {GRID_N}x{GRID_N} grid")
    return row * GRID_N + col


def node_colrow(nid: int) -> tuple[int, int]:
    """Node id → grid (col, row)."""
    return nid % GRID_N, nid // GRID_N


def manhattan_steps(a: int, b: int) -> int:
    """Manhattan distance between two node ids, in grid steps (= shortest path
    length on this complete grid, in units of EDGE_M)."""
    (ca, ra), (cb, rb) = node_colrow(a), node_colrow(b)
    return abs(ca - cb) + abs(ra - rb)


def build_grid_graph() -> nx.Graph:
    """Build the 4×4 grid as an undirected NetworkX graph.

    Node attrs: ``x``, ``y`` (metres). Edge attrs: ``length`` (metres),
    ``travel_time_min`` (walk). This is a deliberately simplified stand-in for a
    real OSMnx ``MultiDiGraph``; the routing layer will adapt it in P3.
    """
    g = nx.Graph()
    for row in range(GRID_N):
        for col in range(GRID_N):
            nid = node_id(col, row)
            g.add_node(nid, x=col * EDGE_M, y=row * EDGE_M, col=col, row=row)
    for row in range(GRID_N):
        for col in range(GRID_N):
            nid = node_id(col, row)
            if col + 1 < GRID_N:
                g.add_edge(nid, node_id(col + 1, row), length=EDGE_M, travel_time_min=EDGE_WALK_MIN)
            if row + 1 < GRID_N:
                g.add_edge(nid, node_id(col, row + 1), length=EDGE_M, travel_time_min=EDGE_WALK_MIN)
    return g


# --------------------------------------------------------------------------- #
# Separately-mapped sidewalks (P7.2)                                           #
# --------------------------------------------------------------------------- #

#: How far a sidewalk is drawn from the roadway centreline it belongs to. Real
#: cities land near this: measured across Philadelphia's Center City, a footway
#: sits a **median of 4.3 m** from its roadway.
SIDEWALK_OFFSET_M: float = 5.0
#: A path with no roadway anywhere near it — a park trail. It must resolve to a
#: corridor of its own; that is a real street space, not a matching failure.
PARK_PATH_OFFSET_M: float = 100.0
#: First node id of the sidewalk layer. Kept clear of the grid's ``0..15`` so a
#: sidewalk node can never be mistaken for a roadway node.
SIDEWALK_NODE_BASE: int = 1000


def build_sidewalk_graph() -> nx.MultiDiGraph:
    """The pedestrian layer as OSM actually maps it in a well-surveyed city:
    sidewalks are **their own ways**, not attributes of the roadway. P7.2.

    This is the case that breaks every id-based reconciliation. OSMnx's ``walk``
    filter *excludes* a road tagged ``sidewalk=separate``, so the pedestrian and
    the car end up on different OSM ways, with different nodes and different
    ``osmid``. Measured on Philadelphia, only 201 of 533 drive ways appear in the
    walk graph at all — so ``walk ∩ drive`` was 6 segments out of 448.

    Three edges, each pinning one behaviour of
    :func:`segx.network.street_space.resolve_street_space`:

    * **sidewalk** — parallel to the grid row ``r=0``, offset
      :data:`SIDEWALK_OFFSET_M`. Shares the roadway's corridor.
    * **crossing** — perpendicular, its midpoint *well inside* the tolerance.
      Rejected by the bearing test alone; without it every corner would drag its
      cross street into the corridor.
    * **park path** — parallel but :data:`PARK_PATH_OFFSET_M` away. Its own
      corridor.

    A ``MultiDiGraph`` (unlike :func:`build_grid_graph`) because that is what the
    edge-id contract ``(u, v, key)`` assumes.
    """
    g = nx.MultiDiGraph()
    g.graph["crs"] = SYNTHETIC_CRS
    coords = {
        SIDEWALK_NODE_BASE + 0: (0.0, SIDEWALK_OFFSET_M),
        SIDEWALK_NODE_BASE + 1: (EDGE_M, SIDEWALK_OFFSET_M),
        SIDEWALK_NODE_BASE + 2: (EDGE_M, -SIDEWALK_OFFSET_M),
        SIDEWALK_NODE_BASE + 3: (0.0, PARK_PATH_OFFSET_M),
        SIDEWALK_NODE_BASE + 4: (EDGE_M, PARK_PATH_OFFSET_M),
    }
    for nid, (x, y) in coords.items():
        g.add_node(nid, x=x, y=y)
    for u, v in (
        (SIDEWALK_NODE_BASE + 0, SIDEWALK_NODE_BASE + 1),  # sidewalk along r=0
        (SIDEWALK_NODE_BASE + 1, SIDEWALK_NODE_BASE + 2),  # crossing
        (SIDEWALK_NODE_BASE + 3, SIDEWALK_NODE_BASE + 4),  # park path
    ):
        (x0, y0), (x1, y1) = coords[u], coords[v]
        length = math.hypot(x1 - x0, y1 - y0)
        g.add_edge(
            u, v, key=0, length=length, travel_time_min=length / (WALK_SPEED_KMH * 1000.0 / 60.0), highway="footway"
        )
    return g


#: The sidewalk edge that must share a corridor with grid edge ``(0, 1, 0)``.
SIDEWALK_EDGE: tuple[int, int, int] = (SIDEWALK_NODE_BASE + 0, SIDEWALK_NODE_BASE + 1, 0)
#: The crossing that must **not** — perpendicular, though only 5 m away.
CROSSING_EDGE: tuple[int, int, int] = (SIDEWALK_NODE_BASE + 1, SIDEWALK_NODE_BASE + 2, 0)
#: The park path that must not — parallel, but 100 m away.
PARK_PATH_EDGE: tuple[int, int, int] = (SIDEWALK_NODE_BASE + 3, SIDEWALK_NODE_BASE + 4, 0)


# --------------------------------------------------------------------------- #
# POIs                                                                         #
# --------------------------------------------------------------------------- #

#: ``poi_id -> (category, node_id)``
POIS: dict[str, tuple[str, int]] = {
    "park_w": ("park", node_id(0, 3)),  # node 12
    "park_e": ("park", node_id(3, 1)),  # node 7
    "clinic": ("health_center", node_id(3, 3)),  # node 15
    "stop_s": ("bus_stop", node_id(1, 0)),  # node 1
}

POI_CATEGORIES: tuple[str, ...] = ("park", "health_center", "bus_stop")


# --------------------------------------------------------------------------- #
# Origins & demographics                                                       #
# --------------------------------------------------------------------------- #

#: the discrete classes the synthetic demographic attribute ranges over
DEMOGRAPHIC_CLASSES: tuple[str, ...] = ("low", "high")

#: ``origin_id -> node_id``
ORIGIN_NODES: dict[str, int] = {
    "o1": node_id(1, 1),  # 5
    "o2": node_id(2, 2),  # 10
    "o3": node_id(0, 1),  # 4
    "o4": node_id(2, 1),  # 6
    "o5": node_id(1, 2),  # 9
    "o6": node_id(2, 3),  # 14
    "o7": node_id(0, 2),  # 8
    "o8": node_id(1, 3),  # 13
}

#: ``origin_id -> {class -> mass}`` (need not sum to 1; treated as weights)
ORIGIN_DEMOGRAPHICS: dict[str, dict[str, float]] = {
    "o1": {"low": 4.0, "high": 0.0},  # entropy 0.0
    "o2": {"low": 2.0, "high": 2.0},  # entropy 1.0
    "o3": {"low": 3.0, "high": 1.0},  # entropy 0.811278...
    "o4": {"low": 1.0, "high": 3.0},  # entropy 0.811278...
    "o5": {"low": 1.0, "high": 1.0},  # entropy 1.0
    "o6": {"low": 0.0, "high": 5.0},  # entropy 0.0
    "o7": {"low": 2.0, "high": 1.0},  # entropy 0.918295...
    "o8": {"low": 1.0, "high": 1.0},  # entropy 1.0
}


# --------------------------------------------------------------------------- #
# Hand-computed ground truth                                                   #
# --------------------------------------------------------------------------- #

#: Named distributions and their Shannon entropy (base 2, bits), worked out by
#: hand — see the module docstring of ``segx.metrics.entropy``. Used to pin
#: ``shannon_entropy`` directly, independent of any routing.
EXPECTED_ENTROPIES_BITS: dict[str, float] = {
    "single_class": 0.0,  # {4, 0}
    "uniform_2": 1.0,  # {2, 2}
    "skew_3_1": 0.8112781244591328,  # {3, 1}: -(.75 log2 .75 + .25 log2 .25)
    "skew_2_1": 0.9182958340544896,  # {2, 1}: -(2/3 log2 2/3 + 1/3 log2 1/3)
    "uniform_3": math.log2(3),  # {1, 1, 1} = 1.5849625007...
    "uniform_4": 2.0,  # {1, 1, 1, 1}
    "empty": math.nan,  # {} — undefined, NOT zero
}

#: Per-origin demographic entropy (bits), derived from ``ORIGIN_DEMOGRAPHICS``.
EXPECTED_ORIGIN_ENTROPY_BITS: dict[str, float] = {
    "o1": 0.0,
    "o2": 1.0,
    "o3": 0.8112781244591328,
    "o4": 0.8112781244591328,
    "o5": 1.0,
    "o6": 0.0,
    "o7": 0.9182958340544896,
    "o8": 1.0,
}

#: Nearest POI of each category per origin (Manhattan distance on the complete
#: grid), as ``origin_id -> {category -> (poi_id, distance_steps)}``. Multiply
#: ``distance_steps`` by ``EDGE_M`` for metres.
NEAREST_POI_GROUND_TRUTH: dict[str, dict[str, tuple[str, int]]] = {
    "o1": {"park": ("park_e", 2), "health_center": ("clinic", 4), "bus_stop": ("stop_s", 1)},
    "o2": {"park": ("park_e", 2), "health_center": ("clinic", 2), "bus_stop": ("stop_s", 3)},
    "o3": {"park": ("park_w", 2), "health_center": ("clinic", 5), "bus_stop": ("stop_s", 2)},
    "o4": {"park": ("park_e", 1), "health_center": ("clinic", 3), "bus_stop": ("stop_s", 2)},
    "o5": {"park": ("park_w", 2), "health_center": ("clinic", 3), "bus_stop": ("stop_s", 2)},
    "o6": {"park": ("park_w", 2), "health_center": ("clinic", 1), "bus_stop": ("stop_s", 4)},
    "o7": {"park": ("park_w", 1), "health_center": ("clinic", 4), "bus_stop": ("stop_s", 3)},
    "o8": {"park": ("park_w", 1), "health_center": ("clinic", 2), "bus_stop": ("stop_s", 3)},
}

#: Catchment budget used for the ground truth below, in grid steps (200 m).
CATCHMENT_BUDGET_STEPS: int = 2

#: All POIs of each category reachable within ``CATCHMENT_BUDGET_STEPS`` per
#: origin, as ``origin_id -> {category -> frozenset[poi_id]}``. An empty set is a
#: *real* result (the resident reaches no POI of that category in budget) — note
#: e.g. ``o7`` reaches no ``bus_stop`` or ``health_center`` within 200 m, and
#: ``o2`` reaches no ``bus_stop``. Tightening the budget to 1 step shrinks these
#: further (e.g. ``o4`` loses its ``bus_stop``), exercising budget sensitivity.
CATCHMENT_GROUND_TRUTH: dict[str, dict[str, frozenset[str]]] = {
    "o1": {"park": frozenset({"park_e"}), "health_center": frozenset(), "bus_stop": frozenset({"stop_s"})},
    "o2": {"park": frozenset({"park_e"}), "health_center": frozenset({"clinic"}), "bus_stop": frozenset()},
    "o3": {"park": frozenset({"park_w"}), "health_center": frozenset(), "bus_stop": frozenset({"stop_s"})},
    "o4": {"park": frozenset({"park_e"}), "health_center": frozenset(), "bus_stop": frozenset({"stop_s"})},
    "o5": {"park": frozenset({"park_w"}), "health_center": frozenset(), "bus_stop": frozenset({"stop_s"})},
    "o6": {"park": frozenset({"park_w"}), "health_center": frozenset({"clinic"}), "bus_stop": frozenset()},
    "o7": {"park": frozenset({"park_w"}), "health_center": frozenset(), "bus_stop": frozenset()},
    "o8": {"park": frozenset({"park_w"}), "health_center": frozenset({"clinic"}), "bus_stop": frozenset()},
}


# --------------------------------------------------------------------------- #
# Public transport (P7) — hand-computable bus + metro lines                    #
# --------------------------------------------------------------------------- #

#: The bus line runs along the bottom row, ``0 - 1 - 2 - 3`` (``r=0``), so it
#: parallels three grid edges and can be compared directly against walking them.
TRANSIT_BUS_STOP_NODES: tuple[int, ...] = (0, 1, 2, 3)

#: The metro runs along the top row, ``12 - 13 - 14 - 15`` (``r=3``). Being rail,
#: its mass lands on the ``rail_line`` layer instead of the streets below it.
TRANSIT_METRO_STOP_NODES: tuple[int, ...] = (12, 13, 14, 15)

#: In-vehicle minutes per stop-to-stop hop, for both lines.
TRANSIT_HOP_MIN: float = 0.5

#: Boarding penalty used by the transit ground truth below.
TRANSIT_BOARD_PENALTY_MIN: float = 1.0

#: Two headways chosen to straddle walking, so the *same* fixture proves the
#: frequency model both ways round (the shortest path should take the bus only
#: when the bus is genuinely faster).
TRANSIT_SLOW_HEADWAY_MIN: float = 10.0
TRANSIT_FAST_HEADWAY_MIN: float = 2.0

#: Walking 0 -> 3 along the bottom row: 3 edges x EDGE_WALK_MIN = 3.6 min.
WALK_0_TO_3_MIN: float = 3 * EDGE_WALK_MIN

#: Riding 0 -> 3: wait (headway/2) + board penalty + 3 hops x TRANSIT_HOP_MIN.
#: Stops sit exactly on grid nodes, so the connectors cost nothing.
#:   slow: 5.0 + 1.0 + 1.5 = 7.5  -> walking (3.6) wins
#:   fast: 1.0 + 1.0 + 1.5 = 3.5  -> the bus wins
BUS_0_TO_3_SLOW_MIN: float = TRANSIT_SLOW_HEADWAY_MIN / 2 + TRANSIT_BOARD_PENALTY_MIN + 3 * TRANSIT_HOP_MIN
BUS_0_TO_3_FAST_MIN: float = TRANSIT_FAST_HEADWAY_MIN / 2 + TRANSIT_BOARD_PENALTY_MIN + 3 * TRANSIT_HOP_MIN

#: A metric CRS for the fixture. The grid's coordinates are already metres, so
#: this only satisfies the engine's "declare your CRS" contract (invariant #1).
SYNTHETIC_CRS: str = "EPSG:32618"

#: The headway-measurement window the synthetic feed declares. Named (rather than
#: inlined) because ``n_trips`` is *derived* from it: ``n_trips × headway_min``
#: must equal the window length, or the fixture would assert a relationship the
#: engine does not actually maintain.
SYNTHETIC_SERVICE_WINDOW: tuple[str, str] = ("06:00", "09:00")


def build_transit_feed(
    headway_min: float = TRANSIT_FAST_HEADWAY_MIN,
    *,
    include_metro: bool = True,
    working_crs: str = SYNTHETIC_CRS,
):
    """A :class:`~segx.io.gtfs.LoadedGTFS` over the grid: one bus + one metro line.

    Stops sit **exactly on grid nodes**, which is what makes the ride-vs-walk
    ground truth above exact: the platform connectors have zero length and so cost
    nothing, leaving wait + board penalty + in-vehicle time as the whole fare.

    Imports geopandas/pandas *inside* the function on purpose — the rest of this
    module stays importable by the geopandas-free routing unit tests.
    """
    import geopandas as gpd
    import pandas as pd

    from segx.config import parse_window
    from segx.io.gtfs import LoadedGTFS

    start, end = parse_window(SYNTHETIC_SERVICE_WINDOW)
    window_minutes = float(end - start)

    lines: list[tuple[str, str, int, tuple[int, ...]]] = [("BUS", "Bottom Row Bus", 3, TRANSIT_BUS_STOP_NODES)]
    if include_metro:
        lines.append(("MET", "Top Row Metro", 1, TRANSIT_METRO_STOP_NODES))

    stop_ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    pattern_rows: list[dict] = []
    pattern_stop_rows: list[dict] = []
    for prefix, name, route_type, nodes in lines:
        pattern_id = f"{prefix}::0"
        pattern_rows.append(
            {
                "pattern_id": pattern_id,
                "route_id": prefix,
                "direction_id": 0,
                "route_type": route_type,
                "route_name": name,
                "headway_min": float(headway_min),
                "n_trips": window_minutes / float(headway_min),
            }
        )
        for seq, nid in enumerate(nodes):
            col, row = node_colrow(nid)
            stop_id = f"{prefix}{nid}"
            stop_ids.append(stop_id)
            xs.append(col * EDGE_M)
            ys.append(row * EDGE_M)
            pattern_stop_rows.append(
                {
                    "pattern_id": pattern_id,
                    "seq": seq,
                    "stop_id": stop_id,
                    "travel_time_min": 0.0 if seq == 0 else TRANSIT_HOP_MIN,
                }
            )

    stops = gpd.GeoDataFrame(
        {"stop_id": stop_ids, "stop_name": stop_ids},
        geometry=gpd.points_from_xy(xs, ys),
        crs=working_crs,
    )
    return LoadedGTFS(
        stops=stops,
        patterns=pd.DataFrame.from_records(pattern_rows),
        pattern_stops=pd.DataFrame.from_records(pattern_stop_rows),
        source="gtfs",
        feed_hash="synthetic",
        service_day="weekday",
        service_window=SYNTHETIC_SERVICE_WINDOW,
        crs=working_crs,
    )


def transit_stop_id(node: int, *, metro: bool = False) -> str:
    """The fixture's stop id for a grid node (``"BUS0"`` / ``"MET12"``)."""
    return f"{'MET' if metro else 'BUS'}{node}"


__all__ = [
    "BUS_0_TO_3_FAST_MIN",
    "BUS_0_TO_3_SLOW_MIN",
    "CATCHMENT_BUDGET_STEPS",
    "CATCHMENT_GROUND_TRUTH",
    "DEMOGRAPHIC_CLASSES",
    "EDGE_M",
    "EDGE_WALK_MIN",
    "EXPECTED_ENTROPIES_BITS",
    "EXPECTED_ORIGIN_ENTROPY_BITS",
    "GRID_N",
    "NEAREST_POI_GROUND_TRUTH",
    "ORIGIN_DEMOGRAPHICS",
    "ORIGIN_NODES",
    "POIS",
    "POI_CATEGORIES",
    "SYNTHETIC_CRS",
    "SYNTHETIC_SERVICE_WINDOW",
    "TRANSIT_BOARD_PENALTY_MIN",
    "TRANSIT_BUS_STOP_NODES",
    "TRANSIT_FAST_HEADWAY_MIN",
    "TRANSIT_HOP_MIN",
    "TRANSIT_METRO_STOP_NODES",
    "TRANSIT_SLOW_HEADWAY_MIN",
    "WALK_0_TO_3_MIN",
    "WALK_SPEED_KMH",
    "build_grid_graph",
    "build_transit_feed",
    "manhattan_steps",
    "node_colrow",
    "node_id",
    "transit_stop_id",
]
