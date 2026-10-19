"""Configuration for the SegX engine.

Everything demographic or column-related is a *runtime input*, not a constant.
One resolved :class:`Settings` object is built by layering, last wins:

1. built-in defaults (this module),
2. a project config file (``segx.toml``, passed via ``--config``),
3. CLI flags / direct keyword overrides.

The future frontend becomes a fourth producer of this same object; it gets no
privileged path into the engine, so terminal runs can never drift from it.

At P0 only the *shape* of the config and its defaults exist. Consumers
(``io/``, ``network/``, ``routing/`` …) are stubbed and will read names, bins
and budgets from the resolved object — never from string literals.
"""

from __future__ import annotations

import tomllib
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from segx.errors import ConfigKeyError, ConfigValueError

# --------------------------------------------------------------------------- #
# CRS                                                                          #
# --------------------------------------------------------------------------- #


class CRSConfig(BaseModel):
    """Coordinate-reference handling.

    Geometry may *enter* the system in any CRS; it is reprojected to
    ``working_crs`` immediately on load and nothing downstream touches lat/lon.
    If ``working_crs`` is ``None`` and ``auto_utm`` is set, the engine derives
    the appropriate UTM zone from the study-area boundary at runtime.
    """

    geographic_crs: str = "EPSG:4326"
    working_crs: str | None = None
    auto_utm: bool = True


# --------------------------------------------------------------------------- #
# Demographic schema                                                           #
# --------------------------------------------------------------------------- #

AttributeKind = Literal["categorical", "binned", "numeric"]


class DemographicAttribute(BaseModel):
    """One attribute that participates in diversity/segregation accounting.

    ``categorical`` — discrete classes (e.g. race); ``categories`` lists them.
    ``binned`` — a numeric quantity discretised into ranges; ``bins`` gives the
    (ascending) bin edges, producing ``len(bins) - 1`` classes.
    ``numeric`` — kept as a raw scalar (used by some metrics, not by entropy).
    """

    name: str
    kind: AttributeKind
    categories: list[str] | None = None
    bins: list[float] | None = None
    source_column: str | None = None  # defaults to ``name`` at load time
    # Explicit per-class source-column names, overriding the derived
    # ``f"{source_column}_{slug}"`` / ``f"{source_column}_b{i}"`` convention. Use
    # this when the input's class columns don't follow that shape — e.g. LODES OD
    # data whose income classes live in ``SE01``/``SE02``/``SE03``. Must list one
    # column per class, in class order (``len(categories)`` / ``len(bins) - 1`` /
    # ``1`` for numeric). See ``io/demographics.class_columns_for``.
    class_columns: list[str] | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> DemographicAttribute:
        if self.kind == "categorical" and not self.categories:
            raise ValueError(f"attribute {self.name!r}: categorical requires non-empty `categories`")
        if self.kind == "binned":
            if not self.bins or len(self.bins) < 2:
                raise ValueError(f"attribute {self.name!r}: binned requires >= 2 `bins` edges")
            if any(self.bins[i] >= self.bins[i + 1] for i in range(len(self.bins) - 1)):
                raise ValueError(f"attribute {self.name!r}: `bins` must be strictly ascending")
        if self.source_column is None:
            object.__setattr__(self, "source_column", self.name)
        if self.class_columns is not None:
            n_expected = len(self.class_labels)
            if len(self.class_columns) != n_expected:
                raise ValueError(
                    f"attribute {self.name!r}: `class_columns` must list {n_expected} column(s) "
                    f"(one per class), got {len(self.class_columns)}"
                )
        return self

    @property
    def class_labels(self) -> list[str]:
        """The discrete classes this attribute contributes to entropy."""
        if self.kind == "categorical":
            assert self.categories is not None
            return list(self.categories)
        if self.kind == "binned":
            assert self.bins is not None
            return [f"[{lo:g}, {hi:g})" for lo, hi in zip(self.bins, self.bins[1:], strict=False)]
        return [self.name]


class DemographicSchema(BaseModel):
    """The full set of demographic attributes plus the polygon-level columns.

    ``polygon_id_column`` / ``population_column`` are the defaults
    :mod:`segx.io.demographics` looks for; override per dataset.
    """

    attributes: list[DemographicAttribute]
    polygon_id_column: str = "id"
    population_column: str = "pop"
    # Where each origin's demographic vector is *read from*:
    #   "demographics" (default) — the per-class count columns live on the
    #       demographics file; origins inherit their polygon/point row's values.
    #   "od" — the class columns live on the **OD file** itself (e.g. LODES
    #       ``SE01``/``SE02``/``SE03`` worker counts per home→work flow). The
    #       demographics file is then used only as a geometry/ID lookup (it needs
    #       just ``polygon_id_column`` + geometry), and the vectors are summed per
    #       distinct origin from the OD rows. See ``population/attributes.py``.
    attribute_source: Literal["demographics", "od"] = "demographics"

    @classmethod
    def ibge_default(cls) -> DemographicSchema:
        """Sensible IBGE-flavoured default. Bin edges are placeholders; override
        them in configuration."""
        return cls(
            attributes=[
                DemographicAttribute(
                    name="income",
                    kind="binned",
                    # monthly household income, BRL — PLACEHOLDER edges
                    bins=[0.0, 1320.0, 2640.0, 5280.0, 10560.0, 21120.0, float("inf")],
                ),
                DemographicAttribute(
                    name="race",
                    kind="categorical",
                    categories=["branca", "preta", "parda", "amarela", "indigena"],
                ),
                DemographicAttribute(
                    name="age",
                    kind="binned",
                    bins=[0.0, 15.0, 30.0, 45.0, 60.0, 75.0, float("inf")],  # PLACEHOLDER edges
                ),
            ],
        )

    @property
    def attribute_names(self) -> list[str]:
        return [a.name for a in self.attributes]

    def get(self, name: str) -> DemographicAttribute:
        for a in self.attributes:
            if a.name == name:
                return a
        raise ConfigKeyError(
            f"no demographic attribute named {name!r}",
            details={"name": name, "available": self.attribute_names},
        )


# --------------------------------------------------------------------------- #
# OD-pair column mapping                                                       #
# --------------------------------------------------------------------------- #


class ODColumnMapping(BaseModel):
    """Declares which columns in the user's OD file mean what.

    The OD file has *no fixed schema* — only this mapping does. For CSV the
    location of an endpoint may be a WKT string or a lon/lat pair; for GeoPackage
    it is a geometry column; either may instead be an address column to geocode.
    Columns that are absent fall back to the ``default_*`` values below.
    """

    origin_id: str = "origin_id"
    #: Destination identity. Two flows ending at the same *place* must collapse to
    #: one destination or the OD destination-spread analysis
    #: (:mod:`segx.analysis.od_spread`) degenerates into a row count. Resolution
    #: order mirrors ``origin_id``: this column, else ``dest_id_ref`` (the id that
    #: already placed the endpoint, e.g. LODES ``wGEOID``), else the destination
    #: coordinates snapped to ``dest_id_precision_m``.
    dest_id: str = "dest_id"

    # origin endpoint — first present of: geometry col, wkt col, lon+lat,
    # id-reference (an ID joined to the demographics geometry), address
    origin_geom: str = "origin_geom"
    origin_wkt: str = "origin_wkt"
    origin_lon: str = "origin_lon"
    origin_lat: str = "origin_lat"
    # Column whose values are IDs referencing the demographics ``polygon_id_column``
    # (e.g. a census-block GEOID); the endpoint is the representative point of the
    # matching geometry. This is how LODES-style ``hGEOID``/``wGEOID`` flow files
    # are placed — they carry no coordinates, only block IDs.
    origin_id_ref: str = "origin_id_ref"
    origin_address: str = "origin_address"

    # destination endpoint — same resolution order
    dest_geom: str = "dest_geom"
    dest_wkt: str = "dest_wkt"
    dest_lon: str = "dest_lon"
    dest_lat: str = "dest_lat"
    dest_id_ref: str = "dest_id_ref"
    dest_address: str = "dest_address"

    category: str = "category"
    mode: str = "mode"
    weight: str = "weight"

    #: Grid used to derive a destination identity from coordinates when no
    #: destination id column exists. In metres because the endpoints are already
    #: in the working CRS by then (invariant #1) — a degree-based fudge would mean
    #: a different distance at every latitude.
    dest_id_precision_m: float = 1.0

    # fallbacks when a column is missing entirely
    default_category: str = "commute"
    default_mode_by_category: dict[str, str] = Field(default_factory=lambda: {"commute": "car"})
    fallback_mode: str = "walk"
    default_weight: float = 1.0

    def mode_for(self, category: str) -> str:
        return self.default_mode_by_category.get(category, self.fallback_mode)


# --------------------------------------------------------------------------- #
# POI binding — dual setting (nearest + catchment)                             #
# --------------------------------------------------------------------------- #

# OSM tag presets per category. Values are OSMnx/QuackOSM-style tag filters
# (``key -> True`` means "any value"). Override or extend via config.
DEFAULT_POI_PRESETS: dict[str, dict[str, Any]] = {
    "parks": {"leisure": ["park", "garden", "nature_reserve"]},
    "supermarkets": {"shop": ["supermarket", "convenience", "greengrocer"]},
    "health_centers": {"amenity": ["clinic", "hospital", "doctors", "health_post"]},
    "bus_stops": {"highway": "bus_stop", "public_transport": ["platform", "stop_position"]},
    # P7: metro/rail stations as an access opportunity in their own right
    # ("which residents live within 30 min of a metro station?").
    "metro_stations": {"railway": "station", "station": "subway", "public_transport": "station"},
}


class NearestBindingConfig(BaseModel):
    """Strategy A — the single closest POI of each category per origin."""

    enabled: bool = True
    # ``None`` = no cap beyond network reachability; otherwise a hard ceiling.
    max_distance_m: float | None = None
    # ``None`` = every category in ``poi_presets``; otherwise a subset.
    categories: list[str] | None = None


class CatchmentBindingConfig(BaseModel):
    """Strategy B — *all* POIs of each category reachable within a cost budget.

    Default budget is 30 min walking @ 5 km/h. A distance budget and per-mode
    budgets are also accepted; if both a time and a distance budget are given,
    the more restrictive wins. ``weight_mode='unit'`` emits one unit-weight
    trajectory per reachable POI (each is a real opportunity); ``'inverse_count'``
    spreads weight ``1/k`` over the ``k`` reachable POIs of the category.
    """

    enabled: bool = True
    budget_minutes: float | None = 30.0
    walk_speed_kmh: float = 5.0
    distance_budget_m: float | None = None
    per_mode_budget_minutes: dict[str, float] = Field(default_factory=dict)
    weight_mode: Literal["unit", "inverse_count"] = "unit"
    categories: list[str] | None = None

    @model_validator(mode="after")
    def _need_a_budget(self) -> CatchmentBindingConfig:
        if self.enabled and self.budget_minutes is None and self.distance_budget_m is None:
            raise ValueError("catchment binding enabled but neither time nor distance budget set")
        return self


class POIBindingConfig(BaseModel):
    presets: dict[str, dict[str, Any]] = Field(default_factory=lambda: dict(DEFAULT_POI_PRESETS))
    nearest: NearestBindingConfig = Field(default_factory=NearestBindingConfig)
    catchment: CatchmentBindingConfig = Field(default_factory=CatchmentBindingConfig)

    def categories_for(self, which: Literal["nearest", "catchment"]) -> list[str]:
        cfg = self.nearest if which == "nearest" else self.catchment
        return list(cfg.categories) if cfg.categories is not None else list(self.presets)


# --------------------------------------------------------------------------- #
# Network                                                                      #
# --------------------------------------------------------------------------- #


class StreetSpaceConfig(BaseModel):
    """Canonical **street space** (corridor) resolution across modes. P7.2.

    Two modes only meet in a metric where they deposit on the same segment id,
    and in a real city they never do: ``walk`` and ``drive`` are separate OSM
    extracts, and where sidewalks are mapped as their own ways (``sidewalk=
    separate``, which the OSMnx walk filter *requires*) the pedestrian and the
    car are on **different geometries entirely** — different ways, different
    nodes, different ``osmid``. Measured on the Philadelphia study area: only
    201 of 533 ``drive`` ways appear in the ``walk`` graph at all.

    So the join cannot be an id remap. Instead every edge of every mode graph is
    matched **geometrically** to a corridor — the roadway block it belongs to —
    and ``cross_mode_exposure`` pools over that corridor instead of over
    ``(u, v, key)``. The same measurement says this works: a footway sits a
    median of 4.3 m from its roadway centreline, 91 % within 15 m.

    The corridor is a property of the **edge**, exactly like ``segx_layer``
    (P7), and rides through aggregation as a passthrough column — it must never
    enter the ``(mode, category, binding)`` partition key (invariant #6).

    The matched corridor is an **inference**, not data, so the parameters below
    are stamped into ``stats["cross_mode"]`` and the output metadata — same
    discipline as ``transit_source = "osm_assumed"`` (invariant #7).
    """

    enabled: bool = Field(
        default=True,
        description="Resolve canonical corridors across modes. Only has an effect when the run holds "
        "more than one mode graph; a single-mode run pools with itself either way.",
    )
    tolerance_m: float = Field(
        default=20.0,
        gt=0.0,
        description="Maximum distance from an edge's midpoint to a candidate corridor anchor, in "
        "working-CRS metres. Measured p90 of sidewalk-to-roadway distance in Philadelphia: 14.2 m.",
        examples=[15.0, 20.0, 30.0],
    )
    max_bearing_deg: float = Field(
        default=30.0,
        ge=0.0,
        le=90.0,
        description="Maximum |bearing difference| (mod 180°) between an edge and its anchor. This is "
        "what stops a perpendicular crossing from being absorbed into the corridor of the street it "
        "crosses.",
        examples=[20.0, 30.0, 45.0],
    )
    anchor_priority: list[Literal["walk", "drive", "transit"]] = Field(
        default_factory=lambda: ["drive", "walk", "transit"],
        description="Mode order used to pick the anchor layer — the first mode present in the run "
        "supplies the corridor geometry. Roadway centrelines ('drive') first, because a corridor is a "
        "roadway block.",
    )
    exclude_highway: list[str] = Field(
        default_factory=lambda: ["corridor", "elevator", "steps"],
        description="OSM ``highway`` values that never join a street corridor — an indoor concourse "
        "under an avenue is not co-presence with it. Such edges get a corridor of their own.",
    )
    exclude_tunnel: bool = Field(
        default=True,
        description="Treat any edge with a truthy OSM ``tunnel`` tag as its own corridor, for the same reason.",
    )


class NetworkConfig(BaseModel):
    """Per-mode routable-graph construction.

    ``walk`` / ``drive`` are street graphs from OSM. ``transit`` (P7) is the
    **merged** walk ⊕ public-transport graph built by
    :mod:`segx.network.transit`; it is only buildable when
    ``transit.enabled`` is set, and it needs ``walk`` (and, for surface-ride
    map-matching, ``drive``) to be built first — :func:`segx.network.builder.
    load_or_fetch_graph` enforces that ordering.
    """

    modes: list[Literal["walk", "drive", "transit"]] = Field(default_factory=lambda: ["walk", "drive"])
    walk_speed_kmh: float = 5.0
    drive_speed_fallback_kmh: float = 30.0
    # Pin these for reproducible academic results.
    osm_snapshot_date: str | None = None  # e.g. "2026-05-01"; None = "now" (warned)
    osm_pbf_url: str | None = None  # if set, QuackOSM reads this PBF instead of live download
    simplify: bool = True
    # User's choice of directed vs. undirected street graph.
    # When ``True`` each per-mode graph is collapsed to undirected at build time so
    # a street is a single segment/line on the output maps; routing, aggregation and
    # rendering all run on it. Distinct from ``simplify`` (degree-2 node removal).
    # NB: an undirected *drive* graph ignores one-way restrictions — the user opts in.
    undirected: bool = False
    # Snap audit: any point whose nearest network node is farther than this
    # gets flagged. Working CRS is metric, so units = m.
    snap_tolerance_m: float = 250.0
    # P7.2: cross-mode corridor resolution. See :class:`StreetSpaceConfig`.
    street_space: StreetSpaceConfig = Field(default_factory=StreetSpaceConfig)


# --------------------------------------------------------------------------- #
# Public transport (P7)                                                        #
# --------------------------------------------------------------------------- #


def _parse_hhmm(value: str) -> int:
    """``"07:30"`` -> minutes after midnight. GTFS also allows hours >= 24."""
    parts = str(value).split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected HH:MM (or HH:MM:SS), got {value!r}")
    try:
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        raise ValueError(f"expected HH:MM (or HH:MM:SS), got {value!r}") from None
    if not 0 <= minutes < 60 or not 0 <= seconds < 60 or hours < 0:
        raise ValueError(f"not a valid time of day: {value!r}")
    return hours * 60 + minutes + (1 if seconds >= 30 else 0)


#: A service window wider than this earns a ``UserWarning`` (P7.1): it mixes
#: peak and off-peak into a single average headway. A warning and not an error,
#: because an all-day accessibility study is a legitimate use (invariant #3).
WIDE_WINDOW_WARN_MIN: int = 240

#: Minutes in one service day. A window longer than this is a hard error — it
#: would count the same service twice and deflate every headway.
DAY_MINUTES: int = 24 * 60


def parse_window(service_window: tuple[str, str]) -> tuple[int, int]:
    """``(start, end)`` in minutes after midnight; ``end <= start`` crosses midnight.

    The single place the window grammar is decided, so ``window_bounds_min``,
    ``window_minutes``, the config validator and the transit cache key can never
    disagree. Two spellings of a night window are therefore the *same* window:
    ``("22:00", "02:00")`` (the intuitive one) and ``("22:00", "26:00")`` (the
    GTFS extended-hours one) both yield ``(1320, 1560)``.
    """
    start, end = (_parse_hhmm(v) for v in service_window)
    if end <= start:
        end += DAY_MINUTES
    return start, end


class TransitConfig(BaseModel):
    """Public transport as a third routing mode (P7).

    Off by default — nothing about a run changes until ``enabled`` is set. The
    model is a **frequency-based merged graph** (walk ⊕ transit in one
    ``MultiDiGraph`` weighted in ``travel_time_min``), so a plain Dijkstra yields
    *walk → ride → transfer → ride → walk* and the whole routing / aggregation /
    metric stack works unchanged.

    Every behavioural assumption below is a knob (invariant #3): waiting time,
    penalties, radii and the surface-vs-rail split are all runtime-configurable,
    and the values shipped here are defaults, not measured facts.

    * ``source`` — ``"gtfs"`` (require a feed), ``"osm"`` (force the fallback), or
      ``"auto"`` (default: GTFS when ``gtfs_path``/``gtfs_url`` is set, else OSM).
      An OSM-derived feed has **no timetables**, so its headways come from
      ``osm_assumed_headway_min`` and the result is stamped
      ``transit_source = "osm_assumed"`` — an approximate run must never be
      mistaken for a measured one (same discipline as ``osm_snapshot_date``).
    * ``service_day`` / ``service_window`` — when headways are measured. A
      06:00–09:00 weekday run and a 14:00–16:00 run are two config values apart.
      The default starts at **06:00** because the early commute is real (people
      do take the bus at 6 a.m.); note that widening a window *raises* the
      average headway, since ``headway = window_minutes / n_trips`` mixes the empty
      6 a.m. into the 8 a.m. peak — run two windows and compare rather than reading
      one average as the whole story (P7.1). ``end <= start`` is read as
      crossing midnight (:func:`parse_window`), so ``("22:00", "02:00")`` works and
      means the same as the GTFS spelling ``("22:00", "26:00")``.
    * ``max_wait_min`` — cap on the modelled wait ``min(headway/2, max_wait)``.
      Half the headway is the expected wait of a passenger arriving at a random
      time, which is the right assumption precisely because the OD data carries
      no departure times.
    * ``surface_route_types`` — GTFS ``route_type`` values whose rides run *on the
      street* and are therefore map-matched onto street segments (tram ``0``, bus
      ``3``, trolleybus ``11``). Everything else (subway ``1``, rail ``2``, ferry
      ``4``, monorail ``12``) rides on its own ``rail_line`` layer, because a
      passenger in a tunnel is co-present with nobody on the avenue above.
    * ``overlay_path`` — the **editing seam**. A GeoJSON/GPKG of ``add_stop`` / ``remove_stop`` / ``move_stop`` /
      ``edit_route_sequence`` records applied after the feed loads and before the
      graph is built, with its content hash folded into the transit cache key so
      an edited scenario caches separately. The interactive *editor* that writes
      this file is deliberately **not** built here; the file format is, so that stays pure UI work — and scenario
      comparison ("what if we add a metro station here?") comes free.
    """

    enabled: bool = False
    source: Literal["gtfs", "osm", "auto"] = "auto"
    gtfs_path: Path | None = None
    gtfs_url: str | None = None

    service_day: Literal["weekday", "saturday", "sunday"] = "weekday"
    service_window: tuple[str, str] = Field(
        default=("06:00", "09:00"),
        description=(
            "Window in which headways are measured, as (start, end) 'HH:MM'. `end <= start` is "
            "read as crossing midnight, so ('22:00', '02:00') and the GTFS extended-hours "
            "spelling ('22:00', '26:00') are the same window. Longer than 4 h warns (it blends "
            "peak and off-peak into one headway); longer than 24 h is an error."
        ),
        examples=[["06:00", "09:00"], ["07:30", "09:00"], ["22:00", "02:00"]],
    )

    max_wait_min: float = 30.0
    board_penalty_min: float = 1.0
    transfer_penalty_min: float = 2.0
    max_transfer_distance_m: float = 400.0
    max_access_distance_m: float = 800.0

    surface_route_types: list[int] = Field(default_factory=lambda: [0, 3, 11])
    map_match_surface_rides: bool = True

    osm_assumed_headway_min: dict[str, float] = Field(
        default_factory=lambda: {"bus": 15.0, "subway": 5.0, "tram": 10.0, "light_rail": 10.0, "rail": 20.0}
    )

    overlay_path: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> TransitConfig:
        raw_start, raw_end = (_parse_hhmm(v) for v in self.service_window)
        if raw_start == raw_end:
            raise ValueError(
                f"transit.service_window start and end are the same time of day ({self.service_window!r}); "
                "a zero-length window has no headway. For a whole service day write ('00:00', '24:00')."
            )
        start, end = parse_window(self.service_window)
        minutes = end - start
        if minutes > DAY_MINUTES:
            raise ValueError(
                f"transit.service_window spans {minutes} min (> 24 h), got {self.service_window!r}: a window "
                "longer than one service day counts the same service twice and deflates every headway"
            )
        if minutes > WIDE_WINDOW_WARN_MIN:
            warnings.warn(
                f"transit.service_window spans {minutes} min: a window this wide blends peak and off-peak "
                "into a single average headway (headway = window_minutes / n_trips). Consider running two "
                "windows and comparing them instead.",
                UserWarning,
                stacklevel=2,
            )
        if end > DAY_MINUTES:
            warnings.warn(
                f"transit.service_window {self.service_window!r} crosses midnight. This is honoured only for "
                "trips encoded with GTFS extended hours (24:00-27:59) under a single service_id; a trip filed "
                "under the *next day's* service_id is dropped by the calendar filter before the window is "
                "applied, because transit.service_day is a single day.",
                UserWarning,
                stacklevel=2,
            )
        for name in ("max_wait_min", "board_penalty_min", "transfer_penalty_min"):
            if getattr(self, name) < 0:
                raise ValueError(f"transit.{name} must be >= 0")
        for name in ("max_transfer_distance_m", "max_access_distance_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"transit.{name} must be > 0")
        if any(v <= 0 for v in self.osm_assumed_headway_min.values()):
            raise ValueError("transit.osm_assumed_headway_min values must be > 0 minutes")
        if self.source == "gtfs" and self.gtfs_path is None and self.gtfs_url is None:
            raise ValueError("transit.source='gtfs' requires transit.gtfs_path or transit.gtfs_url")
        return self

    @property
    def window_minutes(self) -> float:
        """Length of the headway-measurement window, in minutes."""
        start, end = parse_window(self.service_window)
        return float(end - start)

    @property
    def window_bounds_min(self) -> tuple[int, int]:
        """``service_window`` as minutes after midnight, ``(start, end)``.

        ``end`` exceeds 1440 for a window that crosses midnight — the same
        convention GTFS uses for trips running past midnight, which is what lets
        the two be compared directly in :func:`segx.io.gtfs._trips_in_window`.
        """
        return parse_window(self.service_window)

    def resolved_source(self) -> Literal["gtfs", "osm"]:
        """Which feed source an enabled run will actually use."""
        if self.source != "auto":
            return self.source
        return "gtfs" if (self.gtfs_path is not None or self.gtfs_url is not None) else "osm"

    def layer_for_route_type(self, route_type: int) -> Literal["street", "rail_line"]:
        """Which output layer a ride of *route_type* deposits its mass on."""
        return "street" if int(route_type) in set(self.surface_route_types) else "rail_line"


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #


class RouteSetConfig(BaseModel):
    """Route-set per OD pair (P4.5) — spread a commute flow over *k* plausible
    routes instead of piling all its mass on the single geodesic. All parameters are runtime-configurable so the
    methods can be compared.

    * ``k`` — routes per OD pair. **``1`` = off** (single shortest path, the
      default), so existing behaviour is unchanged until enabled.
    * ``method`` — ``"link_penalty"`` (behavioural: shortest → penalise its edges
      → recompute; alternatives are spatially distinct) or ``"yen"`` (exact
      k-shortest *simple* paths; alternatives are provably next-shortest but tend
      to be near-duplicates of route 1). The two are compared in the demo.
    * ``penalty`` — link-penalty multiplier applied to a chosen route's edges
      before recomputing (``> 1``; higher → longer, more distinct alternatives).
      Ignored by ``"yen"``.
    * ``weight_scheme`` — how the flow's mass splits across the *k* routes with
      ``Σ wᵢ = 1``: ``"uniform"`` (1/k), ``"inverse_cost"`` (``wᵢ ∝ 1/costᵢ``), or
      ``"path_size_logit"`` (a route-choice model that down-weights routes
      overlapping others — Ben-Akiva path-size logit).
    * ``logit_beta`` — cost sensitivity of ``path_size_logit`` (per unit of the
      routing weight, i.e. per minute). Larger → the cheapest route dominates.
    """

    k: int = 1
    method: Literal["link_penalty", "yen"] = "link_penalty"
    penalty: float = 1.5
    weight_scheme: Literal["uniform", "inverse_cost", "path_size_logit"] = "path_size_logit"
    logit_beta: float = 0.1

    @model_validator(mode="after")
    def _check(self) -> RouteSetConfig:
        if self.k < 1:
            raise ValueError("route_set.k must be >= 1 (1 = single shortest path, route-set off)")
        if self.penalty <= 1.0:
            raise ValueError("route_set.penalty must be > 1.0 (a penalty of 1 never changes the route)")
        if self.logit_beta < 0.0:
            raise ValueError("route_set.logit_beta must be >= 0")
        return self


class RoutingConfig(BaseModel):
    """Which engine routes, and the mode the access bindings (A/B) use.

    ``engine`` selects the implementation behind ``routing/engine.py``:
    ``"pandana"`` (default, fast, core dep) or ``"networkx"`` (the slower
    validation oracle — handy for tiny runs and as a fallback). The OD binding
    (C) always routes each pair in its own mode; ``access_mode`` is the mode used
    for the nearest/catchment access bindings (plan default: walking). Setting it
    to ``"transit"`` (P7) answers "which parks are reachable within 30 min *by
    bus*" with no other change — the merged graph is just another mode.
    ``route_set`` controls the P4.5 route-set per OD pair (off by default).
    """

    engine: Literal["pandana", "networkx"] = "pandana"
    access_mode: Literal["walk", "drive", "transit"] = "walk"
    route_set: RouteSetConfig = Field(default_factory=RouteSetConfig)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


class CrossModeExposureConfig(BaseModel):
    """How much co-presence counts *between* travel modes (P7.1).

    A pedestrian and a bus passenger on the same street are co-present — less
    than two pedestrians are, and ``factor`` is that discount. It is a parameter
    of the ``cross_mode_exposure`` metric only: it changes no graph, no route and
    no cache key, and the P5 ``exposure`` metric stays untouched and comparable.

    **What ``transit`` means here (P7.2):** *inside the vehicle*. A public-transport
    trip's walking legs (origin → stop, stop → destination) are labelled ``walk``
    per-leg by :func:`segx.network.transit.expand_transit_rides`, so
    ``factor("walk", "transit")`` is genuinely "pedestrian × bus passenger" and no
    longer "pedestrian × (bus passenger + the same passenger walking to the stop)".

    * ``factor`` — the default cross-mode weight (``0.5``). ``0`` reproduces
      ``exposure`` exactly; ``1`` pools every mode as if they were one crowd.
    * ``overrides`` — per mode-pair exceptions, keyed ``"mode_a|mode_b"``. The key
      is **symmetric**: it is stored sorted, so ``"transit|walk"`` and
      ``"walk|transit"`` are one entry (a *directional* factor is a possible
      extension). Same-mode weight
      is always ``1.0`` and cannot be overridden.

    From the CLI the whole dict is replaced at once, since ``--set`` parses JSON::

        --set metrics.cross_mode.overrides='{"transit|walk": 0.7}'
    """

    factor: float = Field(
        default=0.5,
        description=(
            "Weight applied to another mode's mass when measuring a mode's experienced exposure. "
            "0 = modes never meet (identical to `exposure`); 1 = full pooling."
        ),
    )
    overrides: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per mode-pair overrides of `factor`, keyed 'mode_a|mode_b' (symmetric — stored with the "
            "two modes sorted). Unknown modes, repeated modes and negative values are rejected."
        ),
        examples=[{"transit|walk": 0.7}],
    )

    @model_validator(mode="after")
    def _check(self) -> CrossModeExposureConfig:
        # Imported here so `config` keeps no import-time dependency on the network
        # package; `registry` is the one place travel-mode names are enumerated.
        from segx.network.registry import KNOWN_MODES

        if self.factor < 0.0:
            raise ValueError(f"metrics.cross_mode.factor must be >= 0, got {self.factor}")
        canonical: dict[str, float] = {}
        for raw, value in self.overrides.items():
            parts = [p.strip() for p in str(raw).split("|")]
            if len(parts) != 2:
                raise ValueError(f"metrics.cross_mode.overrides key {raw!r} must be of the form 'mode_a|mode_b'")
            unknown = sorted({p for p in parts if p not in KNOWN_MODES})
            if unknown:
                raise ValueError(
                    f"metrics.cross_mode.overrides key {raw!r} names unknown mode(s) {unknown}; "
                    f"known modes: {sorted(KNOWN_MODES)}"
                )
            if parts[0] == parts[1]:
                raise ValueError(
                    f"metrics.cross_mode.overrides key {raw!r} names the same mode twice; "
                    "the same-mode factor is always 1.0 and cannot be overridden"
                )
            if value < 0.0:
                raise ValueError(f"metrics.cross_mode.overrides[{raw!r}] must be >= 0, got {value}")
            key = "|".join(sorted(parts))
            if key in canonical and canonical[key] != float(value):
                raise ValueError(
                    f"metrics.cross_mode.overrides gives conflicting values for {key!r}: "
                    f"{canonical[key]} and {float(value)} (the key is symmetric)"
                )
            canonical[key] = float(value)
        # Store the canonical form: `Settings._with_overrides` round-trips through
        # `model_dump()` -> `model_validate()`, so anything not normalised here
        # would have to be normalised again on every override.
        object.__setattr__(self, "overrides", canonical)
        return self

    def factor_for(self, a: str, b: str) -> float:
        """The weight mode *a* gives to mode *b*'s mass (``1.0`` when equal)."""
        if a == b:
            return 1.0
        return self.overrides.get("|".join(sorted((a, b))), self.factor)


class FragmentationConfig(BaseModel):
    """The scale ladder of the ``path_fragmentation`` metric (P7.3).

    Netto et al. (2015) measure fragmentation as the **linear fractal dimension**
    of a personal trajectory: "trajectories are converted into pixels in different
    resolutions (…) We opted for a 0.7m pixel and a scaling factor of 5 times",
    with ``log N = log K + (1-D) log r`` fitted over the six resolutions, ``N``
    the number of pixels at each. That is **box counting**, and it is the default
    ``method``: ``pixel_m`` / ``pixel_factor`` / ``n_pixel_scales`` are the
    published ladder (0.7, 3.5, 17.5, 87.5, 437.5, 2187.5 m).

    ``method = "divider"`` keeps the earlier reading — Richardson's structured
    walk with rulers *relative to each path's length* (``n_scales``,
    ``scale_ratio``, ``largest_fraction``, ``min_segments``). It is exact on a
    straight line and useful as a cross-check, but each path is measured at its
    own scales, so it does not reproduce the paper.

    All of these are *measurement assumptions*, not data — the same status as the
    P7 board/transfer penalties and the P7.2 corridor tolerances — so they are
    stamped into ``stats["fragmentation"]["params"]`` and travel into the output
    metadata (invariant #7). Two paths compared under different ladders are not
    comparable, and nothing in a results table would say so.
    """

    method: Literal["box_counting", "divider"] = Field(
        default="box_counting",
        description="`box_counting` = the paper's method (pixels at fixed absolute resolutions). "
        "`divider` = structured walk with rulers relative to each path's length.",
    )
    pixel_m: float = Field(
        default=0.7,
        description="Box counting: the finest pixel size, in metres. 0.7 is the paper's.",
    )
    pixel_factor: int = Field(
        default=5,
        description="Box counting: each resolution multiplies the previous pixel by this. Integer, "
        "so the grids nest and the path is rasterised only once.",
    )
    n_pixel_scales: int = Field(
        default=6,
        description="Box counting: number of resolutions. 6 with the defaults = 0.7 … 2187.5 m.",
    )
    drop_saturated: bool = Field(
        default=True,
        description="Box counting: leave out of the fit the resolutions at which the whole path falls "
        "in a single pixel (`N < 2`). Kept in, they pin the fit at `log N = 0` and make `D` a "
        "function of the path's extent — a 300 m straight path scores 0.80 instead of 0.97. If fewer "
        "than `min_scales` resolutions remain the path reports `NaN`.",
    )
    as_written: bool = Field(
        default=False,
        description="Box counting: report `D = 1 - slope`, the paper's equation applied literally to "
        "pixel counts. A straight line then scores 2 and every value is exactly 1 above the default "
        "`D = -slope` (straight line = 1).",
    )
    use_edge_geometry: bool = Field(
        default=True,
        description="Trace each street segment along its real geometry (curved streets included) "
        "rather than the straight chord between its end nodes.",
    )
    n_scales: int = Field(
        default=5,
        description="Divider only: maximum number of ruler lengths in the ladder. The ladder is cut "
        "short by the mean-segment-length floor, so a short path uses fewer.",
    )
    scale_ratio: float = Field(
        default=2.0,
        description="Divider only: each rung divides the previous ruler by this. 2.0 = halve the ruler each step.",
    )
    largest_fraction: float = Field(
        default=0.5,
        description="Divider only: the largest ruler, as a fraction of the path's total length. "
        "0.5 = the first measurement takes about two steps.",
    )
    min_segments: int = Field(
        default=8,
        description="Divider only: paths with fewer segments report `fragmentation = NaN` rather than "
        "a number fitted to too little structure (invariant #2). The default is not arbitrary: it is "
        "exactly the floor the default ladder already implies, "
        "`scale_ratio**(min_scales-1) / largest_fraction` = 2**2 / 0.5. Raise it to demand more "
        "structure; lowering it below the implied floor has no effect, since the ladder rejects "
        "those paths anyway.",
    )
    min_scales: int = Field(
        default=3,
        description="Scales required before a slope is fitted at all (both methods). "
        "Below this the path reports `fragmentation = NaN`.",
    )

    @model_validator(mode="after")
    def _check(self) -> FragmentationConfig:
        if self.pixel_m <= 0.0:
            raise ValueError(f"metrics.fragmentation.pixel_m must be > 0, got {self.pixel_m}")
        if self.pixel_factor < 2:
            raise ValueError(
                f"metrics.fragmentation.pixel_factor must be an integer >= 2, got {self.pixel_factor} "
                "(the resolutions must grow)"
            )
        if self.n_pixel_scales < self.min_scales:
            raise ValueError(
                f"metrics.fragmentation.n_pixel_scales ({self.n_pixel_scales}) is below min_scales "
                f"({self.min_scales}); every path would report NaN"
            )
        if self.scale_ratio <= 1.0:
            raise ValueError(
                f"metrics.fragmentation.scale_ratio must be > 1, got {self.scale_ratio} (the ladder "
                "must shrink; <= 1 never terminates or never descends)"
            )
        if not 0.0 < self.largest_fraction <= 1.0:
            raise ValueError(f"metrics.fragmentation.largest_fraction must be in (0, 1], got {self.largest_fraction}")
        if self.min_scales < 2:
            raise ValueError(
                f"metrics.fragmentation.min_scales must be >= 2, got {self.min_scales} "
                "(a slope needs at least two points)"
            )
        if self.n_scales < self.min_scales:
            raise ValueError(
                f"metrics.fragmentation.n_scales ({self.n_scales}) is below min_scales "
                f"({self.min_scales}); every path would report NaN"
            )
        if self.min_segments < 2:
            raise ValueError(f"metrics.fragmentation.min_segments must be >= 2, got {self.min_segments}")
        return self


class MetricsConfig(BaseModel):
    """Which segment metrics run and their tunables (invariant #5: metrics are
    plugins; nothing is hard-coded).

    ``enabled`` is ``None`` → run every registered metric; otherwise the named
    subset (resolved against :mod:`segx.metrics.registry`). ``disabled`` switches
    individual metrics **off** and is applied after ``enabled``, so one metric can
    be dropped without restating the other names (``metrics.disabled =
    ["path_fragmentation"]``). Both are lists of registry names rather than one
    boolean field per metric, so a newly registered plugin is switchable without
    touching this model. :meth:`active` is the one place the two are resolved.
    A disabled metric's supporting work is skipped too — with
    ``path_fragmentation`` off the per-path accumulator never runs.
    ``exclusive_lq_threshold`` is the location-quotient at/above which a segment's
    dominant group counts as *exclusive* in ``segregated_trajectories``
    (``LQ = 2`` → the group is twice as concentrated on the segment as in the
    partition overall).
    ``cross_mode`` tunes the ``cross_mode_exposure`` metric (P7.1);
    ``fragmentation`` the ``path_fragmentation`` metric (P7.3).
    """

    enabled: list[str] | None = None
    disabled: list[str] = Field(
        default_factory=list,
        description="Metrics to switch off, by registry name. Applied after `enabled`.",
        examples=[["path_fragmentation"], ["path_fragmentation", "cross_mode_exposure"]],
    )
    exclusive_lq_threshold: float = 2.0
    cross_mode: CrossModeExposureConfig = Field(default_factory=CrossModeExposureConfig)
    fragmentation: FragmentationConfig = Field(default_factory=FragmentationConfig)

    def active(self, available: list[str]) -> list[str]:
        """The metric names this run computes, given the registry's *available* names.

        ``enabled`` (or everything, when ``None``) minus ``disabled``, in the order
        ``enabled`` gave them. A name the registry does not know is a typo, and a
        typo in ``disabled`` would otherwise leave the metric silently *on* — so
        both lists are checked and an unknown name raises.
        """
        unknown = sorted({*(self.enabled or []), *self.disabled} - set(available))
        if unknown:
            raise ConfigValueError(
                f"unknown metric name(s) in metrics.enabled/disabled: {unknown}; available: {sorted(available)}",
                details={"unknown": unknown, "available": sorted(available)},
            )
        names = list(self.enabled) if self.enabled is not None else list(available)
        off = set(self.disabled)
        return [name for name in dict.fromkeys(names) if name not in off]


# --------------------------------------------------------------------------- #
# Input analysis (pre-routing diagnostics)                                     #
# --------------------------------------------------------------------------- #


class ODSpreadConfig(BaseModel):
    """The OD destination-spread analysis (P7.5).

    "Group the OD table by origin and see how many destinations the different
    classes go to" — a reading of the *input table*, before any routing, so it
    says what the data itself already contains about unequal reach. See
    :mod:`segx.analysis.od_spread`.
    """

    enabled: bool = Field(
        default=True,
        description="Run the OD destination-spread analysis when OD pairs are supplied. Skipped "
        "automatically (never an error) when the OD file carries no per-class columns.",
    )
    min_mass: float = Field(
        default=0.0,
        description="Ignore a flow whose mass for a class is at or below this. 0.0 counts every "
        "non-zero flow; raising it answers 'where do at least N people of this class go?' and "
        "filters the long tail of one-person flows that dominate a raw destination count.",
        examples=[0.0, 1.0, 5.0],
    )
    max_class_pairs: int = Field(
        default=64,
        description="Guard on the pairwise destination-overlap table: an attribute with C classes "
        "costs C·(C-1)/2 joins over the whole OD file. Above this the overlap table is skipped "
        "(the per-origin spread table still runs) rather than silently taking minutes.",
    )


class AnalysisConfig(BaseModel):
    """Analyses of the **inputs**, as opposed to :class:`MetricsConfig`'s plugins
    over the routed edge table.

    The distinction is worth keeping: everything under ``metrics`` describes what
    people *experience* along their trajectories and needs a street network to
    exist; everything here is a property of the supplied tables and runs (and can
    fail) before a single Dijkstra.
    """

    od_spread: ODSpreadConfig = Field(default_factory=ODSpreadConfig)


# --------------------------------------------------------------------------- #
# Top-level settings                                                           #
# --------------------------------------------------------------------------- #


class Settings(BaseModel):
    """The single resolved configuration object passed through the pipeline."""

    crs: CRSConfig = Field(default_factory=CRSConfig)
    demographics: DemographicSchema = Field(default_factory=DemographicSchema.ibge_default)
    od_columns: ODColumnMapping = Field(default_factory=ODColumnMapping)
    poi: POIBindingConfig = Field(default_factory=POIBindingConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    transit: TransitConfig = Field(default_factory=TransitConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    cache_dir: Path = Path("cache")
    output_dir: Path = Path("out")
    random_seed: int = 42

    # ----------------------------------------------------------------------- #
    # Layered construction                                                    #
    # ----------------------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Build from built-in defaults, then a TOML file, then keyword overrides.

        ``overrides`` keys use dotted paths, e.g.
        ``{"poi.catchment.budget_minutes": 45, "od_columns.origin_id": "COD"}``.
        """
        data: dict[str, Any] = {}
        if config_path is not None:
            text = Path(config_path).read_bytes()
            data = tomllib.loads(text.decode("utf-8"))
        settings = cls.model_validate(data)
        if overrides:
            settings = settings._with_overrides(overrides)
        return settings

    def _with_overrides(self, overrides: dict[str, Any]) -> Settings:
        dumped = self.model_dump()
        for dotted, value in overrides.items():
            cursor = dumped
            *parents, leaf = dotted.split(".")
            for key in parents:
                if key not in cursor or not isinstance(cursor[key], dict):
                    raise ConfigKeyError(
                        f"override path {dotted!r}: no such config section {key!r}",
                        details={"path": dotted, "section": key},
                    )
                cursor = cursor[key]
            if leaf not in cursor:
                raise ConfigKeyError(
                    f"override path {dotted!r}: no such config key {leaf!r}",
                    details={"path": dotted, "key": leaf},
                )
            cursor[leaf] = value
        return type(self).model_validate(dumped)


__all__ = [
    "DAY_MINUTES",
    "WIDE_WINDOW_WARN_MIN",
    "AttributeKind",
    "CRSConfig",
    "CatchmentBindingConfig",
    "CrossModeExposureConfig",
    "DEFAULT_POI_PRESETS",
    "DemographicAttribute",
    "DemographicSchema",
    "MetricsConfig",
    "NearestBindingConfig",
    "NetworkConfig",
    "ODColumnMapping",
    "POIBindingConfig",
    "RouteSetConfig",
    "RoutingConfig",
    "Settings",
    "StreetSpaceConfig",
    "TransitConfig",
    "parse_window",
]
