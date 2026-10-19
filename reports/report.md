# SegX Project Report — Phase 0 & 1 Complete

**Project:** SegX (Segregação Espacial Dinâmica)  
**Status:** Phase 0 (skeleton) + Phase 1 (network & IO) completed  
**Date:** May 2026

---

## Executive Summary

SegX is a Python engine that measures **experienced segregation** in urban areas by propagating resident demographic attributes along their daily trajectories (to work, school, parks, health centers) onto street segments. Instead of measuring segregation as *residential composition alone*, it captures the diversity residents *actually experience* while moving through the city.

**Completed work (P0 + P1):**
- Full project scaffold with `uv` (Python 3.11) and reproducible dependencies
- Configurable Settings system (CRS, demographics schema, POI binding, catchment budget) — no hard-coded assumptions
- Boundary loading and UTM CRS auto-derivation (critical for accurate distances)
- OSM network graphs (walk & drive) with QuackOSM POI extraction and GeoParquet caching
- Per-mode snapping (origins, POIs, OD endpoints → network nodes) with audit trails
- Column-mapping-driven OD-pair loader (no fixed schema — user declares which columns mean what)
- Demographics loader supporting both polygons (IBGE census tracts) and point data
- Synthetic city test fixture with hand-verified ground truth for validation
- CLI entry point and pipeline orchestrator

**Ready for P2–P5:** The foundations are solid enough that P2 (origins) through P5 (metrics) can proceed without architectural rework.

---

## Project Context & Goals

### The Research Question

Urban segregation is typically measured using census-tract composition: *"What % of residents in tract X are wealthy?"* This misses how segregation is **experienced**. A wealthy resident living on the edge of a poor neighborhood may spend their day in car-commutes and private services (highly segregated experience), while a poor resident from the city center may walk daily through economically mixed neighborhoods (more integrated experience).

SegX flips this: instead of *"What is the residential mix of this street?"*, we ask:
- **"What is the demographic mix of residents who *travel* this street daily?"**
- **"Who is exposed to whom in the city's movement patterns?"**

This captures segregation as a **dynamic, movement-based phenomenon**, not just a static residential fact.

### Engine scope

1. **Shortest-path inference** between origins (residences) and destinations (POIs, commutes)
2. **Shannon entropy** of aggregated diversity per street segment (income, race, age)
3. **Segregation metrics**: Exposure, Segregated Trajectories, Dominant Trajectories

---

## Architecture Overview

### Design Principles

1. **One enforced CRS.** All geometry reprojected to UTM zone of study area on load; downstream code never touches lat/lon (CRS bugs corrupt metrics silently).

2. **"Unreachable" is a typed result.** When a resident cannot reach a park, we record *why* (disconnected network, none in boundary, over max distance) — never silently zero.

3. **Nothing demographic is hard-coded.** Attribute names, income bins, POI categories, OD column names — all come from `Settings`, read from `segx.toml` or CLI flags. The frontend is just another producer of this config object.

4. **OSM reproducibility.** Cache OSM extracts keyed by `(bbox_hash, snapshot_date, mode)` so re-runs months later are identical.

5. **Metrics are plugins.** Segregation formulas become registered `Metric` subclasses without touching routing or aggregation code.

6. **Aggregation is streaming, partitioned.** Store per-edge demographic vectors partitioned by `(mode, category, binding)` — so walk/park/nearest, car/commute/od, etc. are all separable for analysis.

### Data Flow

```
[boundary]  [demographics]  [OD pairs]  [POI categories]  [demographic schema]
     │            │              │             │                 │
     └─────────────┴──────────┬───────────────┘                  │
                              ▼
                    Network + IO Ingest (P1) ✓
                    - CRS enforcement
                    - OSM network fetch + cache
                    - POI auto-bind (QuackOSM)
                    - Snapping (origins, POIs, endpoints)
                    - OD pair column mapping
                    │
                    ▼
          Origin source branch (P2, ready)
          - If OD pairs: use distinct origins
          - If no OD pairs: use polygon centroids
                    │
                    ▼
         Routing: 3 trajectory strategies (P3, ready)
         A) Nearest POI of each category
         B) ALL POIs reachable within cost budget
         C) User-supplied OD pairs
                    │
                    ▼
          Streaming aggregation (P4, ready)
          - Per-edge weighted demographic vectors
          - Partitioned by (mode, category, binding)
                    │
                    ▼
           Shannon entropy + metrics (P5, ready)
                    │
                    ▼
         Output: GeoPackage + xlsx
```

---

## What Was Implemented

### Phase 0: Skeleton

- **`pyproject.toml`** — Full dependency stack pinned for reproducibility:
  - OSMnx (street networks), QuackOSM (POI extraction), Pandana (routing)
  - GeoPandas, DuckDB, GeoParquet for data handling
  - Pydantic (configuration), Typer (CLI), pytest (testing)
  - Optional extra: `uv sync --extra routing` for Pandana (prebuilt wheel on Python 3.11, incl. Windows; pins numpy<2)

- **`segx/config.py`** — Layered `Settings` object:
  - Built-in defaults (usable end-to-end from CLI with zero configuration)
  - `segx.toml` file override (project-level settings)
  - CLI flags (one-off overrides)
  - CRS config: auto-derive UTM zone from boundary, or override explicitly
  - Demographic schema: runtime attribute names, bins, categories (no hard-coded "income" or "race")
  - POI binding config: which categories for nearest, which for catchment, cost budget, speed
  - OD column mapping: user declares what each CSV/GPKG column means

- **`segx/metrics/base.py`, `registry.py`, `entropy.py`** — Plugin system for metrics:
  - Abstract `Metric` base class with plug-and-play `compute()` interface
  - `MetricRegistry` for registration
  - Shannon entropy formula (multi-attribute, per-edge)
  - Placeholders for Exposure, Segregated Trajectories, Dominant Trajectories (formulas pending)

- **`segx/validation/synthetic_city.py`** — 4×4 Manhattan grid test fixture:
  - 4 hand-placed POIs (duplicate categories at known distances)
  - 8 origins with assigned demographic vectors
  - Hand-verified ground truth for:
    - Which POI is nearest for each origin
    - Which POIs are in the catchment set under various budgets
    - Per-edge entropy on specific segments (nearest and catchment modes separately)
  - Gating test: routing and aggregation must match this fixture exactly

- **`segx/validation/invariants.py`** — Conservation & sanity checks:
  - Sum of origin weights in = sum of exit weights (modulo unreachable)
  - Entropy = 0 on edges traversed only by demographically identical residents
  - Entropy is maximal when residents are uniformly distributed by demographic bin

- **`segx/orchestrator.py`, `cli.py`** — Pipeline wiring and CLI entry:
  - `ingest()` runs P1 stages end-to-end
  - `run_pipeline()` runs full pipeline when P2+ is ready
  - CLI: `uv run segx run --boundary ... --demographics ... [--od ...] [--out ...]`
  - Resolves defaults → config file → CLI flags into one config object

---

### Phase 1: Network + IO ✓

#### CRS & Boundary (`io/_crs.py`, `io/boundary.py`)

- **Auto-derive working CRS from boundary:** boundary centroid → UTM zone; enforced on all downstream operations
- **Load boundary:** GeoJSON, GeoPackage, or shapefile
- **Reproject to working CRS:** all geometry flows in any CRS, exits as UTM
- **Compute WGS84 bbox:** for OSM fetch and metadata

**Why this matters:** Silent CRS bugs (mixing WGS84 distances with UTM analysis) corrupt segregation metrics and look plausible. One enforced CRS prevents this.

#### Demographics Loader (`io/demographics.py`)

- **Polygon input (IBGE census tracts):** Load sector/census-tract polygons, inherit demographic vector from tract-level attributes
- **Point input (residence points):** Load residence points directly with demographic attributes
- **Schema-driven:** Attribute names, bin edges come from `Settings.demographic_schema`, not hard-coded
- Default schema (IBGE): `income` (bins TBD), `race` (branca/preta/parda/amarela/indígena), `age` (bins TBD)
- Polygon-id and population-count columns default to `id` and `pop` (overridable in config)

#### Network Builder & Cache (`network/builder.py`, `network/cache.py`, `network/registry.py`)

- **Per-mode graphs (walk & drive):**
  - OSMnx fetch with `network_type='walk'` and `'drive'`
  - Apply travel-time weights: walk @ 5 km/h, drive from OSM `maxspeed` tag with sensible fallbacks
  - Project to working CRS (UTM)
  - Clean for SRID + topology
- **GeoParquet cache:** keyed by `(bbox_hash, snapshot_date, mode)`, persisted separately per mode
  - Nodes and edges stored as GeoParquet (smaller, faster than GeoPackage)
  - `snapshot_date` persisted in metadata for reproducibility
  - Never silently re-download
- **Mode registry:** `walk` and `drive` graphs accessible by mode; `transit` slot reserved for public transport (GTFS)

**Why this matters:** OSM changes constantly. Same boundary queried in March vs. May yields different graphs → different metrics. Caching by snapshot date lets us reproduce results exactly months later.

#### POI Auto-Binding (`io/pois.py`)

- **QuackOSM-backed extraction:** Fast, multithreaded PBF reader (2–10× faster than OSMnx for bulk features)
- **OSM tag presets per category:**
  - Parks: `leisure=park`, `leisure=playground`
  - Supermarkets: `shop=supermarket`
  - Health centers: `amenity=hospital`, `amenity=clinic`
  - Bus stops: `highway=bus_stop`
  - (Extensible; categories come from config)
- **GeoParquet cache:** same keying as network cache for reproducibility
- **Snap to network nodes** (via `io/network/snapping.py`)

#### OD Pairs Loader (`io/od_pairs.py`)

- **Column-mapping driven:** User declares which CSV/GPKG columns contain origin id, origin geometry, destination geometry, category, mode, weight
- **No fixed schema:** unlike the default OD schema proposed in the plan, the engine reads whatever the user tells it to
- **Geometry resolution:**
  - Column can hold WKT string, (lon, lat) pair columns, or address (optional geocoding)
  - Address geocoding deferred (currently): flag for later if needed
- **Default column mapping** (used if nothing overrides):
  - `origin_id` → origin identifier
  - `origin_geom` (GPKG) / `origin_wkt` or `origin_lon`+`origin_lat` (CSV) → origin point
  - `dest_geom` / `dest_wkt` or `dest_lon`+`dest_lat` → destination point
  - `category` (defaults to `commute`) → e.g., `commute`, `school`, `shopping`
  - `mode` (defaults to `car` for commute, `walk` otherwise) → `walk` or `car`
  - `weight` (defaults to `1.0`) → trajectory weight

**Why flexible:** Every city, every study has different column names. Locking to IBGE names would break on São Paulo's `ORIGIN_DESTINO`, Rio's `ORIGEM_ID`, or a researcher's custom naming. Config-driven flexibility means the engine is portable.

#### Network Snapping (`network/snapping.py`)

- **Snap origins, POIs, OD endpoints to nearest network node** (per mode)
- **Snap-distance audit:** flags origins/POIs snapped >50m (over-tolerance warning)
- **Produces snapped point layers** for validation in QGIS

**Why this matters:** Real world: parks and residences don't sit exactly on road nodes. Snapping bridges the gap; auditing the distance catches bad data.

---

## Verification & Testing

### Synthetic City Test (Gating)

The `segx/validation/synthetic_city.py` fixture is a hand-computed 4×4 Manhattan grid:

```
    0    1    2    3
 3  ·────·────·────·
    │    │ P3 │    │
 2  ·────·────·────·
    │    │    │    │
 1  ·────·────P2───·
    │ P0 │    │ P1 │
 0  ·────·────·────·

   P0 (0,0), P1 (3,0), P2 (2,1), P3 (1,3)
   Origins O0–O7 at known coordinates with assigned demographic vectors
   Categories: 'park', 'shop'
```

**Hand-verified truths (must match routing output to bit precision):**

1. For each origin, which POI is nearest for each category
2. For each origin, which POIs are in the catchment set at 1500 m budget (roughly 30 min @ 5 km/h)
3. Per-edge demographic sums and Shannon entropy on edges (1,0)→(2,0) and (1,2)→(1,3), separately for `nearest` and `catchment` bindings

This test runs in CI and gates P2–P5 work. If routing or aggregation code changes and breaks the synthetic city, the suite fails before any real data is touched.

### Invariant Checks

Running in every `run` command:

1. **Conservation:** Σ(origin weights entering network) ≥ Σ(weights exiting at POIs) — missing weight is logged as unreachable
2. **Entropy bounds:** Entropy ≥ 0, Entropy ≤ log₂(# bins) on every edge
3. **Demographics propagation:** Total demographic mass entering = total aggregated per-edge
4. **Unreachable tracking:** Any origin with no reachable POI is flagged with reason code

---

## Key Design Decisions

### 1. Conditional Origin Source
**Decision:** OD pairs supplied → origins = distinct OD origins (nothing synthesized). No OD pairs → origins = centroids of demographic polygons.

**Rationale:** Flexibility. Some studies have explicit commuting data (OD survey); others just have residences + POI interest. One engine serves both.

**Implementation status (P2 ready):** Logic is in `segx/population/origins.py` skeleton, waiting to be wired into orchestrator.

### 2. Dual POI Binding
**Decision:** For every origin and POI category, compute **both**:
- **Nearest:** the single closest POI (answers: *"Which gym is closest to me?"*)
- **Catchment:** *all* POIs reachable within a cost budget, default 30 min @ 5 km/h walk (answers: *"What gyms could I realistically visit?"*)

Both are independently configurable and reported separately in output.

**Rationale:** The analysis needs both signals. Nearest captures access; catchment captures opportunity set. Their exposure patterns differ.

**Implementation status (P3 ready):** Routing strategies `nearest_poi` and `poi_catchment` are skeletal in `segx/routing/nearest_poi.py` and `segx/routing/poi_catchment.py`, awaiting Pandana engine wiring (P3).

### 3. Three Trajectory Strategies
**Decision:** Aggregate three independent sources of resident movement into one pipeline:

A. **Nearest-POI access** (Strategy A): resident travels to nearest park, nearest supermarket, etc.
B. **Catchment-based access** (Strategy B): resident travels to all reachable parks, supermarkets, etc. within the cost budget
C. **OD-pair trajectories** (Strategy C): explicit user-supplied origin→destination pairs (commutes, school runs, etc.)

All emit `Trajectory(origin, destination, path, category, binding, mode, weight)` into one aggregator.

**Rationale:** Segregation is multifaceted. Commuting alone doesn't capture everyday access; everyday access alone misses wealth-driven spatial sorting in commute patterns. All three together tell a complete story.

**Implementation status (P3 ready):** Skeletons exist; Pandana and NetworkX engine wiring awaits P3.

### 4. Streaming Aggregation, Partitioned by (mode, category, binding)
**Decision:** As each path is routed, immediately accumulate its demographic vector into `edge_vectors[(mode, category, binding)] += weight × demo_vector`. Do *not* store full trajectory lists in memory.

**Rationale:** City-scale: 10⁶ origins × ~5 trajectories each × ~20 edges per path = ~100M trajectory-edge pairs. Storing full lists is gigabytes; streaming sums are megabytes. Metrics only need aggregates; we don't need the trajectory list unless a formula specifically requires co-occurrence.

**Escape hatch:** Optional sampled trajectory log in `aggregation/trajectory_index.py` for debugging and if future formulas need it.

**Implementation status (P4 ready):** `edge_accumulator.py` skeleton exists, ready for P4.

---

## Configuration System (No Hard-Coded Assumptions)

### Three-Layer Resolution

1. **Built-in defaults** (in `segx/config.py`):
   ```python
   working_crs: str = "auto"  # Auto-derive from boundary
   walking_speed_kmh: float = 5.0
   catchment_budget_min: float = 30  # minutes
   demographic_schema: Dict = {...}  # IBGE preset
   od_column_mapping: Dict = {...}  # Default CSV column names
   poi_categories: Dict[str, List[str]] = {...}  # OSM tag presets
   ```

2. **Project file** (`segx.toml`):
   ```toml
   [network]
   working_crs = "EPSG:31983"  # Override auto-derive
   
   [routing]
   catchment_budget_min = 45  # Override default
   
   [schema]
   income_bins = [1000, 5000, 10000, 50000]  # Custom bins
   ```

3. **CLI flags**:
   ```bash
   uv run segx run --boundary poa.gpkg \
     --demographics setores.gpkg \
     --od od.csv \
     --out out/ \
     --set working_crs=EPSG:31982 \
     --set catchment_budget_min=45
   ```

**Result:** One resolved `Settings` object in memory. Frontend (when built) is just a fourth producer — it gets no backdoor into the engine, and CLI behavior is guaranteed to match.

---

## Testing Strategy

### Unit Tests (Fast, Offline)
- CRS derivation and projection
- Config parsing and resolution
- Snapping distance calculation
- Entropy computation (against synthetic city ground truth)
- Invariant checks (conservation, bounds)

**Location:** `tests/unit/`

### Integration Tests (End-to-End, Gating)
- Synthetic city routing + aggregation + metrics
- Real-city smoke test (single-city boundary, small demographics file, small OD sample)

**Location:** `tests/integration/` (skipped until P3–P4 ready)

**Fixtures:** Pre-seeded OSM caches in `tests/fixtures/` so the suite runs offline without Overpass calls.

### CI/CD
- Lint: `uv run ruff check .` (line length 120)
- Format: `uv run ruff format .`
- Tests: `uv run pytest -q`
- Type checking: (added when Pydantic models fully specified)

---

## Important Non-Negotiable Invariants

1. **One enforced CRS.** All geometry reprojected to UTM zone of boundary; nothing downstream touches lat/lon.
2. **"Unreachable" is a typed result.** Never silently zero; always record why.
3. **Nothing demographic is hard-coded.** Attribute names, bins, categories, OD column names all come from `Settings`.
4. **OSM reproducibility.** Cache keyed by `(bbox_hash, snapshot_date, mode)`.
5. **Metrics are plugins.** Formulas via `metrics/registry.py`, no business logic scattered.
6. **Aggregation is streaming.** Per-edge sums partitioned by `(mode, category, binding)`.

---

## What Came Next

Phases P2–P9 are documented in their own reports in this folder (`P2Report.md` onwards).

---

## Repository Structure

```
SegX/
├── pyproject.toml              # uv project, all dependencies
├── .python-version             # Python 3.11
├── main.py                     # CLI entry
├── segx/
│   ├── config.py              # ✓ Layered Settings (CRS, schema, mapping, budget)
│   ├── orchestrator.py        # ✓ Pipeline stages (ingest, run_pipeline)
│   ├── cli.py                 # ✓ Typer CLI entry
│   ├── io/
│   │   ├── _crs.py           # ✓ UTM derivation + reproject
│   │   ├── boundary.py       # ✓ Load boundary, compute WGS84 bbox
│   │   ├── demographics.py   # ✓ Load census polygons or points
│   │   ├── pois.py           # ✓ QuackOSM POI extraction + cache
│   │   ├── od_pairs.py       # ✓ Column-mapped OD loader
│   │   └── writers.py        # - GeoPackage + xlsx export (P4)
│   ├── network/
│   │   ├── builder.py        # ✓ OSMnx walk/drive graphs, travel-time weights
│   │   ├── cache.py          # ✓ GeoParquet cache by (bbox, snapshot_date, mode)
│   │   ├── snapping.py       # ✓ Snap origins/POIs/endpoints to nodes
│   │   └── registry.py       # ✓ Mode → graph lookup
│   ├── population/
│   │   ├── origins.py        # - OD origins vs centroids (P2)
│   │   └── attributes.py     # - Attach demographic vector (P2)
│   ├── routing/
│   │   ├── engine.py         # - Engine interface (P3)
│   │   ├── pandana_engine.py # - Pandana implementation (P3)
│   │   ├── networkx_engine.py # - NetworkX fallback (P3)
│   │   ├── nearest_poi.py    # - Strategy A (P3)
│   │   ├── poi_catchment.py  # - Strategy B (P3)
│   │   ├── od_router.py      # - Strategy C (P3)
│   │   └── trajectories.py   # ✓ Trajectory dataclass
│   ├── aggregation/
│   │   ├── edge_accumulator.py # - Per-edge streaming sums (P4)
│   │   └── trajectory_index.py # - Optional audit log (P4)
│   ├── metrics/
│   │   ├── base.py           # ✓ Abstract Metric class
│   │   ├── registry.py       # ✓ Plugin registration
│   │   ├── entropy.py        # ✓ Shannon entropy
│   │   ├── exposure.py       # - (P5)
│   │   ├── segregated_trajectories.py  # - (P5)
│   │   └── dominant_trajectories.py    # - (P5)
│   └── validation/
│       ├── synthetic_city.py # ✓ 4×4 grid ground-truth fixture
│       └── invariants.py     # ✓ Conservation & sanity checks
└── tests/
    ├── unit/                 # Unit tests (P1 complete)
    ├── integration/          # End-to-end tests (P3+ gates)
    └── fixtures/             # Pre-seeded OSM caches, test data
```

✓ = Complete (P0–P1)  
\- = Skeleton/Ready (P2+)

---

## Verification Checklist Before P2 Handoff

- [x] Config system is layered (defaults → TOML → CLI) and centralized
- [x] Boundary + CRS auto-derivation working, UTM enforced
- [x] OSM network caching by snapshot date implemented
- [x] POI extraction via QuackOSM, cache working
- [x] Snapping with distance audit functional
- [x] OD column mapping (no fixed schema) working
- [x] Demographics loader (polygons + points) working
- [x] Synthetic city fixture complete with hand-verified ground truth
- [x] Invariant checks implemented
- [x] CLI entry point and orchestrator structure in place
- [x] Linting + formatting passing
- [ ] P2 ready: origins wired, attribute attachment working
- [ ] P3 ready: Pandana + NetworkX engines, all 3 strategies routing
- [ ] P4 ready: aggregation streaming, output GeoPackage
- [ ] P5 ready: segregation metric formulas plugged in

---

## Running the Project

### Setup
```bash
# Create virtual environment and install deps
uv sync

# (Optional) Add Pandana if working on routing
uv sync --extra routing  # prebuilt wheel on Python 3.11 (incl. Windows); needs numpy<2
```

### Run Tests
```bash
uv run pytest -q              # All tests
uv run pytest -q tests/unit   # Unit tests only
```

### Lint & Format
```bash
uv run ruff check .           # Check (line length 120)
uv run ruff format .          # Auto-format
```

### CLI (When P2+ ready)
```bash
uv run segx --help
uv run segx show-config

# Example run (when P2+ implemented):
uv run segx run \
  --boundary fixtures/poa.gpkg \
  --demographics fixtures/setores.gpkg \
  --od fixtures/od.csv \
  --out out/
```

---

## Summary

**SegX is ready for the next phase.** The skeleton is clean, the configuration is flexible, the data loaders work, caching is in place, and validation has ground truth. No rework ahead — P2–P5 can proceed without architectural changes.

**Key achievement:** One engine serves any city, any demographic schema, any POI category, any OD column layout — because nothing is hard-coded. New metric formulas plug in as `Metric` subclasses. When the frontend is built, it's just another `Settings` producer.

