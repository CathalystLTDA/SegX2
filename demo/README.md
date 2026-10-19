# SegX — end-to-end demo (real city, LODES data)

Shows the engine **running for real** on a US city: real street network (OSM) +
real POIs (OSM) + **real public transport (GTFS)** + real home→work flows (LODES)
→ located, income-labelled origins → routed trajectories (nearest, 30-min
catchment, and the home→work commute on foot / by car / by bus / by metro) →
per-segment aggregation → the P5 segregation measures and the P7.1 **cross-mode
exposure**. It produces PNG maps, charts, CSV tables and a console summary.

The demo calls the pipeline stages directly (`ingest` / `build_population` /
`route` / `aggregate`) so it can draw intermediate results, and also writes the
same output files as `segx run` (see *Outputs*).

## Data (example: Philadelphia)

The demo runs on two input files (plus, optionally, a GTFS feed). **The
Philadelphia files are not included in this repository**; any LODES OD file with
the matching census block geometry works.

**`philadelphia_OD.csv`** — LODES-style home→work flows, with the columns:

| column | meaning |
| --- | --- |
| `hGEOID` | census block of **residence** (15-digit GEOID) |
| `wGEOID` | census block of **work** |
| `SE01` / `SE02` / `SE03` | **low / mid / high** income workers |

There is no total column — the runner synthesises the weight `S000 = SE01+SE02+SE03`.

**`philadelphia_blocks.shp`** (+ `.shx/.dbf/.prj`) — block polygons with a
**`GEOID`** column that matches the OD geocodes. It is only an id→point
**lookup**; the income numbers come from the OD file.

Column names live in [`segx_lodes.toml`](segx_lodes.toml) — for another dataset,
edit them there (or use `--set`), without touching the code.

**GTFS (optional, for `--transit`)** — the agency's feed. Accepts a normal GTFS
zip, an already extracted directory, **or a zip of zips** like SEPTA's
`gtfs_public.zip` (which ships `google_bus.zip` + `google_rail.zip`). The runner
unpacks it, prefixes ids per sub-feed (bus `route_id` "1" does not collide with
rail's), thins it to the study area and writes **one** feed to
`out/inputs/gtfs_feed/`. Thinning keeps **whole trips** — every trip that touches
the bbox comes in with its complete stop sequence — so the engine can still
*split* a line that leaves the area and comes back. It is cached by the source
zip's hash + bbox.

## The travel-mode column (`mode`)

LODES is a **flow** table: it says how many workers go from block A to block B,
never **how**. Without that information every flow falls into the default mode
(`commute = car`), the engine runs with a single mode and the
`cross_mode_exposure` metric becomes an **exact copy** of `exposure` — the
dangerous failure mode, because it does not show up as zero, it shows up as "the
modes never meet".

That is why the runner **synthesises** the column with `--mode-split`:

```bash
--mode-split "car=0.45,bus=0.20,subway=0.15,walk=0.20"   # the default
```

- The labels are **travel-survey** labels, not graph names:
  `segx.orchestrator._OD_MODE_ALIAS` translates `car → drive` and
  `bus`/`subway`/`metro`/`pt`/`onibus` → `transit`. The file reads as transport
  data; the translation is the engine's job.
- The assignment is a **seeded hash of the (origin, destination) pair**, not a
  per-row draw: the same home→work pair always gets the same mode, regardless of
  `--bbox`, row order or how many flows came with it. Two runs are comparable; so
  are a clip and the whole city.
- **It is invented, so it is labelled.** The shares, the seed and the observed
  counts go to `out/inputs/od_clip_modes.json`, and the console says
  `SYNTHETIC`. A modal figure coming out of this describes the **engine**, not
  Philadelphia.
- If the source OD file **already has** the column (`od_columns.mode`, default
  `mode`), it passes through untouched and nothing is invented.

The result is written to `out/inputs/od_clip.csv`:

| wGEOID | hGEOID | SE01 | SE02 | SE03 | S000 | mode |
| --- | --- | --- | --- | --- | --- | --- |
| 421010001012003 | 421010002002009 | 0 | 0 | 1 | 1 | subway |
| 421010006002008 | 421010006002009 | 0 | 0 | 1 | 1 | car |
| 421010005004011 | 421010006002009 | 0 | 0 | 1 | 1 | bus |

A file with `bus`/`subway` **requires** `--transit`; without it the runner refuses
immediately (before downloading OSM) and says what to do.

## How to run

```bash
uv sync --group demo        # once (installs matplotlib + contextily for the maps)

uv run python demo/run_demo.py \
    --geometry philadelphia_blocks.shp \
    --od       philadelphia_OD.csv \
    --bbox="-75.165,39.945,-75.150,39.957"     # Center City clip
```

> **Use `--bbox=` with the equals sign.** The longitude starts with `-`, which
> without the `=` argparse would read as a flag.

### OD pairs only, with public transport (what P7/P7.1 added)

```bash
uv run python demo/run_demo.py \
    --geometry philadelphia_blocks.shp \
    --od       philadelphia_OD.csv \
    --bbox="-75.200,39.935,-75.140,39.975" \
    --od-only \
    --transit --gtfs demo/out/inputs/gtfs_public.zip
```

> **Why this clip and not Center City.** The small clip
> (`-75.165,39.945,-75.150,39.957`, ~1.3 km across) is too short for public
> transport to compete with walking: of the 434 `transit` trips, 385 walked the
> whole way and only 7 took a bus — each one alone on its street, hence zero
> entropy. `exposure` for the `transit` mode came out 0.17 / 0.45 for mid / high
> income and described seven passengers, not the network. The clip above
> (~5.1 × 4.4 km, from University City to the Delaware; 2,556 blocks, 31,614
> flows) gives the mode enough mass to be read.

- **`--od-only`** runs **only** binding C (home→work). It turns off the access
  bindings (`nearest` and `catchment`) *and* the POI fetch — no category is
  requested, so nothing is downloaded from Overpass. It is also the only binding
  where `cross_mode_exposure` is non-trivial: the access bindings use a single
  `routing.access_mode`, so they contribute one mode only.
- **`--transit`** turns on public transport as a third mode (merged walk ⊕
  transit graph) and implies `--mode-split` (otherwise the graph is built and
  never used).
- **`--gtfs`** points at the feed. **Without it the run falls back to OSM**, whose
  *headways* are **invented** from `transit.osm_assumed_headway_min` — the result
  is stamped `transit_source = "osm_assumed"` and the console warns in capitals.
- **`--service-window "06:00-09:00"`** is the window in which headways are
  measured (config default). `headway = window_minutes / n_trips`, so a **wider**
  window **raises** the average headway and makes transit look worse — run two
  windows and compare, never read one average on its own. `end <= start` crosses
  midnight (`"22:00-02:00"`).
- **`--cross-mode-factor 0.5`** is the weight of *another* mode's mass in a mode's
  exposure. `0` reproduces `exposure` bit for bit; `1` pools every mode into a
  single crowd.

- **`--disable-metric NAME`** (repeatable) turns a metric off, e.g.
  `--disable-metric path_fragmentation`. It is a shortcut for
  `--set metrics.disabled=[...]`; that metric's console block, maps, charts and
  tables are skipped. Turning fragmentation off also skips rasterising every path,
  which is the expensive part of aggregation (65 s → 18 s on the 31.6k-flow clip).

Maps come out in a **dark theme** over a **city basemap** (streets + building
footprints), via CartoDB DarkMatter tiles (`contextily`), reprojected to the
working CRS. Tiles are downloaded the first time (contextily caches them). To run
**offline**, pass `--no-basemap` — the maps still come out on the dark background,
just without the basemap.

Block-level LODES for the whole city is large (1.28 M rows) and slow — `--bbox`
(WGS84: `minx,miny,maxx,maxy`) clips a neighbourhood and keeps only the flows whose
**both** ends fall inside the clip, keeping the demo quick.

The first run **downloads OSM** (network + POIs) and caches it in
`demo/out/cache/`; later runs are fast. The OSM snapshot date is pinned in the
config (`network.osm_snapshot_date`) for reproducibility.

## Outputs (in `demo/out/`)

Output files (P8, in `out/deliverables/`) — the same files `segx run` writes:

- `segments.gpkg` — opens in QGIS. One line layer per mode and layer
  (`walk_street`, `drive_street`, `transit_street`, `transit_rail_line`): mass per
  class, `n_paths`, `corridor`/`length_m` and every per-segment metric. The
  `segx_metadata` (provenance: OSM date, transit source, window, parameters) and
  `segx_columns` (what each column is) tables come with it.
- `metrics.xlsx` — each metric's network indicators, the per-group tables
  (exposure, cross-mode exposure, fragmentation), `od_spread` and the audits.
- `od_spread_by_origin.csv` / `od_spread_overlap_by_origin.csv` — only when the OD
  file carries per-class columns.

Maps (PNG):

- `segx_demo_map.png` — overview: network, origins by **dominant** income, POIs
  and a **sample** of routes from the three bindings (blue/green/red). It is
  sampled (~20 per binding) only so it does not turn into a tangle.
- `segx_nearest.png` — **all** nearest routes, coloured by POI category.
- `segx_catchment_<category>.png` — one per category (parks, supermarkets,
  health_centers, bus_stops): **all** catchment routes to that category.
- `segx_od_<low|mid|high>_income.png` + `segx_od_all_income.png` — the home→work
  commutes only, per income class and summed; with origins (home, ○) and
  destinations (work, ▲).
- `segx_od_by_mode.png` — the same commutes coloured by **travel mode** (only
  produced when the OD file has more than one mode). Each route is drawn on **its
  own** graph: `walk` on the pedestrian graph, `car` on the car graph, `transit`
  on the merged one — the bus rider's walking legs, the bus stretch *map-matched*
  onto the avenue, and the metro on its own `rail_line` layer.

P5 metrics — the five segregation measures, one output each (in `out/segments/`,
`out/charts/` and `out/tables/`; OD binding):

- `segments/experienced_segregation.png` — **experienced segregation** `1 − h`
  per segment (heat map; light = mixed, hot = homogeneous).
- `segments/intergroup_contact.png` — **intergroup contact probability**
  `1 − Σ pᵢ²` (Gini–Simpson) per segment.
- `segments/dominant_trajectories.png` — **dynamic segregation**: the dominant
  group colours the segment, **opacity = dominance** `1 − norm. entropy` (a
  near-tied segment fades out; a homogeneous one is solid). Distinct from P4's
  `dominant_income.png`, which only paints the winner without the weight.
- `segments/segregated_segments.png` — **most exclusive segments**: only those with
  `LQ ≥ metrics.exclusive_lq_threshold` (default `2`), coloured by group.
- `charts/exposure_by_group.png` — **social exposure**: one value per group (not a
  map) — the mean diversity each group meets along its trajectories. **One panel
  per mode**: `mode` is a partition key, so each mode has its own value.
- `tables/<metric>_by_segment.csv` + `tables/<metric>_summary.csv` — each measure
  per segment and the matching network indicator.

P7.1 metric — **cross-mode exposure** (`out/charts/` and `out/tables/`):

- `charts/cross_mode_exposure.png` — `exposure` **vs** `cross_mode_exposure` side
  by side, one panel per observer mode. Always read the **pair**, never the second
  number alone.
- `tables/cross_mode_exposure_by_segment.csv` + `_summary.csv`.
- `tables/cross_mode_audit.json` — the **join audit** (how many street segments
  carry more than one mode). It travels with the numbers because a join that found
  nothing does **not** show up as zero: it shows up as
  `cross_mode_exposure == exposure`, which reads as the finding "the modes never
  meet". Without the audit the two cases cannot be told apart.
- `tables/cross_mode_segments_by_combination.csv` — the same audit **broken down
  by exact mode combination** (`walk`, `drive+walk`, `drive+transit+walk`, …), one
  row each, with the layer. It is what lets you *diagnose* a low join rate: 36 % of
  single-mode segments means one thing if they are all `walk` (local pedestrian
  streets, a real finding) and something very different if they are split across
  the three modes (three id spaces that did not match). The `rail_line` segments,
  which pooling excludes on purpose, show up here instead of disappearing.

P7.3 metric — **path fragmentation** (`out/charts/` and `out/tables/`):

- `charts/path_fragmentation.png` — per group and per mode, the fractal dimension
  `D` (top) and the mean number of street segments per path (bottom). The two rows
  go together on purpose: `D` measures **how the winding changes with scale**, not
  how winding the path is — a staircase of blocks scores almost the same as a
  straight line. The `D` axis **does not start at zero**.
- `tables/path_fragmentation_by_group.csv` — one row per (mode, binding, class):
  `fragmentation`, `fragmentation_sd`, `mean_segments`, `mean_length_m`,
  `mean_extent_m`, `detour_ratio`, `n_paths`, `n_measured`.
- `tables/path_fragmentation_audit.json` — the **coverage** (paths measured vs.
  too small for the scale ladder) and the method's **parameters**. The default is
  the paper's box counting (0.7 m × 5ᵏ pixels, six scales); two `D` values
  measured with different parameters are not comparable.

Console summary: number of origins/POIs/trajectories and **reachability per
binding** (unreachable is always counted, never turned into zero — invariant #2),
the **P4** block (aggregation + diversity + conservation), the **P5** block (the
five network indicators + exposure per group + exclusive segments, **broken down
by mode**), the **P7.3** block (fragmentation: method, coverage and the per-group
table) and, with `--transit`, two more blocks:

- **P7 (public transport)** — the first line is the *source stamp*: GTFS
  (measured timetable) or **`osm_assumed`** (invented headways). Then the service
  window with the reminder that `headway = window_minutes / n_trips`, the size of
  the merged graph (platforms / rides / transfers), stops without a street
  connector, and surface rides that did **not** match the street (those deposit no
  mass).
- **P7.1 (cross-mode exposure)** — the join rate **before** the numbers, and an
  explicit warning if several modes are present but no segment is shared.

What to expect from the P7.1 block and **not** mistake for a bug:
`cross_mode_exposure` can come out **below** `exposure` (if the incoming mode is
more concentrated), and the effect is strongly **asymmetric** — in the Center City
reference run `walk` rises a lot (1.02 → 1.15) and `drive` barely moves
(1.20 → 1.21), because a few pedestrians do not move the count of an avenue full
of cars, but the reverse does.

## Quick tweaks (`--set`, same as the CLI)

```bash
--set demographics.polygon_id_column=GEOID10       # TIGER 2010
--set poi.catchment.budget_minutes=45              # 45-min catchment
--set routing.engine=networkx                      # slow oracle (small clips)
--set transit.max_wait_min=20                      # cap on the modelled wait
--set metrics.cross_mode.overrides='{"transit|walk":0.7}'   # per mode-pair exception
--set metrics.fragmentation.as_written=true        # D = 1 − slope (paper's literal form; line = 2)
--set metrics.fragmentation.drop_saturated=false   # fit all 6 scales, even the 1-pixel ones
--set metrics.fragmentation.pixel_m=3.5            # coarser pixel (faster)
--set metrics.fragmentation.method=divider         # the earlier method, for comparison
```

The `--transit`, `--od-only`, `--service-window` and `--cross-mode-factor` flags
are only shortcuts for config keys, and an explicit `--set` always wins.

The full configuration is in [`segx_lodes.toml`](segx_lodes.toml), commented.

## Caveats of this demo

- **The `mode` column is synthetic.** See the section above. It exists to exercise
  the multimodal engine; any *modal* conclusion drawn here is about the code, not
  about Philadelphia. With real modal data, just swap the column.
- **The GTFS calendar can overlap service periods.** SEPTA publishes several
  weekday `service_id`s with different date ranges; `active_service_ids` accepts
  every one that marks Monday, which adds up trips from distinct periods and
  **shortens** the average headway. This is engine behaviour, not specific to this
  demo.
- **`pandana` is not bit-for-bit deterministic across runs.** Contraction
  hierarchies are built on 16 threads and tie-breaking between equal-cost paths
  varies, so segment counts fluctuate ~0.1 % between identical runs (3865 vs. 3861
  segments). Use `--set routing.engine=networkx` for a reproducible run on small
  clips.
