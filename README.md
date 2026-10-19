# SegX — dynamic spatial segregation engine

Engine for **Segregação Espacial Dinâmica** (dynamic spatial segregation). It routes
residents' daily trips over the street network and deposits their demographic
composition (income, race, age, …) onto every street segment they pass through.
Segregation can then be measured as **experienced exposure along trajectories**, not
only as residential composition.

## What it does

```text
boundary + demographics (+ OD flows)
   → street networks (walk, drive, optional public transport) from OSM, cached
   → origins with a demographic vector per class
   → trajectories: nearest POI · all POIs within a budget · OD pairs
   → per-segment sums of people per class, per (mode, category, binding)
   → segregation metrics
   → segments.gpkg (QGIS) + metrics.xlsx
```

**Status:** the whole pipeline runs end to end, writers included. That covers
ingest, origins, routing (three strategies, with k alternative routes per OD pair
available), public transport (GTFS or OSM), aggregation, every metric below, and the
output writers. Still to come: a default modal split for OD files with no mode
column, alternative routes through a full run, and city-scale memory use.

## Install

Requires [uv](https://docs.astral.sh/uv/). Python is pinned to **3.11** and numpy
to **<2**, because `pandana` (the routing backend) only ships prebuilt wheels,
including Windows, for that combination.

```bash
uv sync                 # creates .venv and installs everything (pulls Python 3.11 if needed)
uv run segx --help
```

## Run it on your data

```bash
uv run segx run -b boundary.gpkg -d sectors.gpkg --od flows.csv -c segx.toml -o out/
```

`-b` is the study-area polygon, `-d` the demographics, `--od` the optional OD
pairs, `-c` the optional config file, and `-o` the folder for the deliverables.

| input | required | what it is |
| --- | --- | --- |
| boundary | yes | study-area polygon, any CRS. It fixes the working CRS: the UTM zone of the area |
| demographics | yes | census polygons or residence points with one count column per class (`race_branca`, `race_preta`, …), or just a geometry/ID lookup when the classes live on the OD file |
| OD pairs | no | CSV or GPKG. Endpoints given as geometry, WKT, lon/lat or a block ID; an optional `mode` column per flow. Column names come from the config |
| GTFS feed | no | `transit.gtfs_path`, when `transit.enabled = true`. Without one, OSM is used and the headways are *assumed* (the output says so) |

The street network and POIs are fetched from OpenStreetMap on the first run and
cached under `cache/`. **Set `network.osm_snapshot_date`** for reproducible results;
it is stamped into the outputs.

## Outputs

| file | contents |
| --- | --- |
| `segments.gpkg` | one line layer per mode and track type (`walk_street`, `drive_street`, `transit_street`, `transit_rail_line`). Per segment: people per class, `n_paths`, and every per-segment metric. Two extra tables: `segx_metadata` (provenance) and `segx_columns` (what each column means) |
| `metrics.xlsx` | network-wide indicator of each metric; per-group tables (exposure, cross-mode exposure, fragmentation); OD destination spread; audits (mass conservation, reachability, where the modes meet, fragmentation coverage) |
| `od_spread_*.csv` | destinations per origin per class, when the OD file carries per-class columns |

Both files carry the run's **provenance**: OSM snapshot date, transit source
(`gtfs` or `osm_assumed`) and service window, matching and scale parameters, which
metrics ran, and the full resolved configuration. Two runs' numbers are
indistinguishable once they are rows in a table, and this is what tells them apart.
A missing value is empty (`NULL`), never `0`.

## Metrics

Each metric is a plugin (`segx/metrics/`). Any of them can be switched off with
`metrics.disabled`, and a switched-off metric produces no column and no sheet.

| metric | grain | what it measures |
| --- | --- | --- |
| `diversity` | segment | Shannon diversity `Pₜ·H` (Netto 2018, fig. 5) |
| `experienced_segregation` | segment | `1 − H/log₂ n`: social homogeneity of the street |
| `intergroup_contact` | segment | probability that two people on the segment belong to different groups |
| `dominant_trajectories` | segment | dominant group, and by how much it dominates |
| `segregated_trajectories` | segment | location quotient; segments exclusive to one group |
| `exposure` | group | diversity each group meets along its trajectories |
| `cross_mode_exposure` | group | the same, also counting other modes on the same street |
| `path_fragmentation` | group | fractal dimension of trajectories (box counting), plus detour and length |

Formulas and a hand-checked example: [`reports/metric-formulas.md`](reports/metric-formulas.md).

## Invariants

The architecture is shaped by seven rules; code comments refer to them by number.

1. **One enforced CRS.** Geometry may enter in any CRS but is reprojected on load to
   the working CRS (the UTM zone of the boundary); nothing downstream touches lat/lon.
2. **"Unreachable" is a typed result, never a silent zero.** A missing route carries a
   reason code, metrics propagate `NaN`, and `0.0` means "all mass in one class", not
   "no data".
3. **Nothing demographic or column-related is hard-coded.** Attribute names, class
   bins, POI categories, OD column names and budgets all come from the configuration.
4. **OSM reproducibility.** OSM extracts are cached by bounding box, snapshot date and
   network type; the snapshot date is written into the outputs and nothing is
   silently re-downloaded.
5. **Metrics are plugins.** Each measure is a registered `Metric` subclass and never
   touches routing or aggregation.
6. **Aggregation is streaming, partitioned by `(mode, category, binding)`.** No
   per-trajectory storage in the hot path.
7. **An approximate transit run is labelled, never disguised.** OSM has no
   timetables, so an OSM-derived run is stamped `transit_source = "osm_assumed"`.
   Inferred properties of an edge (its `layer`, its street-space `corridor`) travel
   as columns, never as partition keys, and the parameters behind every inference are
   stamped into the outputs.

## Configuration

Nothing demographic or column-related is hard-coded. One configuration object is
built in layers, and the last one wins: **built-in defaults → TOML file (`-c`) →
`--set key=value`**.

```bash
uv run segx show-config                                    # the resolved configuration
uv run segx show-config -c segx.toml --set poi.catchment.budget_minutes=45
```

A complete, commented example for LODES commute data is in
[`demo/segx_lodes.toml`](demo/segx_lodes.toml). The defaults and every option, with
descriptions, are in [`segx/config.py`](segx/config.py).

## Other commands

```bash
# OD destination spread only: no OSM, no routing; seconds on a city-scale OD file
uv run segx od-report -d blocks.gpkg --od od.csv -c segx.toml -o out/

# JSON-RPC-over-stdio sidecar, for embedding the engine in the desktop app
uv run segx-rpc
```

The RPC protocol is defined in [`segx/rpc_protocol.py`](segx/rpc_protocol.py), with
a generated TypeScript mirror in [`contracts/segx-rpc.ts`](contracts/segx-rpc.ts).
`describe` returns the configuration JSON schema, so the app builds its settings
UI from the engine.

## Demo: Philadelphia, end to end

The real-city showcase: OSM network, LODES commute flows, SEPTA GTFS. It renders
maps and charts and writes the same output files to `demo/out/deliverables/`. Flags
and outputs are documented in [`demo/README.md`](demo/README.md). The Philadelphia
input files are not included in this repository; any LODES OD file with matching
census block geometry works.

```bash
uv sync --group demo    # once: matplotlib + contextily for the maps

uv run python demo/run_demo.py \
    --geometry philadelphia_blocks.shp --od philadelphia_OD.csv \
    --bbox="-75.165,39.945,-75.150,39.957" \
    --od-only --transit --gtfs demo/out/inputs/gtfs_public.zip
```

Write `--bbox=` **with the equals sign**; otherwise the leading `-` is read as a
flag.

## Development

```bash
uv run pytest -q                              # full suite, runs offline
uv run ruff check . && uv run ruff format .   # lint + format (line length 120)
uv run python scripts/gen_rpc_types.py        # regenerate the TS contract (--check in CI)
```

The tests run against a hand-computed synthetic city
(`segx/validation/synthetic_city.py`) and canned OSM/GTFS fixtures, never the
network. The NetworkX router is the oracle that `pandana` is checked against.

## Repository layout

| path | contents |
| --- | --- |
| `segx/io/` | loaders (boundary, demographics, OD, POIs, GTFS) and the output writers |
| `segx/network/` | per-mode graphs and their cache, public transport, the shared street space across modes |
| `segx/population/` | origins and their demographic vectors |
| `segx/routing/` | nearest POI, catchment and OD strategies; pandana and NetworkX engines; route sets |
| `segx/aggregation/` | streaming per-segment and per-path sums |
| `segx/metrics/` | the metric plugins |
| `segx/analysis/` | analyses of the input tables (OD destination spread) |
| `segx/validation/` | synthetic city and invariant checks |
| `segx/orchestrator.py` | the pipeline, stage by stage |
| `reports/` | technical write-ups per phase: what was built, how it was verified, measured results |

## License

SegX is licensed under the [Apache License 2.0](LICENSE).

Note on dependencies: the default routing backend, [`pandana`](https://github.com/UDST/pandana),
is licensed under the **AGPL-3.0**, and `igraph` under the GPL. Using SegX as-is is
unaffected, but anyone *redistributing* SegX bundled together with `pandana` (for example,
in a packaged application) must also comply with the AGPL-3.0 for the combined work.
