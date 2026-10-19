# SegX — Output writers (P8): the run's output files

**Status:** **implemented and tested** on the synthetic validation city (434 tests passing,
lint clean). **Not yet run** on the Philadelphia clip with the writers enabled — see §6.
**Related documents:** [MetricFormulas.md](MetricFormulas.md) (what each metric computes),
[P8PathFragmentation.md](P8PathFragmentation.md) and
[P9ODDestinationSpread.md](P9ODDestinationSpread.md) (the two most recent measures, which are
also written to these files).

---

## 1. One-line summary

Until now the engine computed everything, but the results only existed in memory and in the
demo's charts. A run now ends by writing **two files**: a **GeoPackage** with the street segments
and their metrics, which opens directly in QGIS, and a **spreadsheet** with the indicators, the
per-group tables and the audits. Both carry the run's **provenance**: how old the map is, where
the transit timetables came from, and with which parameters each number was computed.

---

## 2. How to generate them

```bash
uv run segx run -b boundary.gpkg -d blocks.gpkg --od flows.csv -c config.toml -o out/
```

A future desktop app works the same way: it calls the engine through the same door (the
JSON-RPC *sidecar*) and gets back the list of written files.

| file | contents |
|---|---|
| `segments.gpkg` | the street segments, one layer per mode, with masses and metrics |
| `metrics.xlsx` | network indicators, per-group tables, OD destinations and audits |
| `od_spread_by_origin.csv`, `od_spread_overlap_by_origin.csv` | destinations per origin (only when the OD file carries per-class columns; at whole-city scale they exceed a sheet's row limit) |

---

## 3. `segments.gpkg` — the map

### 3.1 Layers

One line layer per **mode** and per **kind of way**:

| layer | what it shows |
|---|---|
| `walk_street` | people on foot, on sidewalks and streets |
| `drive_street` | people driving |
| `transit_street` | bus passengers, on the avenue the bus runs along |
| `transit_rail_line` | metro/rail passengers, on the rail line |

The metro gets **its own layer** on purpose: a passenger in the tunnel is not on the avenue
above, and the two are never drawn on top of each other. Only the layers a run produced appear.

### 3.2 Columns

Each row is **one street segment** within a combination of category (e.g. `commute`) and
binding (`od`, `nearest`, `catchment`).

| group | columns | example |
|---|---|---|
| identification | `mode`, `category`, `binding`, `u`, `v`, `key` | `walk`, `commute`, `od`, … |
| segment properties | `layer`, `corridor`, `length_m`, `n_paths` | `n_paths` = trajectories passing there (line thickness) |
| mass per class | `{attribute}_{class}` | `income_low`, `income_mid`, `income_high` |
| per-segment metrics | `{metric}_{attribute}` | `diversity_income`, `experienced_segregation_income`, `intergroup_contact_income` |
| multi-value metrics | `{metric}_{value}_{attribute}` | `dominant_trajectories_dominance_income`, `segregated_trajectories_lq_income`, `segregated_trajectories_exclusive_income` |

The two measures that name a "dominant class" are written under distinct names, because they
**are not the same computation**: `dominant_trajectories_dominant_class_*` is the class with the
most people on the segment; `segregated_trajectories_dominant_class_*` is the class with the
highest location quotient.

The `segx_columns` table, inside the file itself, documents every column (which metric it comes
from, which attribute, which class). Names are generated from the configuration: a demographic
schema with race or age band produces the matching columns with no code change.

**Empty is not zero.** A segment with no value for a metric is left blank (`NULL`), never `0`.
In experienced segregation, for example, `0` means "perfect mixing", not "no data".

### 3.3 Measures that are not per segment

Social exposure, cross-mode exposure and fragmentation are **one value per group**, not per
segment. They therefore are not map columns: they are in the spreadsheet (§4).

---

## 4. `metrics.xlsx` — the numbers

| sheet | contents |
|---|---|
| `metadata` | the provenance (§5) |
| `columns` | the GeoPackage column dictionary |
| `diversity`, `experienced_segregation`, `intergroup_contact`, `dominant_trajectories`, `segregated_trajectories` | each per-segment metric's network indicator, per mode |
| `exposure`, `cross_mode_exposure`, `path_fragmentation` | the per-group table, and next to it (`*_summary`) the summary |
| `od_spread_summary`, `od_spread_overlap` | destinations per class and overlap between classes (see P9ODDestinationSpread.md) |
| `audit_conservation` | people emitted = deposited + unrouted; the balance must be 0 |
| `audit_counts`, `audit_partitions` | how many segments and trajectories, in which combinations |
| `audit_reachability` | how many origins/flows got no route, and why; points too far from the network |
| `audit_cross_mode`, `audit_cross_mode_sets` | where the modes actually meet (e.g. `drive+walk`), per pair and per combination |
| `audit_fragmentation` | how many paths were measured and how many were too small for the scale |
| `audit_output` | rows per layer and segments written without geometry |

The audits exist because failures in this kind of analysis **do not show up as errors**. They
show up as a plausible-looking number: a cross-mode exposure equal to plain exposure, or an
empty fragmentation. The sheets say whether the number was actually measured.

A metric turned off in the configuration **produces no column and no sheet**, and the `metadata`
sheet records that it was not run. That way it cannot be mistaken for missing data.

---

## 5. Provenance — what is stamped on both files

Results from two different runs become indistinguishable once they are rows in a table. The
stamps are what keep them apart. They go into the GeoPackage's `segx_metadata` table, the
spreadsheet's `metadata` sheet and each layer's metadata (visible in QGIS under *Layer
Properties → Metadata*):

| stamp | why it matters |
|---|---|
| `osm_snapshot_date` | OpenStreetMap changes; the same city on another date is another network. It is `null` when the date was not pinned, and that is stated |
| `transit_source` | `gtfs` (measured timetables) or `osm_assumed` (no timetables, headways **assumed** from configuration). A map with invented headways must not pass for a measured one |
| `transit.transit_service_window_bounds_min` | the service window used for headways (default 06:00–09:00). A wider window makes transit look worse |
| `street_space_params` | tolerance (20 m) and angle (30°) of the sidewalk↔roadway matching: a geometric inference, not data |
| `fragmentation_params` | method and scales of the fractal dimension; two Ds measured at different scales are not comparable |
| `metrics_enabled`, `metrics_disabled` | what was and was not computed |
| `working_crs`, `routing_engine`, `route_set_k`, … | coordinate system and routing options |
| `settings_json` | the run's full configuration, to reproduce it |

---

## 6. What was verified and what remains to verify

**Verified (synthetic city, automated tests):**

- Round trip: what is written and read back equals what the engine computed, segment by
  segment, in masses and metrics.
- The metro goes to its own layer and the street to another; every segment gets geometry.
- Provenance is in both files and in the layer metadata.
- A disabled metric does not appear; a segment without a value is blank, not zero.
- Two identical column names (e.g. an attribute named so that it collides with another) stop
  the run with a clear error instead of one column overwriting the other.
- A full run from the command line and from the app *sidecar* writes the files.

**A defect found and fixed during implementation:** the engine stores each segment without a
direction of travel, while the metro line in the graph is one-way. In the first version this
made **every metro stretch come out without geometry**. It is fixed and covered by a test. A
segment whose geometry is not found is still written, without a line, and counted in
`audit_output`. It is never dropped.

**Not yet verified:**

- A run on the Philadelphia clip with the writers enabled. The demo already writes the files to
  `demo/out/deliverables/`; it remains to run it and open the GeoPackage in QGIS to check layers,
  styling and whether `audit_output` reports any segment without geometry.
- Time and file size at whole-city scale.

---

## 7. Format notes

1. **OSM date.** It can currently be left unpinned, and the file then says `null`. For
   reproducible results, **pin the date** in the configuration.
2. **Formats.** GeoPackage + XLSX. Shapefile or per-layer CSV, if needed, is a simple conversion
   from the GeoPackage.
3. **One layer per mode.** Category/binding combinations are columns within each layer,
   filterable in QGIS. One layer per combination (e.g. `walk_commute_od`) would be a small
   change.
