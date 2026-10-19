# SegX — P2: building the origins

**Phase:** P2 — located, demographically labelled origins.

---

## 1. One-line summary

The engine can now **read a city's people and turn them into located, demographically labelled origins** — exactly the input the routing and segregation maths consume. It already ingests a **real commuting-flow CSV** (home→work, with worker counts) and produces, for every census block, a point on the map carrying that block's demographic profile.

P2 is the bridge between "the data is loaded" (P1) and "trajectories can be routed" (P3).

---

## 2. What P2 implements

Three behaviours, all configurable (nothing demographic is hard-coded):

1. **Choosing the origins.**
   - If the user supplies **origin→destination pairs** (e.g. commuting data), the origins are exactly the distinct origins of that file — none are invented.
   - Otherwise, the origins are the **centroids of the census polygons** (one labelled point per block).
   - If the demographic input is already individual **residence points**, they pass through unchanged.

2. **Placing every origin correctly on the map.** A census block is an area; a trajectory needs a point. The block's centroid is used when it falls inside the polygon, and a guaranteed interior point otherwise (for irregularly shaped blocks) — so every origin lands on valid ground.

3. **Attaching the demographic profile.** Every origin receives its vector of demographic classes (e.g. income bands), read from the data, ready for the segregation maths. Crucially, an origin with **no demographic match** is flagged explicitly — "no data" is never silently treated as "zero", because that would corrupt every downstream metric.

---

## 3. Support for LODES-style flow files

P2 was validated against a **LODES-style employment origin–destination file**:

| Column | Meaning |
|---|---|
| `hGEOID` | census block of residence |
| `wGEOID` | census block of work |
| `SE01 / SE02 / SE03` | low / mid / high income worker counts |

This file has two features the generic loader did not yet handle, so P2 added support for both:

- **Origin and destination are *IDs*, not coordinates.** `hGEOID`/`wGEOID` point at the census geometry. The engine resolves every ID to its block's location automatically. All flows leaving the same residence block consolidate into a single origin.
- **The demographics live on the flow rows, not in a separate file.** `SE01/SE02/SE03` are the income distribution. The engine sums them per residence block to get that block's residential income composition. Column names are configurable — the engine reads them from the configuration and never assumes `SE01`.

The only fixed assumption: **there is always an origin-ID column and a destination-ID column** (under any name) that reference the census geometry.

**Verified example** (automated test): two flows leave block `100` —
`SE01/02/03 = 5/2/0` and `3/1/1` — and the engine produces block `100`'s residential vector
`{low: 8, mid: 3, high: 1}`, located at block 100's point.

---

## 4. Defaults

1. **Blocks outside the study area.** Every `hGEOID`/`wGEOID` must exist in the supplied geometry, otherwise it is an error. Real commuting data often has *work* blocks outside the city limit; the alternatives would be to drop or flag them.
2. **A block's demographics.** The residence-side worker total is used (summed over all destinations), reading `SE01/02/03` as low/mid/high.
3. **Income / age / race class boundaries** are provisional — the engine is fully schema-driven, so changing them is a configuration change, not a code change.

---

### Technical appendix
- New/changed engine code: `population/origins.py`, `population/attributes.py`, `io/_geometry.py`, plus the configuration seams (`attribute_source`, `class_columns`, `origin_id_ref`/`dest_id_ref`) and the orchestrator wiring.
- Invariants kept: a single enforced projection; "no data" is flagged, never zeroed; everything demographic/column-related comes from configuration.
- Tests: `tests/unit/test_origins.py`, `tests/unit/test_attributes.py` (incl. the LODES walkthrough), plus the updated orchestrator tests.
