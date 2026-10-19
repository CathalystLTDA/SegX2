# SegX — P5: segregation metrics (and P4.5: alternative routes)

**Phase:** P5 — segregation metrics; P4.5 — route sets per OD pair.

---

## 1. One-line summary

The engine now **measures segregation**: on top of P4's per-segment table it computes five measures — **experienced segregation**, **intergroup contact probability**, **social exposure**, **dynamic segregation** (dominant group) and **most exclusive segments** — each as a per-segment map **and** a single network-wide indicator.

Alongside it, **P4.5**: every origin–destination pair can now be spread over **several plausible routes** instead of stacking the whole flow onto the single geodesic route — with **two algorithms** (behavioural and exact) implemented and **compared side by side in the demo** (section 4).

---

## 2. The five measures

All are metric *plugins* over **P4's per-segment table** — **without touching** routing or aggregation (project invariant: metrics are plugins). Each respects "no data never becomes zero" (segment without mass → `NaN`, never `0.0`) and is computed **within each `(mode, category, binding)` partition**.

1. **Experienced segregation** (`experienced_segregation`) — the degree of **social homogeneity** experienced along trajectories: `1 − h`, where `h` is the normalized Shannon entropy (0 = perfectly mixed street, 1 = fully homogeneous). The network indicator is the **mean over segments**.

2. **Intergroup contact probability** (`intergroup_contact`) — the probability that two people on the segment belong to **different groups**: the Gini–Simpson index `1 − Σ pᵢ²`. Distinct from entropy: entropy measures *diversity*, this measures *probability of encounter*.

3. **Social exposure** (`exposure`) — per group, **how much diversity that group potentially meets** in the city: each segment's composition entropy, averaged over the segments the group traverses, weighted by the group's presence. Result: one value per social group.

4. **Dynamic segregation / dominant group** (`dominant_trajectories`) — per segment, the **group with the highest presence** colours the segment, weighted by a **dominance weight** `1 − normalized entropy` (it uses the whole distribution): 33/33/34 → weight ≈ 0; 0/0/100 → weight = 1.

5. **Most exclusive segments** (`segregated_trajectories`) — segments used **disproportionately by a single group**. The **unit of analysis is the street segment** (not the whole trajectory): per segment, the *location quotient* `LQ = group's local share / group's global share`; a segment is flagged **exclusive** when the dominant group's `LQ` reaches the threshold (default `2.0`). The number of exclusive segments depends on the sample size of trajectories/ODs per group and on how segregated the city is.

---

## 3. Two map/network options

- **Directed vs. undirected graph — a user choice** (`network.undirected`). When undirected, both directions of a street merge into **a single segment** (one line on the map). Routing stays correct (it rebuilds both directions internally). *Note:* an undirected **drive** graph ignores one-way streets — which is why it is an option the user turns on deliberately.
- **Line thickness = number of paths per segment** (`n_paths`). The accumulator counts how many trajectories pass through each segment — the signal for drawing busier streets thicker. Counted **independently of demographics** (a trajectory with unresolved demographics is still flow).

---

## 4. P4.5 — alternative routes per OD pair

**The problem.** Previously, all of an OD pair's flow was deposited on the **single fastest route**. That artificially concentrates demographic mass in one corridor and underestimates mixing on parallel streets — people do not all take exactly the same path.

**What was built.** An OD pair can now emit **`k` alternative routes**, each carrying a fraction of the flow. Two algorithms, **both implemented** and selected at run time (`routing.route_set.method`):

- **`link_penalty` (behavioural)** — finds the shortest route, **makes the streets it used more expensive** by a factor `penalty` (> 1) and recomputes; repeats until `k` routes. Produces **spatially distinct** alternatives (it pushes the route out of the corridor already used).
- **`yen` (exact)** — the true `k` shortest **simple** routes (Yen/Lawler algorithm). Provably the next shortest, but in practice **near-duplicates**: they differ from route 1 by one block.

**Splitting the flow.** Each route gets a **choice weight `wᵢ`** with **`Σ wᵢ = 1`** (`weight_scheme`): `uniform` (1/k), `inverse_cost` (`∝ 1/cost`) or `path_size_logit` (default — discounts routes that overlap heavily with the others). The aggregator deposits **`wᵢ × demographic vector`**, so **weight conservation still holds**: the total emitted does not change, only its spatial distribution.

**Off by default** (`route_set.k = 1`): previous behaviour is unchanged until it is turned on.

### The comparison

The demo runs **both algorithms on the same set of pairs** and measures the difference (`--compare-routesets`). Sample of **40 real Philadelphia OD pairs**, `k = 3`, `penalty = 1.5`:

| metric | `link_penalty` | `yen` (exact) |
|---|---|---|
| routes per pair (mean) | 2.18 | 3.00 |
| mean detour of the alternatives | **10.5 %** | 47.9 % |
| internal overlap (↓ = more distinct) | **0.137** | 0.444 |
| time per pair | 7.5 ms | 5.8 ms |

Reading: **link-penalty** delivers **more distinct and cheaper** alternatives (10 % detour, little overlap) — the spread that matters for segregation. **Yen** always returns `k` routes, but they overlap (0.44) and, on short trips, the "next shortest route" is an expensive detour (hence the 47.9 %). **Route 1 is identical under both methods for 100 % of pairs** (sanity check); the set of covered streets coincides at ~58 %. *On timing:* on this small clip Yen is even slightly faster — the cost difference appears at whole-city scale (573 thousand unique pairs), where Yen is infeasible outside samples.

Outputs: `maps/routeset_comparison.png` (side-by-side map — colour per route, shared stretches highlighted), `tables/routeset_comparison_by_pair.csv` and `tables/routeset_comparison_summary.csv`.

### A note on the penalty factor

In 12 of the 40 pairs link-penalty returned **only 1 route**. Eleven are very short trips (0.2 to 1.0 min) where no plausible alternative exists — the correct result. The twelfth is a 3.3-min pair that fell **just under the threshold**: the first truly distinct alternative needs `penalty ≥ 1.53` and the demo runs with `1.50`. The rule is simple — an alternative that shares a fraction `f` of route 1's cost and detours by `δ` is chosen only when **`(penalty − 1)·(1 − f) > δ`**. The `penalty` therefore controls, explicitly and auditably, **how much** detour the model accepts for a route to count as "plausible".

---

## 5. Parameters and methodological choices

**Defaults adopted:**
- Dominance weight = `1 − normalized entropy` (whole distribution, not just the margin between the top two).
- "Most exclusive" = analysis **per street segment** (location quotient), with no need to keep per-trajectory data.

**Configurable parameters whose value affects the results:**
- **Exclusivity threshold** (`exclusive_lq_threshold`, default `2.0` = the group is twice as concentrated on the segment as in the city).
- **Social exposure input** — the catchment opportunity set (every POI within the budget) or only the nearest POI.
- **Diversity formula** (from P4) — volume-weighted `D` vs. normalized `H`.
- **Route-set parameters** (all implemented and configurable; section 4's comparison informs the choice): **(a)** `k` routes per pair (2–3 is a reasonable range); **(b)** `method` — `link_penalty` (distinct alternatives, scales to the city) vs. `yen` (exact, useful as an oracle on samples); **(c)** `penalty` (1.5–2.0); **(d)** `weight_scheme` for splitting the mass (default `path_size_logit`); **(e)** whether route sets apply **only to the OD/commute binding** (as now) or also to access trajectories (nearest / catchment).

---

## 6. How we know it is correct

- **Hand-computed ground truth.** Unit tests pin values computed on paper for each measure — e.g. Gini–Simpson of `{3, 1}` = `0.375`; dominance weight of `{3, 1}` = `1 − h([3,1])`; a segment where a globally rare group (2 of 22) is 100 % local → `LQ = 11` → **exclusive**.
- **Weight conservation kept.** Nothing in the accumulator breaks the invariant `emitted = deposited + unreachable`; tests confirm that the undirected merge and the path counter preserve the sums.
- **Splitting the flow over `k` routes also conserves.** A dedicated test (for **both** methods): a pair split into `k` routes deposits exactly the same total mass as the same pair on a single route — `Σ wᵢ = 1` checked for all three weight schemes. Also tested: with `k = 1` (default) exactly **one** trajectory per pair comes out, byte for byte as before; link-penalty produces alternatives **provably more distinct** than Yen; Yen returns routes in **non-decreasing** cost.
- **Integration gate green.** The end-to-end pipeline (synthetic city, Pandana **and** the NetworkX oracle) runs through aggregation and computes every metric.

---

## 7. What P5 does **not** do

As in P4, **potential encounters in time** (co-presence within a window of minutes, space–time prism, K function) remain **out** — they require time-stamped trajectories, which the engine does not have (trajectories are timeless). P5 delivers the *structural* segregation metrics over the network.

---

### Technical appendix
- **New metrics:** `metrics/exposure.py` (`ExposureMetric` — social exposure per group), `metrics/dominant_trajectories.py` (`DominantTrajectoriesMetric` — dominant class + weight `1 − h_norm`), `metrics/segregated_trajectories.py` (`SegregatedTrajectoriesMetric` — location quotient + `exclusive` flag), and in `metrics/entropy.py` the two refining measures (`experienced_segregation`, `intergroup_contact`) + the `gini_simpson` helper.
- **Network indicator:** `Metric.summarize()` reduces each per-segment map to one indicator per `(mode, category, binding, attribute)`; the per-group metrics (exposure, exclusivity) override it.
- **Network/aggregation:** `network/builder.to_undirected_graph` (undirected merge at build time, via NetworkX); `aggregation/edge_accumulator.py` gained `canonical_edge`, the `n_paths` counter and `path_counts()`.
- **Configuration:** new `metrics` block (`enabled`, `exclusive_lq_threshold`) and `network.undirected` — all runtime-configurable, nothing hard-coded.
- **Orchestration:** `orchestrator.aggregate` runs **every** registered metric and returns both the per-segment maps and the network `summaries`.
- **Tests:** `test_dominant.py`, `test_exposure.py`, `test_segregated.py`, `test_refining.py`, plus new cases in `test_entropy.py`, `test_aggregation.py` (merge + `n_paths`) and `test_network.py` (undirected graph).
- **Environment note:** `pyarrow` is capped at `<18` in `pyproject.toml` — the 18+ wheels are built against the numpy 2 ABI and crash under the project's `numpy<2` pin (same story as Pandana).

**P4.5 — route set:**
- **Core:** `routing/route_set.py` — `link_penalty_routes` (Dijkstra iterated over a mutable copy of the weights), `yen_routes` (NetworkX `shortest_simple_paths`), `route_weights` (the three schemes, incl. the *path-size* factor that discounts overlap) and the single entry point `build_route_set`. Both methods run on a **min-weight simple `DiGraph` view** (`simple_digraph_view`) and decode the node path back to `(u, v, key)` on the original `MultiDiGraph` via `_paths.node_path_to_edges` — the **same edge identity** the aggregator uses, so both methods and the single-route path are comparable edge for edge.
- **Trajectory:** `Trajectory` gained `route_rank` and `route_weight` (defaults `0` / `1.0`, so nothing changes on the single-route path).
- **OD routing:** `routing/od_router.py` emits `k` trajectories per pair when `k > 1` and the mode's graph is passed (`graphs=`); with `k = 1` it keeps using the fast backend (Pandana). An unreachable pair still becomes a trajectory with `reason='unreachable'`, never dropped.
- **Aggregation:** `orchestrator._scale_vector` multiplies the demographic vector by `wᵢ` before depositing. `NaN × w = NaN`, so an origin without demographics stays flagged and is **never zeroed**.
- **Configuration:** new `routing.route_set` block — `k` (default `1` = off), `method`, `penalty` (validated `> 1`), `weight_scheme`, `logit_beta`.
- **Demo:** `--compare-routesets` (runs both methods on a random sample of pairs and writes a table + CSVs + figure), `--sample`, `--k`, `--penalty`, `--seed`, and `--compare-only` (just the comparison, skipping full routing). Sampling happens **before** snapping, to avoid paying the cost of snapping the whole OD matrix to the network.
- **Tests:** `tests/unit/test_route_set.py` — covers route distinctness, Yen's cost order, `Σ wᵢ = 1` for the three schemes, the degenerate pair (origin = destination) and mass conservation with split flow for both methods.
