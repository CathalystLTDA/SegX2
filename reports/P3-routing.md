# SegX — P3: routing core

**Phase:** P3 — trajectories over the street network.

---

## 1. One-line summary

The engine now **moves people through the streets**. Taking P2's located, labelled origins, it walks the real street network and produces, for every resident/flow, the **trajectory** they would actually take — to the nearest facility, to everything reachable within 30 minutes, and along every home→work commute. Those trajectories are exactly what P4/P5 stamp onto the street segments to measure *experienced segregation*.

P3 is the computational core of the engine: where "loaded data" becomes "paths on the map".

---

## 2. What P3 implements

### 2.1 Three ways a person "uses the city"

The engine produces three kinds of trajectory, each answering a different exposure question — all configurable, no category name hard-coded:

1. **Nearest destination (binding A).** For every origin, the nearest facility of *each* category (nearest park, nearest bus stop…). Measured in **metres** on foot, subject to a configurable maximum distance.
2. **Catchment (binding B).** For every origin, **all** facilities of each category reachable within a budget (default **30 minutes walking at 5 km/h**; distance budgets and per-mode budgets are also accepted). This is the "opportunity" reading: how much of the city is within reach of the people living there.
3. **Home→work commutes (binding C).** Routes each origin→destination pair of the flow file (e.g. LODES) directly, in the pair's mode (car→`drive`, walk→`walk`), measured in **travel time**.

All of this comes out as **one unified trajectory stream**, each trajectory tagged `(mode, category, binding)` — exactly the partition key P4's aggregation consumes. Catchment and nearest stay separable.

### 2.2 The most important invariant: "unreachable" is never zero

This was handled strictly because it is where segregation metrics silently go wrong:

- No facility of the category inside the boundary → trajectory flagged `none_in_boundary`.
- Facility exists but there is no path → `unreachable`.
- Exists but beyond the distance limit → `over_max_distance` (and the engine still **says which** facility was too far).
- Empty catchment (the person reaches nothing of that category within the budget) → a **real zero**, counted and reported, never an error and never an invented number.

Every run returns a reachability summary per binding (how many trajectories were reachable vs. unreachable, and why), ready to display without reprocessing.

> **Correction (found after P3).** The Pandana backend did not fully honour this invariant.
> Pandana reports a disconnected pair with a cost of `4294967.295` (the uint32 maximum in its
> internal units) and an empty path, but the engine's "unreachable" sentinel was set to `1e30`.
> Disconnected pairs therefore came back as *reachable* routes with zero edges — 162 of them on
> a 31,614-flow Philadelphia clip. An empty path touches no segment, so no per-segment value
> changed, but those flows were booked as reachable (and their mass as deposited in the
> conservation ledger) instead of `unreachable`. The NetworkX oracle was not affected, and the
> cross-check of §3 did not catch it. The sentinel now matches Pandana's value
> ([`segx/routing/pandana_engine.py`](../segx/routing/pandana_engine.py)), and a regression
> test routes a pair to a disconnected node through both the batched and the per-pair calls
> (`test_pandana_batched_shortest_paths_match_the_per_pair_call` in
> [`tests/unit/test_routing.py`](../tests/unit/test_routing.py)).

---

## 3. How we know the routing is correct

The engine has **two routing backends**, and one is used to prove the other:

- **Pandana** (default) — based on *contraction hierarchies*, the only one fast enough for city scale (10⁵–10⁶ origins). It became a core dependency in P3; it installs from a prebuilt wheel on Python 3.11 (incl. Windows), which is why the project is pinned to 3.11.
- **NetworkX** (validation oracle) — plain Dijkstra, slow but unquestionably correct.

Both decode paths with **the same code**, so they produce *identical* trajectories — and an automated test confirms Pandana and NetworkX agree (cost and catchment) on the synthetic city. Results are also checked against **hand-computed values** on the synthetic city: distance to the nearest park, the exact set of facilities within the budget, the commute cost.

---

## 4. Defaults

All configurable:

1. **Access mode for bindings A/B = walk.** "Nearest" and "catchment" assume walking by default.
2. **Catchment budget = 30 min walking**, the usual urban-planning standard.
3. **Catchment weighting.** When a person reaches *k* facilities, the engine can give each one unit weight (`unit` — every facility is a real opportunity) or divide the origin's weight by *k* (`inverse_count`).
4. **Cost per binding.** Nearest in metres, catchment in minutes (optionally distance), commute in minutes.

---

### Technical appendix
- **Engine interface** (`routing/engine.py`): `RoutingEngine` protocol with three primitives (`nearest`, `reachable_within`, `shortest_path`), all returning a `RouteResult` whose path is decoded into `(u, v, key)` edges. Strategies are written against the interface, never against a concrete backend.
- **Backends**: `routing/pandana_engine.py` (default) and `routing/networkx_engine.py` (oracle); shared path decoding in `routing/_paths.py` (single `MultiDiGraph` view → single edge identity).
- **Strategies**: `routing/nearest_poi.py` (A), `routing/poi_catchment.py` (B), `routing/od_router.py` (C); unified record in `routing/trajectories.py` (`Trajectory` with `(mode, category, binding)`, `weight`, `path`, `cost`, `reason`).
- **Orchestration**: `orchestrator.route()` snaps origins/POIs to the access-mode graph and each OD endpoint to its own mode's graph; over-tolerance snaps are *flagged* in `stats`, not dropped.
- **New config**: `RoutingConfig` (`engine`, `access_mode`).
- **Tests**: `tests/unit/test_routing.py` (ground truth + Pandana↔NetworkX cross-check), updated orchestrator tests, and the `tests/integration/test_pipeline_synthetic.py` integration test extended through P3.
- **Invariants kept**: single projection; "unreachable" is a typed result, never zero; everything category/budget-related comes from configuration; aggregation ready to be *streaming*, partitioned by `(mode, category, binding)`.
