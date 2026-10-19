# SegX — P4: per-segment aggregation and diversity

**Phase:** P4 — demographic mass per street segment + entropy.

---

## 1. One-line summary

The engine now **produces its first real measurement**: on every street segment it sums how many people of each income class pass through, and computes that segment's **social diversity** (Shannon entropy). This is the core operation of *A temporal geography of encounters* (Netto et al., 2018): *"we count the income-classified trajectories for each street segment… accumulated per income group."*

P4 is the bridge between "we have trajectories" and "we have numbers measured on the streets".

---

## 2. What P4 implements

1. **Per-segment accumulation (streaming).** As each trajectory is walked, every segment on its path receives that trajectory's class vector, summed per partition `(mode, category, binding)` — so "walk/park/nearest", "walk/park/catchment" and "drive/commute/OD" stay **separable**. State is bounded by segments × partitions, **never** by the number of trajectories (no per-trajectory storage in the hot path).

2. **Per-segment diversity (Shannon entropy).** For each segment, the social diversity — how mixed, in income terms, the population passing through it is. This produces the "entropy map" (Figure 5 of the paper) and supports the reading of where "the other" is most likely to be met.

3. **Dominant group per segment.** Derived directly from the aggregated table: the class with the most agents colours the segment — the "dominant class network" (Figure 3 of the paper).

4. **"No data" never becomes zero.** A trajectory whose origin has no resolved demographics is **flagged and counted separately**, never summed as zero — otherwise it would corrupt every diversity value downstream.

---

## 3. The entropy formula: weighted D vs. normalized H

Figure 5 of the paper writes the diversity formula as:

$$D = -\sum_i P_i \, \log_2\!\left(\frac{P_i}{P_t}\right)$$

where $P_i$ is the **absolute count** of people of class $i$ on the segment and $P_t$ the total. Note that $P_i$ appears **as a count, outside the log** — which algebraically gives:

$$D = P_t \cdot H \qquad \text{with} \qquad H = -\sum_i p_i \log_2 p_i$$

i.e. entropy **weighted by the volume of people** on the segment. This differs from "classic" Shannon entropy $H$ (with proportions $p_i = P_i/P_t$), which measures *mixing* only, regardless of how many people pass.

- **With the formula as written (D):** busy AND mixed streets score higher (it measures "amount of diversity").
- **With normalized $H$:** it measures only the degree of mixing, indifferent to volume.

**What is implemented:** the `diversity` metric uses **the formula exactly as in the paper** ($D = P_t \cdot H$). The normalized $H$ is also in the code. **Switching is a one-line change** — nothing in routing or aggregation depends on which one is used.

*Deposit model:* each trajectory deposits its class counts directly — for OD the **per-flow** vector is used [that flow's SE01/02/03], for POI access the **origin's** vector. A block's workers are therefore split across their destinations rather than multiplied by them.

---

## 4. How we know it is correct

- **Weight conservation (invariant).** Every unit of mass that enters is accounted for: `emitted = deposited + unreachable`. Checked on every run; a violation fails the run instead of emitting a plausible-but-wrong number.
- **Hand-computed ground truth.** Unit tests pin values computed on paper — e.g. a segment with `{6 low, 2 high}` must give `D = 6.4902`; a homogeneous segment `{8, 0}` → `D = 0`; a segment with no mass → `NaN` (never 0).
- **Integration gate.** The "per-segment sums" and "weight conservation" integration tests run for real on the synthetic city.
- **Real-data proof (Philadelphia, Center City clip).** The pipeline ran end to end through P4: **15,131 segments** carrying mass, **9 partitions**, **exact conservation** (emitted 328,535 = deposited, 0 lost), diversity computed for every segment.

---

## 5. What P4 does **not** do

The paper's headline part — **potential encounters in time** (co-presence within a 5-minute window, space–time prism, Ripley's K function, proximity network) — is **not** P4. It requires **time-stamped** trajectories and pairwise co-presence detection between agents; the engine's trajectories are timeless (commute/access, no clock).

---

### Technical appendix
- **Accumulator:** `aggregation/edge_accumulator.py` (`EdgeAccumulator` — per-segment/class sums partitioned by `(mode, category, binding)`; conservation ledger + unresolved counter; `to_frame()` produces a long table).
- **Metric:** `metrics/entropy.py` — `segment_diversity` implements the paper's formula `D = -Σ Pᵢ log₂(Pᵢ/Pₜ)`; the normalized `shannon_entropy` lives in the same module; the registered metric is named `diversity`.
- **Orchestration:** `orchestrator.aggregate` runs P4's three stages (accumulate → metrics → `check_weight_conservation`).
- **Tests:** `tests/unit/test_aggregation.py` (hand-computed ground truth) + the integration tests in `tests/integration/test_pipeline_synthetic.py`.
- **Demo:** `demo/run_demo.py` runs P4, organises outputs into folders (`inputs/ maps/ segments/ charts/ tables/`) and draws the **dominant group** (Fig. 3) and **diversity** (Fig. 5) maps, plus distribution charts.
