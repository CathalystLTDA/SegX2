# SegX — How each metric is computed

**Scope of this document:** for every delivered measure (P4 + P5), the **exact formula** used,
the **edge-case convention**, and the **code that implements it**, with links to the source.

Every metric is a *plugin* (`segx/metrics/`): it reads **the same
per-segment table** produced by aggregation (P4) and returns (a) a per-segment value and
(b) a single network-wide indicator. None of them touch routing or aggregation.

---

## 0. The input to every metric — the per-segment table

Before any formula, we need to know **what a segment's "mass" (`mass`) is**.

Each routed trajectory (P3) carries the demographic vector of its origin
(`demo_vector` = agent counts per class, e.g. `{income: {low: 12, mid: 30, high: 5}}`).
The accumulator deposits that vector **on every edge of the traversed path**, summing per
partition `(mode, category, binding)`:

[`segx/aggregation/edge_accumulator.py:134-147`](segx/aggregation/edge_accumulator.py#L134-L147)

```python
partition = trajectory.partition
edges = self._acc.setdefault(partition, {})
for edge in trajectory.path:
    ce = canonical_edge(edge)
    bucket = edges.get(ce)
    if bucket is None:
        bucket = {name: [0.0] * len(labels) for name, labels in self._attrs}
        edges[ce] = bucket
    for name, labels in self._attrs:
        vec = demo_vector[name]
        arr = bucket[name]
        for i, label in enumerate(labels):
            arr[i] += vec[label]
    self._edge_mass_total += mass
```

Formally, for segment *s*, class *i* of a demographic attribute, and the set *T(s)* of
trajectories passing through *s* within a partition:

$$P_i(s) \;=\; \sum_{t \in T(s)} w_t \cdot v_i(t)$$

where `v_i(t)` is the agent count of class *i* at the trajectory's origin and `w_t` is the
**route-choice weight** (`route_weight`, P4.5 — equal to `1` when there is a single route;
with *k* alternative routes, `Σ wᵢ = 1`, so the flow is **split**, not duplicated):

[`segx/orchestrator.py:426-442`](segx/orchestrator.py#L426-L442)

```python
if trajectory.route_weight != 1.0:
    return _scale_vector(vector, trajectory.route_weight)
return vector
```

The result is a long table (`to_frame()`), one row per
`(mode, category, binding, u, v, key, attribute, class)` with columns `mass` and `n_paths`:
[`segx/aggregation/edge_accumulator.py:161-197`](segx/aggregation/edge_accumulator.py#L161-L197).

**Notation used from here on:**

| Symbol | Meaning |
|---|---|
| `Pᵢ` | mass (agent count) of class *i* on the segment |
| `Pₜ` | total segment mass, `Pₜ = Σᵢ Pᵢ` |
| `pᵢ` | proportion of class *i* on the segment, `pᵢ = Pᵢ / Pₜ` |
| `n` | number of **possible** classes for the attribute (from the demographic schema, not the observed set) |
| partition | `(mode, category, binding)` — everything is computed **within** a partition |

**Edge-case convention ("no data never becomes zero"):**
a segment with `Pₜ = 0` → **`NaN`** (the measure is undefined, it is not zero).
`0.0` is reserved for the real case "all mass is in a single class".

---

## 1. Base functions (shared)

Three pure mathematical functions, tested against hand-computed values on the synthetic city
(`tests/unit/test_entropy.py`, `segx/validation/synthetic_city.py`).

### 1.1 Shannon entropy `H`

$$H \;=\; -\sum_i p_i \log_2 p_i \qquad p_i = \frac{P_i}{P_t}$$

Measures the **mixing** of the segment, in *bits*, independent of how many people pass.
`H = 0` → a single class. `H = log₂ n` → all classes equally present.

[`segx/metrics/entropy.py:53-73`](segx/metrics/entropy.py#L53-L73)

```python
def shannon_entropy(counts: Iterable[float], base: float = 2.0) -> float:
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    log_b = math.log(base)
    h = 0.0
    for c in counts:
        if c > 0.0:
            p = c / total
            h -= p * math.log(p) / log_b
    # clamp tiny negative dust from floating-point
    return 0.0 if h < 0.0 else h
```

### 1.2 Normalized entropy `h` (Pielou evenness)

$$h \;=\; \frac{H}{\log_2 n} \;\in\; [0, 1]$$

The same entropy rescaled to 0–1 by dividing by the maximum possible — which makes segments
and cities **comparable** to one another.

[`segx/metrics/entropy.py:76-88`](segx/metrics/entropy.py#L76-L88)

```python
def normalized_entropy(counts: Iterable[float], n_classes: int, base: float = 2.0) -> float:
    if n_classes < 2:
        return 0.0
    h = shannon_entropy(counts, base=base)
    if math.isnan(h):
        return math.nan
    return h / (math.log(n_classes) / math.log(base))
```

### 1.3 Gini–Simpson index

$$G \;=\; 1 - \sum_i p_i^{2}$$

[`segx/metrics/entropy.py:114-134`](segx/metrics/entropy.py#L114-L134)

```python
def gini_simpson(counts: Iterable[float]) -> float:
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    sum_sq = math.fsum((c / total) ** 2 for c in counts)
    g = 1.0 - sum_sq
    # clamp tiny negative dust from floating-point
    return 0.0 if g < 0.0 else g
```

---

## 2. Per-segment diversity — `diversity` (P4)

**What it is.** The formula **exactly as written in Netto et al. (2018),
*A temporal geography of encounters*, fig. 5** — the social diversity of each street segment.

$$D \;=\; -\sum_i P_i \log_2\!\left(\frac{P_i}{P_t}\right)$$

Note that the `Pᵢ` stay as **absolute counts** outside the log (not proportions).
Algebraically:

$$D \;=\; P_t \cdot H$$

i.e. **entropy weighted by the segment's agent volume** — busy *and* mixed streets score
higher; an alley with 3 people from 3 different groups does not tie with an avenue carrying
3,000.

[`segx/metrics/entropy.py:91-111`](segx/metrics/entropy.py#L91-L111)

```python
def segment_diversity(counts: Iterable[float], base: float = 2.0) -> float:
    counts = [float(c) for c in counts]
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")
    total = math.fsum(counts)
    if total <= 0.0:
        return math.nan
    log_b = math.log(base)
    d = 0.0
    for c in counts:
        if c > 0.0:
            d -= c * math.log(c / total) / log_b  # log(c/total) ≤ 0 → contribution ≥ 0
    return 0.0 if d < 0.0 else d
```

Applied to the table (one value per segment, per attribute, per partition):

[`segx/metrics/entropy.py:155-165`](segx/metrics/entropy.py#L155-L165)

```python
rows = (
    edge_table.groupby(list(self.GROUP_KEYS), sort=False)["mass"]
    .apply(lambda s: segment_diversity(s.to_numpy()))
    .reset_index(name=self.name)
)
```

> **Alternative:** the normalized `h` (comparable across segments) instead of `D` (the paper's
> formula, sensitive to volume). Swapping is a one-line change; nothing in routing/aggregation
> depends on the choice.

---

## 3. Experienced segregation — `experienced_segregation` (P5)

**What it is.** The degree of **social homogeneity** experienced along trajectories: the
complement of the normalized entropy.

$$S(s) \;=\; 1 - h(s)$$

- `0` → perfectly mixed segment (`h = 1`);
- `1` → fully homogeneous segment (a single class);
- `NaN` → segment with no mass.

**Network indicator:** mean over segments (skipping `NaN`).

[`segx/metrics/entropy.py:191-208`](segx/metrics/entropy.py#L191-L208)

```python
class ExperiencedSegregationMetric(Metric):
    name = "experienced_segregation"

    def compute(self, edge_table, settings: Settings):
        def _seg(masses):
            h = normalized_entropy(masses, n_classes=len(masses))
            return math.nan if math.isnan(h) else 1.0 - h

        return _per_segment(edge_table, _seg, self.name)
```

The per-segment grouping is the shared helper — since `to_frame()` emits **one row per
possible class** (zeros included), `len(group)` already is the number of classes `n`:

[`segx/metrics/entropy.py:174-188`](segx/metrics/entropy.py#L174-L188)

```python
return (
    edge_table.groupby(list(_SEGMENT_KEYS), sort=False)["mass"]
    .apply(lambda s: fn(s.to_numpy()))
    .reset_index(name=value_name)
)
```

---

## 4. Intergroup contact probability — `intergroup_contact` (P5)

**What it is.** The probability that **two people drawn at random from the segment belong to
different groups** — the *exposure/probability* measure.

$$C(s) \;=\; 1 - \sum_i p_i^{2}$$

Distinct from entropy: entropy measures *diversity*; this measures *contact likelihood*.
`0.0` → everyone in the same group (intergroup contact impossible). `NaN` → segment with no mass.

**Network indicator:** mean over segments.

[`segx/metrics/entropy.py:211-223`](segx/metrics/entropy.py#L211-L223)

```python
class IntergroupContactMetric(Metric):
    name = "intergroup_contact"

    def compute(self, edge_table, settings: Settings):
        return _per_segment(edge_table, gini_simpson, self.name)
```

---

## 5. Social exposure — `exposure` (P5)

**What it is.** Per **social group** (not per segment): how much diversity that group
potentially encounters along its daily trajectories.

Step 1 — the composition entropy of each segment, `H(s)` (§1.1).
Step 2 — for group *g*, the average of `H(s)` over the segments *g* actually traverses
(`P_g(s) > 0`), **weighted by g's presence** on each:

$$E_g \;=\; \frac{\displaystyle\sum_{s \,:\, P_g(s) > 0} P_g(s)\, H(s)}{\displaystyle\sum_{s \,:\, P_g(s) > 0} P_g(s)}$$

The weighting is what makes the measure *experienced*: the segments where the group spends more
time / has more people weigh more in what that group "sees".

Result: **one value per group**, per partition `(mode, category, binding, attribute)`.
A group that traverses no segment (or only segments with undefined `H`) → `NaN`.

**Network indicator:** mean exposure across groups.

[`segx/metrics/exposure.py:47-66`](segx/metrics/exposure.py#L47-L66)

```python
# per-segment composition entropy H (bits), one value per segment
seg_h = (
    edge_table.groupby(list(_SEGMENT_KEYS), sort=False)["mass"]
    .apply(lambda s: shannon_entropy(s.to_numpy()))
    .reset_index(name="_H")
)
merged = edge_table.merge(seg_h, on=list(_SEGMENT_KEYS), how="left")

records: list[dict] = []
for keys, sub in merged.groupby([*_PARTITION_KEYS, "class"], sort=False):
    mass = sub["mass"].to_numpy(dtype=float)
    h = sub["_H"].to_numpy(dtype=float)
    mask = (mass > 0.0) & ~np.isnan(h)
    if not mask.any():
        value = float("nan")
    else:
        w = mass[mask]
        value = float((w * h[mask]).sum() / w.sum())
    records.append({**dict(zip([*_PARTITION_KEYS, "class"], keys, strict=True)), self.name: value})
```

---

## 5b. Cross-mode exposure — `cross_mode_exposure` (P7.1)

**What it is.** The same measure as §5, re-read after counting the people on the segment who
are travelling by a **different mode**. `mode` is a partition key in aggregation, so in
`exposure` a pedestrian and a bus passenger on the same avenue are in disjoint buckets and
never meet. They *are* co-present — less than two pedestrians are, and `factor` is that
discount.

For a segment *s* and an **observer** mode *m*, with `f(m, m) = 1`:

$$P^{\text{eff}}_i(s, m) \;=\; \sum_{m'} f(m, m')\, P_i(s, m')
\qquad H_{\text{cross}}(s, m) \;=\; H\big(P^{\text{eff}}(s, m)\big)$$

Then, exactly as in §5, per group *g*: the mean of `H_cross` over the segments *g* traverses,
weighted by *g*'s **own** presence `P_g(s, m)`. Output shape and network indicator are
identical to `exposure`, so the two are read side by side.

**Parameters** (`metrics.cross_mode`): `factor` (default **0.5**) and per-pair `overrides`
keyed `"mode_a|mode_b"`, stored with the two modes sorted — the factor is **symmetric**.

**Two rules that make it well-defined:**

- Pooling runs across `mode` **only**, within `(category, binding, attribute)`. The reason is
  dimensional: a catchment trajectory under `weight_mode="unit"` deposits its origin's whole
  vector once *per reachable POI*, so its mass is inflated by a POI count; adding it to a
  commute flow's workers would be adding unlike units.
- Only the **cross** term is filtered to `layer == "street"`. A `rail_line` segment keeps
  reporting its own composition — a passenger in a tunnel is co-present with nobody on the
  avenue above. Consequently `factor = 0` reproduces `exposure` **row for row**, and
  `factor = 1` is total pooling (`H_cross` identical across observer modes).

[`segx/metrics/cross_mode_exposure.py`](segx/metrics/cross_mode_exposure.py)

```python
for mode in pd.unique(df["mode"]):                       # at most 3 modes
    weights = df["mode"].map(lambda other, m=mode: cfg.factor_for(m, other))
    pooled = (mass * weights).where(street, 0.0).groupby(pool_by, sort=False).transform("sum")
    effective = pooled.where(street, mass)               # non-street rows keep their own mass
```

**Worked example** — one street segment, attribute `income ∈ {low, high}`, `f = 0.5`:

| | low | high | | effective vector | `H_cross` |
|---|---|---|---|---|---|
| walk | 3 | 1 | observer **walk** | `(3 + 0, 1 + 2) = (3, 3)` | **1.0000** |
| transit | 0 | 4 | observer **transit** | `(0 + 1.5, 4 + 0.5) = (1.5, 4.5)` | **0.8113** |

Plain `exposure` would give walk `H(3, 1) = 0.8113`.

**Three things a reader must be told, or they will read a bug:**

1. **`cross_mode_exposure` can be LOWER than `exposure`.** Adding a mode whose composition is
   *more* concentrated than the observer's reduces the entropy. Correct, and counterintuitive.
2. **The measure is strongly asymmetric, and the signal is on the transit rows.** A segment
   with 1000 pedestrians and 10 passengers gives `H_cross(walk) ≈ H(walk)` — the passengers are
   diluted — while `H_cross(transit)` (10 of their own against `0.5 × 1000 = 500`) is dominated
   by the pedestrians. **The `walk` rows will look almost identical to `exposure`.**
3. **Real scope: the commute.** The access bindings use a single `routing.access_mode`, so
   `nearest`/`catchment` contribute one mode and the metric reduces identically to `exposure`
   there. It is non-trivial **only for `binding="od"`**, and only when the OD file mixes modes.

> **Coverage caveat.** Two modes only mix where they deposit on the same `(u, v, key)`.
> `network.simplify` runs independently per extract, so the `walk` and `drive` id spaces can
> diverge in a real city and the join can find nothing — which shows up not as a zero but as
> `cross_mode_exposure == exposure`. `AggregationResult.stats["cross_mode"]` reports
> `n_modes_present`, `n_segments_total`, `n_segments_multimode`, `cross_mode_join_rate` and
> `n_surface_rides_unmatched`; the metric emits a `UserWarning` when several modes are present
> and the join rate is 0. **Read that number before interpreting this metric.**

---

## 6. Dynamic segregation / dominant group — `dominant_trajectories` (P5)

**What it is.** Per segment, **which group has the highest presence** (the segment's colour on
the map) and **how sharply** it dominates (the intensity).

$$\text{dominant\_class}(s) \;=\; \arg\max_i P_i(s)$$

$$\text{dominance}(s) \;=\; 1 - h(s)$$

The weight uses the **whole distribution** (normalized
entropy), not just the winner's share:

- `33 / 33 / 34` → dominance ≈ `0` (near-perfect tie, even though there is a winner);
- `0 / 0 / 100` → dominance `= 1`.

A segment with no mass → `dominant_class = None` and `dominance = NaN` (never a spurious winner).

**Network indicator:** mean dominance across segments (`value_column = "dominance"`).

[`segx/metrics/dominant_trajectories.py:47-64`](segx/metrics/dominant_trajectories.py#L47-L64)

```python
records: list[dict] = []
for keys, sub in edge_table.groupby(list(_SEGMENT_KEYS), sort=False):
    masses = sub["mass"].to_numpy(dtype=float)
    classes = sub["class"].to_numpy()
    total = math.fsum(masses)
    if total <= 0.0:
        dominant, dominance = None, math.nan
    else:
        winner = int(masses.argmax())
        dominant = str(classes[winner])
        dominance = 1.0 - normalized_entropy(masses, n_classes=len(masses))
    records.append(
        {
            **dict(zip(_SEGMENT_KEYS, keys, strict=True)),
            "dominant_class": dominant,
            self.value_column: dominance,
        }
    )
```

> Note that `dominance` and `experienced_segregation` (§3) use **the same expression `1 − h`**:
> the former tracks *who* dominates the segment, the latter is the aggregate homogeneity
> indicator. They are two readings of the same quantity, and that is intentional.

---

## 7. Most exclusive segments — `segregated_trajectories` (P5)

**What it is.** The segments used **disproportionately by a single group**. The
unit of analysis is the **street segment** (not the whole trajectory — which is why nothing
needs to be retained per trajectory).

For each segment *s* and class *g*, a **location quotient**:

$$LQ_g(s) \;=\; \frac{p_g(s)}{p_g(\text{partition})} \;=\; \frac{P_g(s) / P_t(s)}{\sum_{s'} P_g(s') \big/ \sum_{s'} P_t(s')}$$

— group *g*'s share **on that segment** divided by *g*'s share **across the whole partition**.
`LQ = 1` → the group is as present there as in the city average; `LQ > 1` →
**over-represented**.

A segment is flagged **`exclusive`** when its **dominant group**'s `LQ` reaches the threshold:

$$\text{exclusive}(s) \;=\; \big[\, LQ_{\text{dominant}}(s) \;\ge\; \tau \,\big], \qquad \tau = \texttt{metrics.exclusive\_lq\_threshold} \;(\text{default } 2.0)$$

The threshold is configurable ([`segx/config.py:382-383`](segx/config.py#L382-L383)) and is
a parameter that directly affects the results. A segment with no mass, or a group
absent from the whole partition (`0/0`), → `LQ = NaN` and `exclusive = False`.

[`segx/metrics/segregated_trajectories.py:53-75`](segx/metrics/segregated_trajectories.py#L53-L75)

```python
threshold = settings.metrics.exclusive_lq_threshold
df = edge_table
part = list(_PARTITION_KEYS)
# global group share within the partition: total_g / total_all
total_g = df.groupby([*part, "class"], sort=False)["mass"].transform("sum")
total_all = df.groupby(part, sort=False)["mass"].transform("sum")
global_p = total_g / total_all  # NaN/inf where the partition has no mass
# segment group share: mass_g / segment_total
seg_total = df.groupby(list(_SEGMENT_KEYS), sort=False)["mass"].transform("sum")
seg_p = df["mass"] / seg_total
df = df.assign(_lq=(seg_p / global_p))

records: list[dict] = []
for keys, sub in df.groupby(list(_SEGMENT_KEYS), sort=False):
    masses = sub["mass"].to_numpy(dtype=float)
    total = math.fsum(masses)
    if total <= 0.0:
        dominant, lq, exclusive = None, math.nan, False
    else:
        winner = int(masses.argmax())
        dominant = str(sub["class"].to_numpy()[winner])
        lq = float(sub["_lq"].to_numpy()[winner])
        exclusive = bool(lq >= threshold) if not math.isnan(lq) else False
```

**Network indicator** (this metric overrides the default summary): count and **share of
exclusive segments per dominant group** —

$$\text{share\_exclusive}(g) \;=\; \frac{\#\{s : \text{dominant}(s) = g \ \wedge\ \text{exclusive}(s)\}}{\#\{s : \text{dominant}(s) = g\}}$$

[`segx/metrics/segregated_trajectories.py:86-97`](segx/metrics/segregated_trajectories.py#L86-L97)

```python
keys = ["mode", "category", "binding", "attribute", "dominant_class"]
out = grouped.agg(n_segments=("exclusive", "size"), n_exclusive=("exclusive", "sum")).reset_index()
out["share_exclusive"] = out["n_exclusive"] / out["n_segments"]
```

---

## 8. From per-segment value to network indicator

Except for exclusivity (§7), every metric reduces the map to **one number per partition** by the
same rule: the **mean of the value column, skipping `NaN`** — segments with no demographic mass
simply drop out of the count, instead of dragging the mean down as if they were zeros.

$$\overline{M} \;=\; \frac{1}{|S^{*}|}\sum_{s \in S^{*}} M(s), \qquad S^{*} = \{s : M(s) \neq \text{NaN}\}$$

[`segx/metrics/base.py:42-57`](segx/metrics/base.py#L42-L57)

```python
def summarize(self, per_segment, settings: Settings):
    col = self.value_column or self.name
    keys = ["mode", "category", "binding", "attribute"]
    if per_segment.empty:
        return pd.DataFrame(columns=[*keys, col])
    return per_segment.groupby(keys, sort=False)[col].mean().reset_index()
```

And the orchestrator runs **every registered metric** (or the subset in
`settings.metrics.enabled`) over the same table, storing each one's map + summary:

[`segx/orchestrator.py:471-479`](segx/orchestrator.py#L471-L479)

```python
metric_names = settings.metrics.enabled if settings.metrics.enabled is not None else available_metrics()
_stage("compute_metrics", ", ".join(metric_names))
metrics: dict[str, pd.DataFrame] = {}
summaries: dict[str, pd.DataFrame] = {}
for name in metric_names:
    metric = get_metric(name)()
    per_segment = metric.compute(edge_table, settings)
    metrics[name] = per_segment
    summaries[name] = metric.summarize(per_segment, settings)
```

---

## 9. Full numerical example (one segment, three groups)

A segment with per-class income masses `P = (low 10, mid 30, high 60)`, so `Pₜ = 100`,
`p = (0.1 · 0.3 · 0.6)` and `n = 3` possible classes.

| Step | Computation | Result |
|---|---|---|
| Entropy `H` | `−(0.1·log₂0.1 + 0.3·log₂0.3 + 0.6·log₂0.6)` = `0.33219 + 0.52109 + 0.44218` | **1.29546 bits** |
| Normalized `h` | `1.29546 / log₂3` = `1.29546 / 1.58496` | **0.81735** |
| Diversity `D` (§2) | `Pₜ · H` = `100 × 1.29546` | **129.546** |
| Experienced segregation (§3) | `1 − h` | **0.18265** |
| Intergroup contact (§4) | `1 − (0.1² + 0.3² + 0.6²)` = `1 − 0.46` | **0.54** |
| Dominant group (§6) | `argmax P` | **high** |
| Dominance (§6) | `1 − h` | **0.18265** |
| `LQ` of the dominant (§7) | assuming `high` = 30% of the partition mass: `0.6 / 0.3` | **2.0 → `exclusive` ✅** (τ = 2.0) |

And the **exposure** (§5) of group `low`, assuming it traverses two segments — the one above
(`P_low = 10`, `H = 1.29546`) and a second with `P_low = 40`, `H = 0.5`:

```text
E_low = (10 × 1.29546 + 40 × 0.5) / (10 + 40)
      = (12.9546 + 20) / 50
      = 0.65909 bits
```

That is: although the first segment is well mixed, group `low` spends most of its path on the
second — and the weighted exposure reflects that.

---

## 10. Where each formula is verified

Every measure has a test with a hand-computed value (the synthetic city in
`segx/validation/synthetic_city.py` provides the ground truth):

| Metric | Implementation | Test |
|---|---|---|
| `shannon_entropy`, `normalized_entropy`, `segment_diversity`, `gini_simpson` | [segx/metrics/entropy.py](segx/metrics/entropy.py) | [tests/unit/test_entropy.py](tests/unit/test_entropy.py) |
| `diversity` | [segx/metrics/entropy.py](segx/metrics/entropy.py) | [tests/unit/test_entropy.py](tests/unit/test_entropy.py), [tests/unit/test_aggregation.py](tests/unit/test_aggregation.py) |
| `experienced_segregation`, `intergroup_contact` | [segx/metrics/entropy.py](segx/metrics/entropy.py) | [tests/unit/test_refining.py](tests/unit/test_refining.py) |
| `exposure` | [segx/metrics/exposure.py](segx/metrics/exposure.py) | [tests/unit/test_exposure.py](tests/unit/test_exposure.py) |
| `dominant_trajectories` | [segx/metrics/dominant_trajectories.py](segx/metrics/dominant_trajectories.py) | [tests/unit/test_dominant.py](tests/unit/test_dominant.py) |
| `segregated_trajectories` | [segx/metrics/segregated_trajectories.py](segx/metrics/segregated_trajectories.py) | [tests/unit/test_segregated.py](tests/unit/test_segregated.py) |
| aggregated-mass conservation | [segx/aggregation/edge_accumulator.py](segx/aggregation/edge_accumulator.py) | [tests/integration/test_pipeline_synthetic.py](tests/integration/test_pipeline_synthetic.py) |

Reproduce: `uv run pytest -q`.
