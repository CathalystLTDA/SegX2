"""Cross-mode exposure — the diversity a mode's travellers experience once the
*other* modes sharing the same street are counted too. P7.1.

``mode`` is a partition key in the :class:`~segx.aggregation.edge_accumulator.
EdgeAccumulator` (invariant #6), so today a pedestrian and a bus passenger on the
same avenue fall into disjoint buckets and meet in no metric at all. They *are*
co-present — less than two pedestrians are, and
``settings.metrics.cross_mode.factor`` (default ``0.5``) is that discount.

For a segment *s* and an **observer** mode *m*:

.. code-block:: text

    effective_c(s, m) = mass_c(s, m)  +  Σ_{m' ≠ m} factor(m, m') · mass_c(s, m')
    H_cross(s, m)     = shannon_entropy(effective vector over classes c)

then, per group *g*, the mean of ``H_cross`` over the segments *g* traverses,
weighted by *g*'s own presence — **exactly the idiom of**
:class:`~segx.metrics.exposure.ExposureMetric`, so the two are read side by side
and the inherited :meth:`~segx.metrics.base.Metric.summarize` needs no change.

Three properties this is built around, each pinned by a test:

* ``factor = 0`` reproduces ``exposure`` **row for row** — the cross term
  vanishes and only a mode's own mass remains;
* ``factor = 1`` is total pooling — ``H_cross`` is then identical across observer
  modes on the same segment;
* only the **cross** term is filtered to ``layer == "street"``. A ``rail_line``
  segment keeps reporting its own, uncontaminated composition: a passenger in a
  tunnel is co-present with nobody on the avenue above. The filter is on the *layer column*, never on the
  accident that synthetic metro nodes carry negative ids.

Pooling runs across ``mode`` **only**, within ``(category, binding, attribute)``.
The reason is dimensional, not temporal: a catchment trajectory under
``weight_mode="unit"`` deposits its origin's whole vector once *per reachable
POI*, so its mass is inflated by a POI count and adding it to a commute flow's
workers would be adding unlike units.

**Scope, stated plainly** (P7.1): the access bindings use a single
``routing.access_mode`` for the whole run, so ``nearest`` and ``catchment``
contribute exactly one mode and this metric reduces identically to ``exposure``
there. It is non-trivial **only for ``binding="od"``**, and only when the OD file
mixes modes — which is where real modal data and income segregation both live.
Two more things a reader needs to be told: ``cross_mode_exposure`` can land
*below* ``exposure`` (adding a more concentrated mode lowers the entropy), and
the measure is strongly asymmetric — 10 bus passengers among 1000 pedestrians
barely move the pedestrians' number while the pedestrians dominate the
passengers'. The ``walk`` rows looking almost unchanged is the expected result,
not a bug.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from segx.metrics._segments import per_segment_values
from segx.metrics.base import Metric
from segx.metrics.entropy import shannon_entropy
from segx.metrics.registry import register_metric

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

__all__ = [
    "CrossModeExposureMetric",
    "cross_mode_combination_counts",
    "cross_mode_pair_rates",
    "cross_mode_stats",
]

_PARTITION_KEYS: tuple[str, ...] = ("mode", "category", "binding", "attribute")
_SEGMENT_KEYS: tuple[str, ...] = (*_PARTITION_KEYS, "u", "v", "key")

#: The keys the cross-mode pool is grouped over. Deliberately **without** ``mode``
#: (that is the dimension being pooled) and without ``layer`` (whose filtering is
#: applied per row, so a colliding rail segment cannot leak into a street one).
#: ``corridor`` (P7.2) replaces the raw ``(u, v, key)`` when the run resolved a
#: canonical street space — see :func:`_pool_keys`.
_POOL_KEYS: tuple[str, ...] = ("category", "binding", "attribute", "u", "v", "key", "class")
_POOL_KEYS_CORRIDOR: tuple[str, ...] = ("category", "binding", "attribute", "corridor", "class")


def _pool_keys(edge_table) -> tuple[str, ...]:
    """Pool over the canonical corridor when there is one, else over the raw edge.

    The fallback is what keeps every pre-P7.2 caller and every synthetic fixture
    working unchanged: with no ``corridor`` column the pool is the exact
    ``(u, v, key)`` join it always was.
    """
    return _POOL_KEYS_CORRIDOR if "corridor" in getattr(edge_table, "columns", ()) else _POOL_KEYS


def _segment_columns(edge_table) -> list[str]:
    """The columns identifying a *poolable* segment, for the coverage audits."""
    return ["corridor"] if "corridor" in getattr(edge_table, "columns", ()) else ["u", "v", "key"]


def cross_mode_stats(edge_table) -> dict:
    """Coverage audit for the cross-mode join — how much of it actually joined.

    Cross-mode mixing can only happen where two modes deposit into the **same
    pool**. Before P7.2 the pool was the raw ``(u, v, key)``, and in a real city
    two modes essentially never share one: ``walk`` and ``drive`` are separate
    OSM extracts simplified independently, and — the dominant effect — where
    sidewalks are mapped as their own ways the pedestrian and the car are on
    different geometries entirely. Measured on Philadelphia: ``walk ∩ drive`` was
    **6 segments out of 448**. P7.2 pools over the canonical ``corridor``
    instead, which is what these numbers now count.

    The failure is invisible in the output, because it does not look like a zero:
    it looks like ``cross_mode_exposure == exposure``, which reads as the
    *finding* "the modes don't mix here". Hence these numbers, the per-pair
    breakdown in :func:`cross_mode_pair_rates`, and the warning in
    :meth:`CrossModeExposureMetric.compute`.

    Returns ``n_modes_present``, ``n_segments_total``, ``n_segments_multimode``,
    ``cross_mode_join_rate`` (the share of street pools carrying more than one
    mode) and ``pair_join_rates``.
    """
    import pandas as pd

    stats = {
        "n_modes_present": 0,
        "n_segments_total": 0,
        "n_segments_multimode": 0,
        "cross_mode_join_rate": 0.0,
        "pair_join_rates": {},
    }
    if edge_table is None or edge_table.empty:
        return stats
    stats["n_modes_present"] = int(edge_table["mode"].nunique())

    street = (
        edge_table["layer"].eq("street") if "layer" in edge_table.columns else pd.Series(True, index=edge_table.index)
    )
    seg_cols = _segment_columns(edge_table)
    group = ["category", "binding", *seg_cols]
    segments = edge_table.loc[street, ["mode", *group]]
    if segments.empty:
        return stats
    modes_per_segment = segments.drop_duplicates().groupby(group, sort=False).size()
    n_total = int(len(modes_per_segment))
    n_multimode = int((modes_per_segment > 1).sum())
    stats["n_segments_total"] = n_total
    stats["n_segments_multimode"] = n_multimode
    stats["cross_mode_join_rate"] = (n_multimode / n_total) if n_total else 0.0
    stats["pair_join_rates"] = cross_mode_pair_rates(edge_table)
    return stats


def cross_mode_pair_rates(edge_table) -> dict[str, dict]:
    """Join coverage **per mode pair** — the number the scalar rate cannot replace.

    The Philadelphia run reported ``cross_mode_join_rate = 0.64``, which reads
    like success. Broken out, ``transit ∩ walk`` was 1463 pools and
    ``walk ∩ drive`` was **six** — the 64 % was carried entirely by a join that is
    true by construction (``build_transit_graph`` copies the walk graph, so a
    transit trip's walking legs *are* walk ids). A scalar cannot show that; this
    can, and it is what the per-pair warning fires on.

    Returns ``{"drive|walk": {"n_shared": int, "n_pools_a": int, "n_pools_b": int,
    "rate": float}}``, keyed with the two modes sorted — the same symmetric
    spelling as ``metrics.cross_mode.overrides``. ``rate`` is shared over the
    *smaller* of the two mode's pool counts, so a mode covering few streets is not
    penalised for the other covering many.
    """
    import pandas as pd

    if edge_table is None or edge_table.empty:
        return {}
    street = (
        edge_table["layer"].eq("street") if "layer" in edge_table.columns else pd.Series(True, index=edge_table.index)
    )
    seg_cols = _segment_columns(edge_table)
    segments = edge_table.loc[street, ["mode", *seg_cols]].drop_duplicates()

    # Seeded from every mode in the table, not only those with street rows: a mode
    # that deposits *nothing* on a street (all of its mass on `rail_line`, or its
    # ids matching nothing) has an empty pool set, and its pairs must still be
    # reported as zero rather than vanishing from the audit.
    pools: dict[str, set] = {str(mode): set() for mode in pd.unique(edge_table["mode"])}
    for mode, group in segments.groupby("mode", sort=False):
        pools[str(mode)] = set(map(tuple, group[seg_cols].to_numpy().tolist()))
    out: dict[str, dict] = {}
    modes = sorted(pools)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            shared = len(pools[a] & pools[b])
            smaller = min(len(pools[a]), len(pools[b]))
            out["|".join((a, b))] = {
                "n_shared": shared,
                f"n_pools_{a}": len(pools[a]),
                f"n_pools_{b}": len(pools[b]),
                "rate": (shared / smaller) if smaller else 0.0,
            }
    return out


def cross_mode_combination_counts(edge_table):
    """The join audit broken out: how many pools carry *which* set of modes.

    :func:`cross_mode_stats` collapses the join to one number, which answers "did
    the modes meet at all" but not "where". This returns one row per **exact**
    combination — ``drive`` alone, ``drive+walk``, ``drive+transit+walk`` — so a
    low join rate can be read: a city whose 36 % single-mode segments are all
    ``walk`` is a city of pedestrian side-streets, while one where they split
    evenly across the three modes looks much more like three id spaces that never
    matched.

    Grouping keeps ``layer``, so the ``rail_line`` segments that the metric
    deliberately excludes from pooling (a passenger in a tunnel is co-present
    with nobody above — D4) are *shown* rather than silently dropped. A segment
    has exactly one layer, so this cannot split a segment across rows.

    Columns: ``layer``, ``category``, ``binding``, ``modes`` (``+``-joined,
    sorted), ``n_modes``, ``n_segments``. Sorted by layer, then mode count, then
    name — so the single-mode rows come first and the table reads top to bottom.
    """
    import pandas as pd

    columns = ["layer", "category", "binding", "modes", "n_modes", "n_segments"]
    if edge_table is None or edge_table.empty:
        return pd.DataFrame(columns=columns)

    df = edge_table
    layer = df["layer"] if "layer" in df.columns else pd.Series("street", index=df.index)
    seg_cols = _segment_columns(df)
    segments = pd.DataFrame(
        {
            "layer": layer,
            "category": df["category"],
            "binding": df["binding"],
            **{col: df[col] for col in seg_cols},
            "mode": df["mode"],
        }
    ).drop_duplicates()

    keys = ["layer", "category", "binding", *seg_cols]
    combos = segments.sort_values("mode").groupby(keys, sort=False)["mode"].agg("+".join).reset_index(name="modes")
    out = combos.groupby(["layer", "category", "binding", "modes"], sort=False).size().reset_index(name="n_segments")
    out["n_modes"] = out["modes"].str.count(r"\+") + 1
    return out[columns].sort_values(["layer", "n_modes", "modes"], ignore_index=True)


def _warn_on_empty_joins(stats: dict) -> None:
    """Warn about **each** mode pair that never meets, not just a global zero.

    The pre-P7.2 guard fired only when *no* street segment carried two modes. On
    the Philadelphia run the global rate was 0.64 and ``walk ∩ drive`` was exactly
    zero, so it stayed silent while the drive numbers were meaningless. Naming the
    dead pair is the whole point.
    """
    if stats.get("n_modes_present", 0) <= 1:
        return
    pairs = stats.get("pair_join_rates") or {}
    dead = sorted(pair for pair, info in pairs.items() if not info.get("n_shared"))
    if not dead:
        return
    scope = "every" if len(dead) == len(pairs) else "some"
    warnings.warn(
        f"cross_mode_exposure: {scope} mode pair never shares a street pool — {', '.join(dead)} "
        f"(over {stats.get('n_segments_total', 0)} pools). For those pairs the metric equals plain "
        "`exposure`, which reads as the finding 'the modes never meet' but may instead mean their "
        "segments never joined: walk and drive are separate OSM extracts, and where sidewalks are "
        "mapped as their own ways the pedestrian and the car are on different geometries. Check "
        "stats['cross_mode'] — corridor_match_rate and pair_join_rates — before interpreting this.",
        UserWarning,
        stacklevel=3,
    )


@register_metric
class CrossModeExposureMetric(Metric):
    """Per-group mean segment diversity with other modes counted at a discount. P7.1."""

    name = "cross_mode_exposure"
    description = (
        "Exposure after also counting the people on the same street who travel by another mode (a "
        "pedestrian and a bus passenger on one avenue), discounted by metrics.cross_mode.factor "
        "(default 0.5). Non-trivial only for OD flows that mix modes; it can be lower than "
        "exposure."
    )
    formula = (
        "Pᵢᵉᶠᶠ(s, m) = Σₘ′ f(m, m′) · Pᵢ(s, m′)   with f(m, m) = 1\n"
        "E_g(m) = Σₛ P_g(s, m) · H(Pᵉᶠᶠ(s, m)) / Σₛ P_g(s, m)"
    )

    def compute(self, edge_table, settings: Settings):
        import numpy as np
        import pandas as pd

        columns = [*_PARTITION_KEYS, "class", self.name]
        if edge_table.empty:
            return pd.DataFrame(columns=columns)

        df = edge_table.reset_index(drop=True)
        cfg = settings.metrics.cross_mode
        mass = df["mass"].astype(float)
        street = df["layer"].eq("street") if "layer" in df.columns else pd.Series(True, index=df.index)
        pool_keys = list(_pool_keys(df))

        stats = cross_mode_stats(df)
        _warn_on_empty_joins(stats)

        # Per (pool, mode), the mode's representative mass — the **length-weighted
        # mean** over that mode's edges in the pool, not their sum. A corridor holds
        # several sidewalk edges per roadway block, so summing would inflate the
        # pedestrian term by however finely OSM happened to segment the sidewalk;
        # the mean makes the number a property of the city instead. With one edge
        # per mode per pool it reduces to that edge's mass, i.e. exactly the pre-P7.2
        # behaviour.
        repr_by_mode = self._pooled_representatives(df, mass, street, pool_keys)

        records: list[dict] = []
        for mode in pd.unique(df["mode"]):  # at most a handful of modes
            cross = pd.Series(0.0, index=df.index)
            for other in repr_by_mode.columns:
                if other == mode:
                    continue
                factor = cfg.factor_for(mode, other)
                if factor:
                    cross = cross + factor * repr_by_mode[other]
            # Only the *cross* term is layer-filtered: a non-street row keeps its own
            # mass, so rail segments stay uncontaminated and `factor = 0` is identical
            # to `exposure` everywhere.
            effective = mass + cross.where(street, 0.0)

            own = df["mode"].eq(mode)
            sub = df.loc[own, [*_PARTITION_KEYS, "u", "v", "key", "class"]].copy()
            sub["mass"] = mass[own]
            sub["effective"] = effective[own]

            seg_h = per_segment_values(sub, _SEGMENT_KEYS, "effective", shannon_entropy, "_H")
            merged = sub.merge(seg_h, on=list(_SEGMENT_KEYS), how="left")

            for keys, group in merged.groupby([*_PARTITION_KEYS, "class"], sort=False):
                presence = group["mass"].to_numpy(dtype=float)
                h = group["_H"].to_numpy(dtype=float)
                mask = (presence > 0.0) & ~np.isnan(h)
                if not mask.any():
                    value = float("nan")
                else:
                    w = presence[mask]
                    value = float((w * h[mask]).sum() / w.sum())
                records.append({**dict(zip([*_PARTITION_KEYS, "class"], keys, strict=True)), self.name: value})

        return pd.DataFrame.from_records(records, columns=columns)

    @staticmethod
    def _pooled_representatives(df, mass, street, pool_keys: list[str]):
        """Per pool and per mode, that mode's representative mass in the pool.

        The **length-weighted mean** ``Σ mass·len / Σ len`` over the mode's street
        edges in the pool. Granularity-invariant by construction: whether OSM split
        a block's sidewalk into one edge or five, the pedestrian mass the corridor
        reports is the same. Falls back to the plain mean when the pool carries no
        length at all (synthetic fixtures build graphs without ``length``), which is
        a real answer rather than a division by zero.

        Returns a frame aligned to ``df``'s index with one column per mode, zero
        where that mode is absent from the row's pool.
        """
        import numpy as np
        import pandas as pd

        modes = list(pd.unique(df["mode"]))
        zeros = pd.DataFrame(0.0, index=df.index, columns=modes)
        street_rows = df.index[street]
        if len(street_rows) == 0:
            return zeros

        length = df["length_m"].astype(float) if "length_m" in df.columns else pd.Series(0.0, index=df.index)
        work = pd.DataFrame(
            {
                **{key: df.loc[street_rows, key] for key in pool_keys},
                "mode": df.loc[street_rows, "mode"],
                "wm": mass.loc[street_rows] * length.loc[street_rows],
                "w": length.loc[street_rows],
                "m": mass.loc[street_rows],
            }
        )
        grouped = (
            work.groupby([*pool_keys, "mode"], sort=False, dropna=False)
            .agg(wm=("wm", "sum"), w=("w", "sum"), total=("m", "sum"), n=("m", "size"))
            .reset_index()
        )
        weighted = grouped["w"] > 0.0
        grouped["repr"] = np.where(
            weighted,
            grouped["wm"] / grouped["w"].where(weighted, 1.0),
            grouped["total"] / grouped["n"],
        )
        wide = grouped.pivot_table(index=pool_keys, columns="mode", values="repr", fill_value=0.0)

        aligned = wide.reindex(pd.MultiIndex.from_frame(df[pool_keys]))
        aligned.index = df.index
        for mode in modes:
            if mode not in aligned.columns:
                aligned[mode] = 0.0
        return aligned[modes].fillna(0.0)
