"""OD destination spread — how many destinations each class actually reaches. P7.5.

Group the OD table by origin and
read, per demographic class, how wide a set of destinations that class reaches.

This is an analysis of the **input**, not of the routed network. Nothing here
touches a graph: it runs before ingest is finished, in seconds, and answers a
question the table already contains the answer to — *is the unequal reach in the
data, or does it only appear once we route it?* If the low-income workers of a
block reach four workplaces and the high-income workers of the same block reach
forty, that is segregation visible in the commute matrix itself, before a single
street is considered.

The measures per ``(origin, category, attribute, class)``:

===========================  ====================================================
``n_destinations``           distinct destinations the class reaches from here.
``effective_destinations``   ``2^H`` over the class's destination shares — the
                             number of *equally used* destinations that would
                             produce the same concentration. The honest headline:
                             a class with 40 destinations and 97 % of its flow in
                             one of them has an effective 1.2, not 40.
``rarefied_destinations``    the same count with every class cut to the **same
                             number of people** (``rarefaction_n``). **The raw
                             counts are confounded by group size** — a class with
                             300 workers touches more workplaces than one with 3
                             no matter how confined it is — and on the measured
                             Philadelphia clip that confound explains essentially
                             the whole raw gap. This is the column to compare
                             classes on; the raw count is what it is built from.
                             See :func:`_rarefied_destinations`.
``rarefaction_n``            the common head-count used at that origin (the mass
                             of its smallest present class), so a mean rarefied
                             count can be read for what it is.
``top_share``                share of the class's flow in its single largest
                             destination.
``concentration``            ``Σ pᵢ²`` over destinations (Herfindahl) — the
                             probability two randomly drawn trips from this origin
                             and class end in the same place.
``mean_distance_m``          flow-weighted **straight-line** origin→destination
                             distance. Not a network distance: routing has not
                             happened yet, and saying so is the point (a metric
                             that looks like a travel distance but is not would be
                             indistinguishable in a table — invariant #7's
                             discipline).
``radius_of_gyration_m``     the standard distance of the class's destinations
                             about their own flow-weighted centre: the *size of
                             the activity space*, independent of how far that
                             space sits from home. A class commuting 8 km to one
                             single district has a large mean distance and a
                             radius of gyration near zero.
===========================  ====================================================

Then :func:`class_destination_overlap` asks the question the per-class counts
cannot: **do the classes go to the same places?** Two classes can each reach
thirty destinations and share none of them. Per origin and class pair it reports
the Jaccard overlap of the destination sets, the cosine similarity of the
destination *share* vectors (so a shared destination that one class barely uses
does not read as agreement), and the directional ``shared_share_*`` — the share
of one class's flow that lands where the other also goes. This is the same
lesson P7.1 learned from ``cross_mode_join_rate``: a scalar per class cannot show
that two coverages never meet, and the pairwise breakdown can.

**Read the overlap numbers against each other, not against 1.0.** With 24 522
candidate work blocks and a few dozen destinations per class per origin, two
classes drawn at random would share almost nothing either, so a Jaccard of 0.013
is not by itself evidence of segregation — the *ordering* of the pairs is
(measured on Philadelphia: ``low|high`` 0.0128 is the lowest, ``mid|high`` 0.0183
the highest, i.e. the income extremes share the least). A size- and
sparsity-corrected overlap baseline is not implemented yet.

**Not every OD file can answer this.** The question needs per-class counts *per
flow* (LODES ``SE01``/``SE02``/``SE03``: workers per earnings class per home→work
pair), i.e. ``demographics.attribute_source == "od"``. When the class vector
instead lives on the demographics file, every flow out of an origin carries that
origin's whole composition, so every class would report the identical destination
set by construction — a tautology, not a finding.
:func:`od_spread_unavailable_reason` says so in words and
:func:`analyse_od_spread` raises; the orchestrator checks the reason first and
skips the stage rather than failing an ordinary run.

**NaN, never zero** (invariant #2): a class with no mass at an origin gets ``NaN``
measures and ``mass = 0.0`` — "nobody of this class lives here" is not
"this class goes nowhere from here", and ``n_destinations = 0`` would say the
second.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from segx.errors import SchemaError
from segx.io.demographics import class_columns_for

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from segx.config import Settings
    from segx.io.od_pairs import LoadedODPairs

__all__ = [
    "ODSpreadResult",
    "analyse_od_spread",
    "class_destination_overlap",
    "class_flows",
    "destination_spread",
    "od_spread_unavailable_reason",
    "summarize_destination_overlap",
    "summarize_destination_spread",
]

#: One flow of one class out of one origin. ``dx``/``dy`` are the destination's
#: offset *from its origin* in metres, not absolute coordinates: the radius of
#: gyration is translation-invariant anyway, and differencing here keeps the
#: ``Σ p·x²`` sums off the 1e11 magnitudes that UTM easting squared would reach,
#: where the cancellation in ``Σ p·x² − (Σ p·x)²`` eats the precision the answer
#: is made of.
FLOW_COLUMNS: tuple[str, ...] = (
    "origin_id",
    "category",
    "attribute",
    "class",
    "dest_id",
    "mass",
    "distance_m",
    "dx",
    "dy",
)

#: The grain of the per-origin table: the OD table grouped by origin, per class.
ORIGIN_KEYS: tuple[str, ...] = ("origin_id", "category", "attribute", "class")

#: Per-origin measure columns, in report order.
SPREAD_COLUMNS: tuple[str, ...] = (
    "mass",
    "n_destinations",
    "effective_destinations",
    "rarefied_destinations",
    "rarefaction_n",
    "top_share",
    "concentration",
    "mean_distance_m",
    "max_distance_m",
    "radius_of_gyration_m",
)


@dataclass(frozen=True, slots=True)
class ODSpreadResult:
    """Everything the OD destination-spread analysis produces.

    ``per_origin`` is one row per ``(origin_id, category, attribute, class)`` —
    the deliverable table, joinable back onto the origin geometry for a map of
    *whose* reach is narrow and where. ``summary`` collapses it to one row per
    ``(category, attribute, class)``, mass-weighted so the averages are per
    person rather than per block. ``overlap`` / ``overlap_summary`` are the
    pairwise "do they go to the same places" tables (empty when the attribute has
    a single class, or when the pair count exceeded
    ``analysis.od_spread.max_class_pairs``). ``stats`` records what the table was
    and how it was read.
    """

    per_origin: pd.DataFrame
    summary: pd.DataFrame
    overlap: pd.DataFrame
    overlap_summary: pd.DataFrame
    stats: dict


# --------------------------------------------------------------------------- #
# Availability                                                                 #
# --------------------------------------------------------------------------- #


def _class_columns(od_pairs: LoadedODPairs, settings: Settings) -> dict[str, list[str]]:
    """``attribute -> per-class columns``, all of them present on the OD frame.

    Raises :class:`~segx.errors.SchemaError` naming what is missing and why it
    matters, rather than returning an empty mapping that would silently produce
    an empty analysis.
    """
    df = od_pairs.gdf
    schema = settings.demographics
    if schema.attribute_source != "od":
        raise SchemaError(
            "OD destination spread needs per-class counts on each *flow*, but "
            f"demographics.attribute_source is {schema.attribute_source!r}: the class vector comes "
            "from the demographics file, so every flow out of an origin carries that origin's whole "
            "composition and every class would report the identical destination set by construction. "
            "Set demographics.attribute_source = 'od' and declare the per-flow class columns "
            "(DemographicAttribute.class_columns, e.g. LODES SE01/SE02/SE03).",
            details={"attribute_source": schema.attribute_source},
        )
    out: dict[str, list[str]] = {}
    missing: list[str] = []
    for attr in schema.attributes:
        cols = class_columns_for(attr)
        absent = [c for c in cols if c not in df.columns]
        if absent:
            missing.extend(absent)
            continue
        out[attr.name] = cols
    if not out:
        raise SchemaError(
            f"the OD file carries none of the declared per-class columns {sorted(set(missing))}, so "
            "'which classes go where' cannot be read from it. Available columns: "
            f"{sorted(map(str, df.columns))}",
            details={"missing": sorted(set(missing)), "available": sorted(map(str, df.columns))},
        )
    if missing:
        warnings.warn(
            f"OD destination spread: skipping attribute(s) whose class columns are absent from the "
            f"OD file ({sorted(set(missing))}); analysing {sorted(out)} only.",
            UserWarning,
            stacklevel=3,
        )
    return out


def od_spread_unavailable_reason(od_pairs: LoadedODPairs | None, settings: Settings) -> str | None:
    """Why this analysis cannot run on these inputs, or ``None`` if it can.

    The orchestrator asks this before announcing the stage: an ordinary run whose
    demographics live on the census file must not fail because a *diagnostic* of
    the OD table is inapplicable to it. A caller who asked for the analysis
    directly (the ``segx od-report`` command) gets the raise instead, because
    there silence would be the wrong answer.
    """
    if od_pairs is None:
        return "no OD pairs were supplied"
    if not settings.analysis.od_spread.enabled:
        return "analysis.od_spread.enabled is false"
    try:
        _class_columns(od_pairs, settings)
    except SchemaError as exc:
        return exc.message
    return None


# --------------------------------------------------------------------------- #
# The long per-class flow table                                                #
# --------------------------------------------------------------------------- #


def class_flows(od_pairs: LoadedODPairs, settings: Settings) -> pd.DataFrame:
    """The OD table exploded to one row per ``(origin, destination, class)``.

    Zero-mass entries are dropped rather than carried: a LODES flow lists a count
    for every earnings class and most are zero, and a class that sends nobody
    down a flow is not a destination of that class. Duplicate
    ``(origin, destination)`` rows — the same pair appearing twice because the
    file splits it by travel mode, say — are summed, because two ways of reaching
    one place are still one place.

    Columns: :data:`FLOW_COLUMNS`.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    df = od_pairs.gdf
    columns_by_attribute = _class_columns(od_pairs, settings)
    min_mass = float(settings.analysis.od_spread.min_mass)

    if "dest_id" not in df.columns:
        raise SchemaError(
            "the loaded OD frame has no `dest_id` column; destinations cannot be counted. "
            "Reload the OD pairs with segx.io.od_pairs.load_od_pairs.",
            details={"available": sorted(map(str, df.columns))},
        )

    origin = gpd.GeoSeries(df.geometry.to_numpy(), crs=df.crs)
    dest = gpd.GeoSeries(df["dest_geom"].to_numpy(), crs=df.crs)
    dx = dest.x.to_numpy() - origin.x.to_numpy()
    dy = dest.y.to_numpy() - origin.y.to_numpy()
    base = pd.DataFrame(
        {
            "origin_id": df["origin_id"].astype(str).to_numpy(),
            "category": df["category"].astype(str).to_numpy(),
            "dest_id": df["dest_id"].astype(str).to_numpy(),
            "distance_m": np.hypot(dx, dy),
            "dx": dx,
            "dy": dy,
        }
    )

    parts: list[pd.DataFrame] = []
    for attribute, cols in columns_by_attribute.items():
        labels = settings.demographics.get(attribute).class_labels
        for label, column in zip(labels, cols, strict=True):
            mass = df[column].to_numpy(dtype=float)
            keep = np.isfinite(mass) & (mass > min_mass)
            if not keep.any():
                continue
            part = base.loc[keep].copy()
            part["attribute"] = attribute
            part["class"] = label
            part["mass"] = mass[keep]
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=list(FLOW_COLUMNS))

    flows = pd.concat(parts, ignore_index=True)
    collapsed = flows.groupby(["origin_id", "category", "attribute", "class", "dest_id"], as_index=False).agg(
        mass=("mass", "sum"),
        distance_m=("distance_m", "first"),
        dx=("dx", "first"),
        dy=("dy", "first"),
    )
    return collapsed[list(FLOW_COLUMNS)]


# --------------------------------------------------------------------------- #
# Per-origin spread                                                            #
# --------------------------------------------------------------------------- #


def destination_spread(flows: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per ``(origin, category, attribute, class)`` destination-spread measures.

    Every measure is a groupby *sum* of a per-row term, so the whole table is one
    pass over the flows however many million rows it has — no per-origin Python
    loop. The class grid is then completed: an origin where a class has no mass
    gets a row with ``mass = 0.0`` and ``NaN`` measures, because its absence is a
    different fact from a narrow reach (invariant #2).

    Columns: ``origin_id``, ``category``, ``attribute``, ``class`` +
    :data:`SPREAD_COLUMNS`.
    """
    import numpy as np
    import pandas as pd

    keys = list(ORIGIN_KEYS)
    columns = [*keys, *SPREAD_COLUMNS]
    if flows.empty:
        return pd.DataFrame(columns=columns)

    total = flows.groupby(keys, sort=False)["mass"].transform("sum")
    p = flows["mass"].to_numpy(dtype=float) / total.to_numpy(dtype=float)
    dx = flows["dx"].to_numpy(dtype=float)
    dy = flows["dy"].to_numpy(dtype=float)
    work = pd.DataFrame(
        {
            **{key: flows[key] for key in keys},
            "mass": flows["mass"].astype(float),
            "p": p,
            # -p·log₂p, the Shannon term. p > 0 by construction (zero-mass flows
            # were dropped upstream), so no guard is needed here.
            "plogp": -p * np.log2(p),
            "p2": p * p,
            "pd": p * flows["distance_m"].to_numpy(dtype=float),
            "distance_m": flows["distance_m"].astype(float),
            "px": p * dx,
            "py": p * dy,
            "px2": p * dx * dx,
            "py2": p * dy * dy,
        }
    )
    agg = work.groupby(keys, sort=False).agg(
        mass=("mass", "sum"),
        n_destinations=("p", "size"),
        entropy=("plogp", "sum"),
        top_share=("p", "max"),
        concentration=("p2", "sum"),
        mean_distance_m=("pd", "sum"),
        max_distance_m=("distance_m", "max"),
        px=("px", "sum"),
        py=("py", "sum"),
        px2=("px2", "sum"),
        py2=("py2", "sum"),
    )
    agg = agg.reset_index()

    entropy = agg["entropy"].to_numpy(dtype=float).clip(min=0.0)
    agg["effective_destinations"] = np.exp2(entropy)
    variance = (agg["px2"] - agg["px"] ** 2) + (agg["py2"] - agg["py"] ** 2)
    # Clamp floating-point dust: the two sums are equal to within rounding when
    # every destination coincides, and a negative variance there is arithmetic,
    # not geography.
    agg["radius_of_gyration_m"] = np.sqrt(variance.to_numpy(dtype=float).clip(min=0.0))
    agg["n_destinations"] = agg["n_destinations"].astype(float)

    agg = agg.merge(_rarefied_destinations(flows), on=keys, how="left")
    return _complete_class_grid(agg[columns], settings)


def _rarefied_destinations(flows: pd.DataFrame) -> pd.DataFrame:
    """Destinations per class **at a common number of people** — the size control.

    The raw count is confounded by group size, and badly. A class with 300 workers
    leaving a block touches more workplaces than one with 3 however confined it
    is, so "the high-income group reaches 2.7× more destinations" may be no more
    than "the high-income group is 2.7× larger here" — which is what the measured
    Philadelphia clip turned out to be, and nothing in a table of counts would
    have shown it. Any comparison of destination counts between groups of
    different sizes is meaningless without this correction.

    So the classes are compared at equal head-count, the way species richness is
    compared between samples of different sizes — **Hurlbert (1971) rarefaction**.
    For each origin, ``n`` is the mass of its **smallest present class**, and every
    class at that origin is asked how many destinations *n* of its own members
    would be expected to reach, drawing those *n* people from its own ``M``
    without replacement:

    .. code-block:: text

        rarefied(origin, class) = Σ_d [ 1 − C(M − m_d, n) / C(M, n) ]

    with ``m_d`` the class's mass at destination *d* and ``M`` its total. Each
    class is evaluated against **its own** distribution — nothing about one class
    enters another's number — and at the same ``n``, so the comparison within an
    origin is fair. ``n`` varies between origins, which is why the summary keeps
    ``rarefaction_n``: a mean rarefied count over origins is a mean of fair
    comparisons, not itself a count of anything.

    Sampling **without** replacement is not a detail here. The with-replacement
    closed form ``Σ_d [1 − (1 − p_d)^n]`` looks equivalent and is not: drawing 2
    of a 2-person class collides half the time while drawing 2 of a 10-person
    class collides a tenth of the time, so it charges the smaller class for
    collisions between people who are in fact distinct — reintroducing, in the
    correction, a milder version of the very group-size bias being corrected.
    Two workers commuting to two workplaces are two destinations whichever class
    they belong to, and only the hypergeometric form says so.

    Computed through ``lgamma`` so it stays vectorised over millions of flows;
    ``C(M − m_d, n)`` is zero whenever fewer than ``n`` of the class's members
    work anywhere else, and that destination is then certainly sampled (term 1).
    """
    import numpy as np
    import pandas as pd
    from scipy.special import gammaln

    keys = list(ORIGIN_KEYS)
    if flows.empty:
        return pd.DataFrame(columns=[*keys, "rarefied_destinations", "rarefaction_n"])

    origin_keys = ["origin_id", "category", "attribute"]
    sizes = flows.groupby(keys, sort=False)["mass"].sum().reset_index(name="class_mass")
    # The common sample size: the smallest class actually present at the origin.
    # Floored at one person (a zero would make every class report no destinations)
    # and capped at the class's own mass, which only binds for sub-unit weights.
    common = sizes.groupby(origin_keys, sort=False)["class_mass"].transform("min").clip(lower=1.0)
    sizes["rarefaction_n"] = np.minimum(common, sizes["class_mass"])

    work = flows[[*keys, "mass"]].merge(sizes, on=keys, how="left")
    m = work["mass"].to_numpy(dtype=float)
    total = work["class_mass"].to_numpy(dtype=float)
    n = work["rarefaction_n"].to_numpy(dtype=float)

    rest = total - m  # the class's members working anywhere but here
    can_miss = rest >= n  # otherwise this destination is certainly sampled
    safe = np.where(can_miss, rest, n)  # keep lgamma's argument positive
    log_ratio = gammaln(safe + 1.0) + gammaln(total - n + 1.0) - gammaln(safe - n + 1.0) - gammaln(total + 1.0)
    work["term"] = np.where(can_miss, 1.0 - np.exp(log_ratio), 1.0)

    rarefied = work.groupby(keys, sort=False)["term"].sum().reset_index(name="rarefied_destinations")
    return rarefied.merge(sizes[[*keys, "rarefaction_n"]], on=keys, how="left")


def _complete_class_grid(per_origin: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add the ``(origin, class)`` rows the OD table never mentions, as ``NaN``.

    Without this the table silently omits the most extreme case there is: a class
    that no flow out of an origin carries at all. Omission reads as "not
    applicable"; an explicit ``mass = 0.0`` row with ``NaN`` measures reads as
    what it is — nobody of this class lives here, so their reach is undefined
    rather than zero.
    """
    import pandas as pd

    classes = pd.DataFrame(
        [
            {"attribute": attr.name, "class": label}
            for attr in settings.demographics.attributes
            for label in attr.class_labels
        ]
    )
    present = per_origin[["attribute", "class"]].drop_duplicates()
    # Only complete the grid for attributes the analysis actually ran on — an
    # attribute skipped for missing columns must stay absent, not appear as a
    # table of NaN.
    classes = classes[classes["attribute"].isin(present["attribute"])]
    if classes.empty:
        return per_origin

    origins = per_origin[["origin_id", "category"]].drop_duplicates()
    grid = origins.merge(classes, how="cross")
    out = grid.merge(per_origin, on=list(ORIGIN_KEYS), how="left")
    out["mass"] = out["mass"].fillna(0.0)
    return out[[*ORIGIN_KEYS, *SPREAD_COLUMNS]].reset_index(drop=True)


def summarize_destination_spread(per_origin: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-origin table to one row per ``(category, attribute, class)``.

    The per-origin means are **mass-weighted**, so they answer "how spread out is
    the typical *person* of this class", not "the typical block": a block with
    three workers should not outvote one with three hundred.

    Two further columns come from ``flows`` rather than from the per-origin
    table, and they answer a different question — not how far each origin's
    members spread, but how much of the city the class occupies in total:
    ``n_destinations_pooled`` (distinct destinations reached from anywhere) and
    ``effective_destinations_pooled`` (``2^H`` over the class's whole
    destination distribution). A class can be tightly confined at every single
    origin and still cover the map, or the reverse; both readings are needed.

    ``rarefied_destinations_paired`` repeats the size-controlled count over the
    **origins where every class is present**. ``rarefied_destinations`` is a fair
    comparison *within* each origin, but averaging it per class still averages
    over different sets of origins — a class living in fewer, denser blocks is
    being asked a different question than one living everywhere. The paired
    column asks all of them the same one, over ``n_origins_paired`` blocks.
    """
    import numpy as np
    import pandas as pd

    keys = ["category", "attribute", "class"]
    measures = [c for c in SPREAD_COLUMNS if c != "mass"]
    columns = [
        *keys,
        "n_origins",
        "n_origins_absent",
        "mass_total",
        *measures,
        "n_origins_paired",
        "rarefied_destinations_paired",
        "n_destinations_pooled",
        "effective_destinations_pooled",
    ]
    if per_origin.empty:
        return pd.DataFrame(columns=columns)

    paired = _paired_origins(per_origin)
    rows: list[dict] = []
    for key, group in per_origin.groupby(keys, sort=False):
        sub = group[group["mass"] > 0.0]
        record = dict(zip(keys, key, strict=True))
        record["n_origins"] = int(len(sub))
        record["n_origins_absent"] = int(len(group) - len(sub))
        record["mass_total"] = float(sub["mass"].sum())
        weights = sub["mass"].to_numpy(dtype=float)
        for measure in measures:
            values = sub[measure].to_numpy(dtype=float)
            mask = np.isfinite(values) & (weights > 0.0)
            record[measure] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else float("nan")
        both = sub[sub.set_index(["origin_id", "category", "attribute"]).index.isin(paired)]
        record["n_origins_paired"] = int(len(both))
        record["rarefied_destinations_paired"] = _weighted_mean(both["rarefied_destinations"], both["mass"])
        rows.append(record)
    summary = pd.DataFrame.from_records(rows)

    pooled = _pooled_destinations(flows, keys)
    summary = summary.merge(pooled, on=keys, how="left")
    return summary[columns]


def _weighted_mean(values, weights) -> float:
    """Mass-weighted mean, skipping ``NaN`` — ``NaN`` when nothing is left."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & (w > 0.0)
    return float(np.average(v[mask], weights=w[mask])) if mask.any() else float("nan")


def _paired_origins(per_origin: pd.DataFrame):
    """The ``(origin, category, attribute)`` groups where **every** class is present.

    The only subset on which a per-class average is a like-for-like comparison
    between classes rather than between the places each class happens to live in.
    """
    presence = per_origin.assign(_present=per_origin["mass"] > 0.0)
    counts = presence.groupby(["origin_id", "category", "attribute"], sort=False)["_present"].agg(["sum", "size"])
    return counts.index[counts["sum"] == counts["size"]]


def _pooled_destinations(flows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Citywide destination count and effective count per class."""
    import numpy as np
    import pandas as pd

    if flows.empty:
        return pd.DataFrame(columns=[*keys, "n_destinations_pooled", "effective_destinations_pooled"])
    by_dest = flows.groupby([*keys, "dest_id"], sort=False)["mass"].sum().reset_index()
    total = by_dest.groupby(keys, sort=False)["mass"].transform("sum")
    p = by_dest["mass"].to_numpy(dtype=float) / total.to_numpy(dtype=float)
    by_dest["plogp"] = -p * np.log2(p)
    agg = by_dest.groupby(keys, sort=False).agg(n_destinations_pooled=("dest_id", "size"), entropy=("plogp", "sum"))
    agg = agg.reset_index()
    agg["effective_destinations_pooled"] = np.exp2(agg["entropy"].to_numpy(dtype=float).clip(min=0.0))
    agg["n_destinations_pooled"] = agg["n_destinations_pooled"].astype(float)
    return agg[[*keys, "n_destinations_pooled", "effective_destinations_pooled"]]


# --------------------------------------------------------------------------- #
# Pairwise overlap — do the classes go to the same places?                     #
# --------------------------------------------------------------------------- #


def class_destination_overlap(flows: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per origin and class **pair**, how much their destination sets coincide.

    The per-class counts answer "how many places", never "the same places". Two
    classes with thirty destinations each may share all thirty or none, and only
    the second is segregation. Three readings, because they disagree in
    informative ways:

    * ``jaccard`` — set overlap, ignoring how much flow each destination carries.
    * ``cosine`` — similarity of the destination *share* vectors, so a shared
      destination that one class barely uses does not read as agreement.
    * ``shared_share_a`` / ``shared_share_b`` — the share of each class's flow
      landing on destinations the other also reaches. Deliberately directional:
      when a small class sends 90 % of its trips into places a large class also
      uses while the large class sends 5 % of its own into places the small one
      reaches, one number could only report that asymmetry by hiding it.

    Only origins where **both** classes have mass appear: an overlap with an
    absent class is undefined, not zero.
    """
    import numpy as np
    import pandas as pd

    columns = [
        "origin_id",
        "category",
        "attribute",
        "class_a",
        "class_b",
        "n_shared",
        "n_destinations_a",
        "n_destinations_b",
        "jaccard",
        "cosine",
        "shared_share_a",
        "shared_share_b",
        "mass_a",
        "mass_b",
    ]
    if flows.empty:
        return pd.DataFrame(columns=columns)

    per_class = _per_class_shares(flows)
    pairs = _class_pairs(flows, settings)
    if not pairs:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for attribute, class_a, class_b in pairs:
        side_a = per_class[(per_class["attribute"] == attribute) & (per_class["class"] == class_a)]
        side_b = per_class[(per_class["attribute"] == attribute) & (per_class["class"] == class_b)]
        if side_a.empty or side_b.empty:
            continue
        group_keys = ["origin_id", "category", "attribute"]
        totals_a = _per_class_totals(side_a)
        totals_b = _per_class_totals(side_b)
        # Inner join: only origins where both classes are present can be compared.
        both = totals_a.merge(totals_b, on=group_keys, suffixes=("_a", "_b"))
        if both.empty:
            continue

        shared = side_a.merge(side_b, on=[*group_keys, "dest_id"], suffixes=("_a", "_b"))
        if shared.empty:
            # Disjoint destination sets everywhere — a real answer (the classes
            # never meet), so the columns are zeros, not absences.
            merged = both.assign(n_shared=0.0, dot=0.0, shared_share_a=0.0, shared_share_b=0.0)
        else:
            shared["dot"] = shared["p_a"] * shared["p_b"]
            overlap = (
                shared.groupby(group_keys, sort=False)
                .agg(
                    n_shared=("dest_id", "size"),
                    dot=("dot", "sum"),
                    shared_share_a=("p_a", "sum"),
                    shared_share_b=("p_b", "sum"),
                )
                .reset_index()
            )
            merged = both.merge(overlap, on=group_keys, how="left")
            for column in ("n_shared", "dot", "shared_share_a", "shared_share_b"):
                merged[column] = merged[column].fillna(0.0).astype(float)
        union = merged["n_destinations_a"] + merged["n_destinations_b"] - merged["n_shared"]
        merged["jaccard"] = np.where(union > 0, merged["n_shared"] / union.where(union > 0, 1.0), np.nan)
        norm = np.sqrt(merged["concentration_a"].to_numpy() * merged["concentration_b"].to_numpy())
        merged["cosine"] = np.where(norm > 0, merged["dot"].to_numpy() / np.where(norm > 0, norm, 1.0), np.nan)
        merged["class_a"] = class_a
        merged["class_b"] = class_b
        frames.append(merged[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _per_class_shares(flows: pd.DataFrame) -> pd.DataFrame:
    """``flows`` with each row's share ``p`` of its class's outflow from the origin."""
    import pandas as pd

    total = flows.groupby(list(ORIGIN_KEYS), sort=False)["mass"].transform("sum")
    return pd.DataFrame(
        {
            "origin_id": flows["origin_id"],
            "category": flows["category"],
            "attribute": flows["attribute"],
            "class": flows["class"],
            "dest_id": flows["dest_id"],
            "mass": flows["mass"].astype(float),
            "p": flows["mass"].to_numpy(dtype=float) / total.to_numpy(dtype=float),
        }
    )


def _per_class_totals(side: pd.DataFrame) -> pd.DataFrame:
    """Per origin, one class's destination count, mass and ``Σ p²``."""
    work = side.copy()
    work["p2"] = work["p"] * work["p"]
    return (
        work.groupby(["origin_id", "category", "attribute"], sort=False)
        .agg(n_destinations=("dest_id", "size"), mass=("mass", "sum"), concentration=("p2", "sum"))
        .reset_index()
    )


def _class_pairs(flows: pd.DataFrame, settings: Settings) -> list[tuple[str, str, str]]:
    """The ``(attribute, class_a, class_b)`` pairs to compare, in class order.

    Guarded by ``analysis.od_spread.max_class_pairs``: an attribute binned into
    twenty classes costs 190 joins over the whole OD file, and a diagnostic that
    is supposed to run in seconds should say it is declining rather than take
    minutes without warning.
    """
    limit = int(settings.analysis.od_spread.max_class_pairs)
    present = set(map(tuple, flows[["attribute", "class"]].drop_duplicates().to_numpy().tolist()))
    pairs: list[tuple[str, str, str]] = []
    for attribute in flows["attribute"].drop_duplicates():
        labels = [label for label in settings.demographics.get(attribute).class_labels if (attribute, label) in present]
        pairs.extend((attribute, labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels)))
    if len(pairs) > limit:
        warnings.warn(
            f"OD destination spread: {len(pairs)} class pairs exceeds analysis.od_spread.max_class_pairs "
            f"({limit}); the pairwise destination-overlap table is skipped. The per-origin spread table "
            "is unaffected. Raise the limit, or narrow `metrics`/`demographics.attributes`, to get it.",
            UserWarning,
            stacklevel=3,
        )
        return []
    return pairs


def summarize_destination_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    """Collapse the pairwise table to one row per ``(category, attribute, pair)``.

    Weighted by ``mass_a + mass_b`` — the people the comparison is about — so a
    block where two workers of each class meet does not weigh as much as one
    where two hundred do.
    """
    import numpy as np
    import pandas as pd

    keys = ["category", "attribute", "class_a", "class_b"]
    measures = ["jaccard", "cosine", "shared_share_a", "shared_share_b"]
    columns = [*keys, "n_origins", "mass_a", "mass_b", *measures]
    if overlap.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for key, group in overlap.groupby(keys, sort=False):
        weights = (group["mass_a"] + group["mass_b"]).to_numpy(dtype=float)
        record = dict(zip(keys, key, strict=True))
        record["n_origins"] = int(len(group))
        record["mass_a"] = float(group["mass_a"].sum())
        record["mass_b"] = float(group["mass_b"].sum())
        for measure in measures:
            values = group[measure].to_numpy(dtype=float)
            mask = np.isfinite(values) & (weights > 0.0)
            record[measure] = float(np.average(values[mask], weights=weights[mask])) if mask.any() else float("nan")
        rows.append(record)
    return pd.DataFrame.from_records(rows)[columns]


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def analyse_od_spread(od_pairs: LoadedODPairs, settings: Settings) -> ODSpreadResult:
    """Run the whole analysis over a loaded OD table.

    Raises :class:`~segx.errors.SchemaError` when the OD file cannot answer the
    question (see the module docstring); callers that must not fail on that —
    the orchestrator — check :func:`od_spread_unavailable_reason` first.
    """
    flows = class_flows(od_pairs, settings)
    per_origin = destination_spread(flows, settings)
    summary = summarize_destination_spread(per_origin, flows)
    overlap = class_destination_overlap(flows, settings)
    overlap_summary = summarize_destination_overlap(overlap)
    stats = _stats(od_pairs, flows, per_origin, settings)
    _warn_on_degenerate(stats)
    return ODSpreadResult(
        per_origin=per_origin,
        summary=summary,
        overlap=overlap,
        overlap_summary=overlap_summary,
        stats=stats,
    )


def _stats(od_pairs: LoadedODPairs, flows: pd.DataFrame, per_origin: pd.DataFrame, settings: Settings) -> dict:
    """What the table was and how it was read — the provenance of the numbers."""
    df = od_pairs.gdf
    n_origins = int(df["origin_id"].nunique())
    n_flows = int(len(flows))
    stats = {
        "n_od_rows": int(len(df)),
        "n_origins": n_origins,
        "n_destinations": int(df["dest_id"].nunique()) if "dest_id" in df.columns else None,
        "n_class_flows": n_flows,
        "attributes": sorted(flows["attribute"].unique().tolist()) if n_flows else [],
        "min_mass": float(settings.analysis.od_spread.min_mass),
        "flows_per_origin": (int(len(df)) / n_origins) if n_origins else 0.0,
        "n_origin_class_absent": int((per_origin["mass"] <= 0.0).sum()) if not per_origin.empty else 0,
        # `distance_m` here is straight-line, not routed — see the module
        # docstring. Stamped so a table of these numbers cannot be mistaken for
        # travel distances once it has left this process (invariant #7).
        "distance_basis": "euclidean_input_geometry",
    }
    if not per_origin.empty:
        measured = per_origin[per_origin["mass"] > 0.0]
        stats["n_origin_class_measured"] = int(len(measured))
        stats["n_origins_single_destination"] = int((measured["n_destinations"] <= 1).sum())
    return stats


def _warn_on_degenerate(stats: dict) -> None:
    """Warn when the table cannot distinguish origins — the measure is then trivial.

    An OD file with no origin id column gets one synthetic origin per row (see
    :func:`segx.io.od_pairs.load_od_pairs`), and one flow per origin makes every
    class reach exactly one destination with an effective count of 1.0. That is
    arithmetic about the file's shape, not a finding about the city, and it looks
    identical to a real result in a spreadsheet.
    """
    measured = stats.get("n_origin_class_measured", 0)
    single = stats.get("n_origins_single_destination", 0)
    if measured and single == measured:
        warnings.warn(
            f"OD destination spread: every one of the {measured} (origin, class) groups reaches a single "
            "destination, so every spread measure is trivially 1. This usually means the OD file names no "
            "origin id (`od_columns.origin_id` / `origin_id_ref`), in which case each row became its own "
            "origin and there is nothing to group by.",
            UserWarning,
            stacklevel=3,
        )
