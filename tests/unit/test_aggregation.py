"""P4 aggregation + diversity — hand-computed ground truth.

The per-edge accumulator and the paper's diversity formula (Netto 2018 fig. 5)
are checked against values worked out by hand on tiny, controlled trajectories,
plus the weight-conservation invariant.
"""

from __future__ import annotations

import math

import pytest

from segx.aggregation.edge_accumulator import EdgeAccumulator, canonical_edge
from segx.config import DemographicAttribute, DemographicSchema, Settings
from segx.metrics.entropy import ShannonEntropyMetric, segment_diversity
from segx.routing.trajectories import Trajectory
from segx.validation.invariants import check_weight_conservation


def _settings() -> Settings:
    """One categorical income attribute with classes low/high."""
    return Settings(
        demographics=DemographicSchema(
            attributes=[DemographicAttribute(name="income", kind="categorical", categories=["low", "high"])]
        )
    )


def _vec(low: float, high: float) -> dict[str, dict[str, float]]:
    return {"income": {"low": low, "high": high}}


def _traj(origin_id, path, *, binding="nearest", category="park", mode="walk", dest="p") -> Trajectory:
    return Trajectory(origin_id, dest, mode, category, binding, path=path)


# --------------------------------------------------------------------------- #
# segment_diversity (the paper's D = -Σ Pᵢ log₂(Pᵢ/Pₜ) = Pₜ·H)                 #
# --------------------------------------------------------------------------- #


def test_segment_diversity_hand_values():
    assert segment_diversity([8, 0]) == 0.0  # all one class → 0
    assert segment_diversity([2, 2]) == pytest.approx(4.0)  # Pt=4, H=1 → 4
    # {6,2}: -6·log2(0.75) - 2·log2(0.25) = 2.4902 + 4 = 6.4902
    assert segment_diversity([6, 2]) == pytest.approx(6.490225, abs=1e-5)


def test_segment_diversity_zero_mass_is_nan():
    assert math.isnan(segment_diversity([0, 0]))  # undefined, never 0.0


def test_segment_diversity_is_volume_weighted():
    # same proportions (half/half), ten times the volume → ten times the diversity
    assert segment_diversity([20, 20]) == pytest.approx(10 * segment_diversity([2, 2]))


def test_segment_diversity_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        segment_diversity([1, -1])


# --------------------------------------------------------------------------- #
# EdgeAccumulator                                                              #
# --------------------------------------------------------------------------- #


def test_accumulator_sums_and_diversity():
    acc = EdgeAccumulator(_settings())
    # o1 (4 low) walks edges (0,1,0)->(1,2,0); o2 (2 low, 2 high) walks (1,2,0).
    acc.add(_traj("o1", [(0, 1, 0), (1, 2, 0)]), _vec(4, 0))
    acc.add(_traj("o2", [(1, 2, 0)]), _vec(2, 2))

    frame = acc.to_frame()
    mass = {(r.u, r.v, r.key, r["class"]): r.mass for _, r in frame.iterrows()}
    # shared edge (1,2,0): low = 4+2 = 6, high = 0+2 = 2
    assert mass[(1, 2, 0, "low")] == pytest.approx(6.0)
    assert mass[(1, 2, 0, "high")] == pytest.approx(2.0)
    # o1-only edge (0,1,0): low = 4, high = 0
    assert mass[(0, 1, 0, "low")] == pytest.approx(4.0)
    assert mass[(0, 1, 0, "high")] == pytest.approx(0.0)

    diversity = ShannonEntropyMetric().compute(frame, _settings())
    dv = {(r.u, r.v, r.key): r.diversity for _, r in diversity.iterrows()}
    assert dv[(1, 2, 0)] == pytest.approx(6.490225, abs=1e-5)  # mixed 6/2
    assert dv[(0, 1, 0)] == pytest.approx(0.0)  # homogeneous


def test_accumulator_conservation_ledger():
    acc = EdgeAccumulator(_settings())
    acc.add(_traj("o1", [(0, 1, 0), (1, 2, 0)]), _vec(4, 0))  # mass 4, reachable
    acc.add(_traj("o2", [(1, 2, 0)]), _vec(2, 2))  # mass 4, reachable
    acc.add(_traj("o3", None), _vec(3, 1))  # mass 4, unreachable

    ledger = acc.ledger()
    assert ledger.emitted == pytest.approx(12.0)
    assert ledger.deposited == pytest.approx(8.0)
    assert ledger.unreachable == pytest.approx(4.0)
    check_weight_conservation(ledger)  # emitted == deposited + unreachable
    # edge_mass_total = mass × path length = 4·2 + 4·1 = 12
    assert acc.edge_mass_total == pytest.approx(12.0)
    assert (acc.n_reachable, acc.n_unreachable, acc.n_unresolved) == (2, 1, 0)


def test_accumulator_unresolved_is_flagged_never_zeroed():
    acc = EdgeAccumulator(_settings())
    acc.add(_traj("o1", [(0, 1, 0)]), _vec(4, 0))
    acc.add(_traj("bad", [(0, 1, 0)]), _vec(math.nan, math.nan))  # unresolved origin

    assert acc.n_unresolved == 1
    ledger = acc.ledger()
    assert ledger.emitted == pytest.approx(4.0)  # the NaN trajectory is NOT booked as 0
    # the unresolved trajectory deposited nothing on the shared edge
    mass = {(r.u, r.v, r.key, r["class"]): r.mass for _, r in acc.to_frame().iterrows()}
    assert mass[(0, 1, 0, "low")] == pytest.approx(4.0)


def test_accumulator_partitions_are_separate():
    acc = EdgeAccumulator(_settings())
    acc.add(_traj("o1", [(0, 1, 0)], binding="nearest"), _vec(4, 0))
    acc.add(_traj("o1", [(0, 1, 0)], binding="catchment"), _vec(4, 0))
    frame = acc.to_frame()
    # same edge, two partitions → the bindings stay separable
    assert set(frame["binding"]) == {"nearest", "catchment"}
    assert len(acc.partitions()) == 2


def test_to_frame_empty_when_nothing_added():
    frame = EdgeAccumulator(_settings()).to_frame()
    assert frame.empty
    assert list(frame.columns) == [
        "mode",
        "category",
        "binding",
        "u",
        "v",
        "key",
        "layer",
        "corridor",
        "length_m",
        "attribute",
        "class",
        "mass",
        "n_paths",
    ]
    # the metric tolerates an empty table
    assert ShannonEntropyMetric().compute(frame, _settings()).empty


# --------------------------------------------------------------------------- #
# Undirected segments + per-segment path counts               #
# --------------------------------------------------------------------------- #


def test_canonical_edge_orders_endpoints():
    assert canonical_edge((5, 2, 0)) == (2, 5, 0)
    assert canonical_edge((2, 5, 0)) == (2, 5, 0)
    assert canonical_edge((2, 5, 1)) == (2, 5, 1)  # key preserved


def test_reverse_traversals_fold_into_one_segment():
    acc = EdgeAccumulator(_settings())
    acc.add(_traj("o1", [(1, 2, 0)]), _vec(4, 0))  # traversed 1->2
    acc.add(_traj("o2", [(2, 1, 0)]), _vec(2, 2))  # same street, 2->1
    frame = acc.to_frame()
    # one undirected segment (1,2,0), masses summed across both directions
    assert set(zip(frame.u, frame.v, frame.key, strict=True)) == {(1, 2, 0)}
    mass = {r["class"]: r.mass for _, r in frame.iterrows()}
    assert mass == pytest.approx({"low": 6.0, "high": 2.0})
    # both trajectories counted as traffic on the single segment
    assert set(frame["n_paths"]) == {2}


def test_n_paths_counts_all_traffic_including_unresolved():
    acc = EdgeAccumulator(_settings())
    acc.add(_traj("o1", [(0, 1, 0), (1, 2, 0)]), _vec(4, 0))  # resolved, both edges
    acc.add(_traj("bad", [(0, 1, 0)]), _vec(math.nan, math.nan))  # unresolved but still traffic
    counts = {(r.u, r.v, r.key): r.n_paths for _, r in acc.path_counts().iterrows()}
    assert counts[(0, 1, 0)] == 2  # both trajectories traversed it
    assert counts[(1, 2, 0)] == 1
    # mass on (0,1,0) is from the resolved trajectory only (unresolved never deposits)
    mass = {(r.u, r.v, r.key, r["class"]): r.mass for _, r in acc.to_frame().iterrows()}
    assert mass[(0, 1, 0, "low")] == pytest.approx(4.0)
