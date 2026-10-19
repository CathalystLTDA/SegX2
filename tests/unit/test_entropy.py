"""P0: Shannon entropy — pinned against hand-computed values."""

from __future__ import annotations

import math

import pytest

from segx.metrics.entropy import gini_simpson, normalized_entropy, shannon_entropy
from segx.validation.synthetic_city import EXPECTED_ENTROPIES_BITS

_NAMED_DISTRIBUTIONS: dict[str, list[float]] = {
    "single_class": [4.0, 0.0],
    "uniform_2": [2.0, 2.0],
    "skew_3_1": [3.0, 1.0],
    "skew_2_1": [2.0, 1.0],
    "uniform_3": [1.0, 1.0, 1.0],
    "uniform_4": [1.0, 1.0, 1.0, 1.0],
    "empty": [],
}


@pytest.mark.parametrize("name", sorted(_NAMED_DISTRIBUTIONS))
def test_shannon_entropy_matches_hand_values(name):
    counts = _NAMED_DISTRIBUTIONS[name]
    expected = EXPECTED_ENTROPIES_BITS[name]
    got = shannon_entropy(counts)
    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected, abs=1e-12)


def test_entropy_is_zero_for_single_class_not_nan():
    assert shannon_entropy([5.0, 0.0, 0.0]) == 0.0


def test_entropy_undefined_for_zero_mass():
    assert math.isnan(shannon_entropy([0.0, 0.0]))
    assert math.isnan(shannon_entropy([]))


def test_entropy_rejects_negative():
    with pytest.raises(ValueError):
        shannon_entropy([1.0, -0.5])


def test_entropy_is_scale_invariant():
    assert shannon_entropy([1.0, 1.0]) == pytest.approx(shannon_entropy([7.0, 7.0]))


def test_entropy_base_e():
    # nats: H({1,1}) = ln 2
    assert shannon_entropy([1.0, 1.0], base=math.e) == pytest.approx(math.log(2))


def test_normalized_entropy():
    assert normalized_entropy([2.0, 2.0], n_classes=2) == pytest.approx(1.0)
    assert normalized_entropy([4.0, 0.0], n_classes=2) == pytest.approx(0.0)
    # observed mass in 2 of 4 possible classes, evenly → H=1 bit, max=log2(4)=2 → 0.5
    assert normalized_entropy([3.0, 3.0], n_classes=4) == pytest.approx(0.5)
    assert normalized_entropy([1.0], n_classes=1) == 0.0
    assert math.isnan(normalized_entropy([0.0, 0.0], n_classes=3))


# --------------------------------------------------------------------------- #
# gini_simpson — probability of intergroup contact (1 − Σ pᵢ²)                 #
# --------------------------------------------------------------------------- #


def test_gini_simpson_hand_values():
    assert gini_simpson([4.0, 0.0]) == pytest.approx(0.0)  # one class → no contact possible
    assert gini_simpson([2.0, 2.0]) == pytest.approx(0.5)  # 1 − (0.25 + 0.25)
    # {3,1}: 1 − (0.75² + 0.25²) = 1 − 0.625 = 0.375
    assert gini_simpson([3.0, 1.0]) == pytest.approx(0.375)
    # uniform over 3 classes: 1 − 3·(1/3)² = 2/3
    assert gini_simpson([1.0, 1.0, 1.0]) == pytest.approx(2.0 / 3.0)


def test_gini_simpson_is_scale_invariant():
    assert gini_simpson([2.0, 2.0]) == pytest.approx(gini_simpson([9.0, 9.0]))


def test_gini_simpson_zero_mass_is_nan_not_zero():
    assert math.isnan(gini_simpson([0.0, 0.0]))
    assert math.isnan(gini_simpson([]))


def test_gini_simpson_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        gini_simpson([1.0, -0.5])


def test_experienced_segregation_is_one_minus_normalized_entropy():
    # the metric ships 1 − h; pin the underlying arithmetic on known distributions
    assert 1.0 - normalized_entropy([2.0, 2.0], n_classes=2) == pytest.approx(0.0)  # perfectly mixed
    assert 1.0 - normalized_entropy([4.0, 0.0], n_classes=2) == pytest.approx(1.0)  # homogeneous
