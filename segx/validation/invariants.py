"""Runtime invariants asserted on every real run. These are cheap, catch silent-correctness bugs, and make the CLI exit
non-zero on violation rather than emitting plausible-but-wrong numbers.

Only the pure, dependency-free checks are implemented at P0; the ones that need
the aggregated edge table (``entropy == 0`` on homogeneous edges, maximal entropy
on uniform edges) are added in P4 alongside the accumulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class InvariantViolation(AssertionError):
    """Raised when a conservation/sanity invariant fails."""


@dataclass(frozen=True)
class WeightLedger:
    """Bookkeeping for the conservation check.

    ``emitted`` is the total trajectory weight × origin mass that entered routing;
    ``deposited`` is the total that landed on edges; ``unreachable`` is the total
    that was dropped because no path existed (this is expected and logged, not an
    error). The invariant is ``emitted == deposited + unreachable`` (within tol).
    """

    emitted: float
    deposited: float
    unreachable: float


def check_weight_conservation(ledger: WeightLedger, *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> None:
    """Assert emitted weight is fully accounted for. Raises :class:`InvariantViolation`."""
    expected = ledger.deposited + ledger.unreachable
    if not math.isclose(ledger.emitted, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise InvariantViolation(
            "weight not conserved: emitted="
            f"{ledger.emitted!r} but deposited+unreachable={expected!r} "
            f"(deposited={ledger.deposited!r}, unreachable={ledger.unreachable!r})"
        )


def check_non_negative_weights(values: list[float], *, where: str = "weights") -> None:
    """Assert all weights/masses are non-negative."""
    bad = [v for v in values if v < 0 or math.isnan(v)]
    if bad:
        raise InvariantViolation(f"{where}: found negative or NaN values: {bad[:5]}{'...' if len(bad) > 5 else ''}")


__all__ = [
    "InvariantViolation",
    "WeightLedger",
    "check_non_negative_weights",
    "check_weight_conservation",
]
