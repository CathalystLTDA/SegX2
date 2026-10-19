"""Progress reporting and cooperative cancellation for the pipeline (P0 seam).

The orchestrator and every long-running stage accept an optional ``progress``
callback and an optional ``cancel`` check. Both default to ``None`` (no-ops),
so terminal/library callers ignore them entirely — but the future native
desktop app (which embeds this engine)
passes a ``progress`` sink to drive a progress bar and a ``cancel`` check backed
by, e.g., ``threading.Event.is_set`` so its *Cancel* button actually stops a
multi-minute OSM fetch or routing pass.

The seam is introduced now, while the call sites are few, precisely so adding it
later doesn't churn every stage signature. Stages call :func:`report` at their
start; long inner loops may call :func:`check_cancel` between iterations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from segx.errors import OperationCancelled

#: A sink the caller provides to observe progress. Never raises; must be cheap.
ProgressCallback = Callable[["ProgressEvent"], None]

#: A predicate the caller provides; returns ``True`` once cancellation is wanted.
#: ``threading.Event().is_set`` satisfies this directly.
CancelCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One progress tick. ``index`` is 0-based; ``total`` is the run's stage count."""

    stage: str
    index: int
    total: int
    message: str = ""

    @property
    def fraction(self) -> float:
        """Completed fraction in ``[0, 1]`` after this stage starts."""
        return (self.index + 1) / self.total if self.total else 0.0


def check_cancel(cancel: CancelCheck | None, *, stage: str | None = None) -> None:
    """Raise :class:`OperationCancelled` if ``cancel`` reports a stop request."""
    if cancel is not None and cancel():
        raise OperationCancelled(stage=stage)


def report(
    progress: ProgressCallback | None,
    cancel: CancelCheck | None,
    *,
    stage: str,
    index: int,
    total: int,
    message: str = "",
) -> None:
    """Check cancellation, then emit a :class:`ProgressEvent` if a sink is set.

    Called at the start of each pipeline stage: cancellation is honoured before
    the stage's work begins, and the event announces the stage about to run.
    """
    check_cancel(cancel, stage=stage)
    if progress is not None:
        progress(ProgressEvent(stage=stage, index=index, total=total, message=message))


__all__ = ["CancelCheck", "ProgressCallback", "ProgressEvent", "check_cancel", "report"]
