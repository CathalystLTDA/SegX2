"""Lookup of routable graphs by mode: ``'walk'``, ``'drive'`` and — since P7 —
``'transit'``, the merged walk ⊕ public-transport graph built by
:mod:`segx.network.transit`.

Two failure modes, deliberately distinguished by type and error code so the
embedding app can tell them apart without parsing messages:

* :class:`~segx.errors.ModeNotAvailableError` — the *name* is not a travel mode
  this engine knows (a typo, or a mode from a future module);
* :class:`~segx.errors.ModeNotRegisteredError` — a real mode whose graph simply
  has not been built for this run (e.g. asking for ``transit`` when
  ``transit.enabled`` is off, or ``drive`` when ``network.modes`` omitted it).
"""

from __future__ import annotations

from typing import Any

from segx.errors import ModeNotAvailableError, ModeNotRegisteredError

#: Travel modes the engine can build a routable graph for. ``transit`` became
#: real in P7; it used to be a *reserved* name that raised on sight.
KNOWN_MODES: frozenset[str] = frozenset({"walk", "drive", "transit"})


def _check_known(mode: str) -> None:
    if mode not in KNOWN_MODES:
        raise ModeNotAvailableError(
            f"unknown travel mode {mode!r}; known modes: {sorted(KNOWN_MODES)}",
            details={"mode": mode, "known": sorted(KNOWN_MODES)},
        )


class GraphRegistry:
    """Holds one built graph per mode for the duration of a run."""

    def __init__(self) -> None:
        self._graphs: dict[str, Any] = {}

    def register(self, mode: str, graph: Any) -> None:
        _check_known(mode)
        self._graphs[mode] = graph

    def get(self, mode: str) -> Any:
        _check_known(mode)
        try:
            return self._graphs[mode]
        except KeyError:
            raise ModeNotRegisteredError(
                f"no graph registered for mode {mode!r}; available: {sorted(self._graphs)}",
                details={"mode": mode, "available": sorted(self._graphs)},
            ) from None

    def modes(self) -> list[str]:
        return sorted(self._graphs)

    def __contains__(self, mode: object) -> bool:
        return mode in self._graphs


__all__ = ["KNOWN_MODES", "GraphRegistry"]
