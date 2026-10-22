"""Retrying live OpenStreetMap downloads (Overpass) with exponential backoff.

The public Overpass servers shed load with transient errors — ``502 … server is
currently at capacity``, ``429 Too Many Requests``, ``504``, timeouts — and ask
clients to retry with exponential backoff. OSMnx only retries 429/504 itself;
anything else surfaced as an untyped exception and failed the run as
``internal``. :func:`with_overpass_retries` wraps one download: transient
failures are retried (``network.fetch_retries`` times, waiting
``network.fetch_backoff_s`` then doubling), each wait is logged (the desktop app
shows it in the live log) and checks for cancellation every second, and when the
retries run out the error is a typed :class:`~segx.errors.OSMFetchError`.
Nothing is cached on failure — callers only cache what a fetch returned.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from segx.errors import OSMFetchError
from segx.progress import CancelCheck, check_cancel

if TYPE_CHECKING:  # pragma: no cover
    from segx.config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T")

#: HTTP statuses worth retrying: rate limited, or the server busy / unreachable.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})

#: OSMnx's ResponseStatusCodeError reads "'overpass-api.de' responded: 502 Bad Gateway …".
_STATUS_RE = re.compile(r"responded:?\s*(\d{3})")


def transient_status(exc: BaseException) -> int | None:
    """The HTTP status if *exc* is a transient Overpass failure (0 = no status:
    timeout / connection error), else ``None`` (not worth retrying)."""
    import requests

    if isinstance(exc, requests.Timeout | requests.ConnectionError):
        return 0
    match = _STATUS_RE.search(str(exc))
    if match and int(match.group(1)) in TRANSIENT_STATUS:
        return int(match.group(1))
    return None


def _wait(seconds: float, cancel: CancelCheck | None, *, sleep: Callable[[float], None]) -> None:
    """Sleep *seconds*, checking for cancellation every second."""
    remaining = seconds
    while remaining > 0:
        check_cancel(cancel, stage="osm_fetch_retry")
        step = min(1.0, remaining)
        sleep(step)
        remaining -= step
    check_cancel(cancel, stage="osm_fetch_retry")


def with_overpass_retries(
    fetch: Callable[[], T],
    *,
    what: str,
    settings: Settings,
    cancel: CancelCheck | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *fetch*, retrying transient Overpass failures with exponential backoff.

    *what* names the download in logs and errors (``"walk network"``,
    ``"POIs 'parks'"``). Non-transient errors propagate unchanged.
    """
    retries = settings.network.fetch_retries
    backoff = settings.network.fetch_backoff_s
    for attempt in range(retries + 1):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 — classified just below
            status = transient_status(exc)
            if status is None:
                raise
            reason = f"HTTP {status}" if status else type(exc).__name__
            if attempt == retries:
                raise OSMFetchError(
                    f"The OpenStreetMap server (Overpass) is unavailable — {reason} while fetching "
                    f"{what}, {attempt + 1} attempt(s). It is usually overloaded; try again in a "
                    "few minutes. Nothing partial was cached.",
                    details={
                        "what": what,
                        "status": status or None,
                        "attempts": attempt + 1,
                        "retryable": True,
                        "error": str(exc)[:500],
                    },
                ) from exc
            delay = backoff * 2**attempt
            log.warning(
                "OpenStreetMap server (Overpass) unavailable (%s) while fetching %s — retry %d of %d in %.0f s",
                reason,
                what,
                attempt + 1,
                retries,
                delay,
            )
            _wait(delay, cancel, sleep=sleep)
    raise AssertionError("unreachable")  # pragma: no cover — the loop returns or raises
