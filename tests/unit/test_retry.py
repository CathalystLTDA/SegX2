"""Retrying live OSM downloads: which failures are transient, the backoff schedule,
the typed error when retries run out, and cancellation during the wait."""

from __future__ import annotations

import logging

import pytest
import requests

from segx.config import NetworkConfig, Settings
from segx.errors import OperationCancelled, OSMFetchError
from segx.network.retry import transient_status, with_overpass_retries

AT_CAPACITY = (
    "'overpass-api.de' responded: 502 Bad Gateway # 502 Gateway Timeout This Overpass API "
    "server is currently at capacity."
)


def _settings(retries: int = 3, backoff: float = 30.0) -> Settings:
    return Settings(network=NetworkConfig(fetch_retries=retries, fetch_backoff_s=backoff))


class _Flaky:
    """A fetch that fails ``failures`` times with ``exc`` and then returns ``"graph"``."""

    def __init__(self, failures: int, exc: Exception) -> None:
        self.failures, self.exc, self.calls = failures, exc, 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "graph"


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (RuntimeError(AT_CAPACITY), 502),
        (RuntimeError("'overpass-api.de' responded: 429 Too Many Requests"), 429),
        (RuntimeError("'overpass-api.de' responded: 504 Gateway Timeout"), 504),
        (requests.Timeout("read timed out"), 0),
        (requests.ConnectionError("connection reset"), 0),
        (RuntimeError("'overpass-api.de' responded: 400 Bad Request"), None),  # our query is wrong
        (ValueError("no features"), None),
    ],
)
def test_transient_status(exc, status):
    assert transient_status(exc) == status


def test_retries_with_exponential_backoff_then_succeeds(caplog):
    slept: list[float] = []
    fetch = _Flaky(2, RuntimeError(AT_CAPACITY))

    with caplog.at_level(logging.WARNING, logger="segx.network.retry"):
        result = with_overpass_retries(fetch, what="the walk street network", settings=_settings(), sleep=slept.append)

    assert result == "graph"
    assert fetch.calls == 3
    assert sum(slept) == pytest.approx(30 + 60)  # backoff, then doubled
    assert all(step <= 1.0 for step in slept)  # waits in 1 s steps (cancel is checked between them)
    messages = [r.getMessage() for r in caplog.records]
    assert "retry 1 of 3 in 30 s" in messages[0] and "HTTP 502" in messages[0]
    assert "retry 2 of 3 in 60 s" in messages[1]


def test_gives_up_with_a_typed_retryable_error():
    fetch = _Flaky(99, RuntimeError(AT_CAPACITY))

    with pytest.raises(OSMFetchError) as exc:
        with_overpass_retries(
            fetch, what="the drive street network", settings=_settings(retries=2, backoff=1), sleep=lambda _s: None
        )

    assert exc.value.code == "network.osm_fetch"
    assert exc.value.details["attempts"] == 3
    assert exc.value.details["status"] == 502
    assert exc.value.details["retryable"] is True
    assert "drive street network" in exc.value.message
    assert fetch.calls == 3


def test_a_non_transient_error_is_not_retried():
    fetch = _Flaky(1, ValueError("bad polygon"))
    slept: list[float] = []

    with pytest.raises(ValueError, match="bad polygon"):
        with_overpass_retries(fetch, what="POIs", settings=_settings(), sleep=slept.append)

    assert fetch.calls == 1
    assert slept == []


def test_zero_retries_fails_on_the_first_transient_error():
    with pytest.raises(OSMFetchError):
        with_overpass_retries(
            _Flaky(1, requests.Timeout("t")), what="x", settings=_settings(retries=0), sleep=lambda _s: None
        )


def test_cancel_during_the_wait_stops_the_retries():
    slept: list[float] = []
    fetch = _Flaky(5, RuntimeError(AT_CAPACITY))

    with pytest.raises(OperationCancelled):
        with_overpass_retries(
            fetch,
            what="the walk street network",
            settings=_settings(backoff=30),
            cancel=lambda: len(slept) >= 3,  # the user cancels 3 s into the first wait
            sleep=slept.append,
        )

    assert fetch.calls == 1
    assert sum(slept) == pytest.approx(3)
