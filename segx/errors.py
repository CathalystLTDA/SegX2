"""Typed error hierarchy for the SegX engine (P0; retrofitted onto P1 IO/network).

Every fault the engine raises deliberately is a :class:`SegXError` carrying a
stable, machine-readable ``code`` and an optional ``details`` mapping. This is
what lets the future native desktop app — which **embeds** this engine — render failures in its UI and branch on
``code`` instead of screen-scraping message strings, which is fragile and not
localizable. It also gives callers (CLI, app, tests) one base class to catch.

**Backward compatibility:** each leaf type also subclasses the builtin it
semantically replaces (``ValueError`` / ``FileNotFoundError`` / ``KeyError``),
so existing ``except FileNotFoundError`` / ``pytest.raises(ValueError, ...)``
call sites keep working unchanged. New code should catch :class:`SegXError`
(or a subclass) and read ``.code``.

**Not for "no data".** Per invariant #2, *"unreachable"* is a typed **result**,
not an exception (``nearest_poi`` returns ``None`` + reason code; ``poi_catchment``
may return an empty set). Exceptions here are for genuine faults — a missing
file, an unusable schema, an unbuildable graph — never for a legitimate
"all-mass-in-one-class" or "nothing reachable" signal.
"""

from __future__ import annotations

from typing import Any


class OperationCancelled(Exception):
    """Raised to unwind the pipeline when the caller requests cancellation.

    Deliberately **not** a :class:`SegXError`: a user pressing *Cancel* in the
    app is not a fault. The app catches this separately and reports "cancelled",
    distinct from an error dialog. Carries the stage at which the cancel was
    observed so the UI can say where it stopped.
    """

    code: str = "cancelled"

    def __init__(self, message: str = "operation cancelled", *, stage: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage


class SegXError(Exception):
    """Base class for every error the SegX engine raises deliberately.

    ``code`` is a stable dotted identifier the embedding app can switch on and
    localize; treat it as part of the public contract — don't change a code once
    shipped. ``details`` carries structured context (offending path, column
    names, available options, …) for the UI to render without parsing the
    message.
    """

    code: str = "segx.error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:  # normalize: KeyError-based subclasses otherwise repr-quote
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Serializable form for IPC to the frontend and for structured logs."""
        return {"code": self.code, "message": self.message, "details": self.details}


# --------------------------------------------------------------------------- #
# Input / IO                                                                   #
# --------------------------------------------------------------------------- #


class InputError(SegXError, ValueError):
    """Bad user-supplied input data (wrong schema, empty, unusable geometry)."""

    code = "input.invalid"


class InputFileNotFoundError(SegXError, FileNotFoundError):
    """A required input file does not exist."""

    code = "input.file_not_found"


class EmptyInputError(InputError):
    """An input file loaded but contains no usable features / rows."""

    code = "input.empty"


class SchemaError(InputError):
    """Input is missing required columns or has an unusable schema/mapping."""

    code = "input.schema"


class GeometryError(InputError):
    """Input geometry is null, mixed, or otherwise not a usable endpoint."""

    code = "input.geometry"


# --------------------------------------------------------------------------- #
# CRS                                                                          #
# --------------------------------------------------------------------------- #


class CRSError(SegXError, ValueError):
    """Working CRS cannot be derived, is out of range, or mismatches (risk #1)."""

    code = "crs.invalid"


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


class ConfigError(SegXError):
    """Base for resolved-config problems."""

    code = "config.invalid"


class ConfigKeyError(ConfigError, KeyError):
    """An override path or lookup names a key the config doesn't have."""

    code = "config.unknown_key"


class ConfigValueError(ConfigError, ValueError):
    """A config value is present but invalid for how it's used."""

    code = "config.invalid_value"


# --------------------------------------------------------------------------- #
# Network                                                                      #
# --------------------------------------------------------------------------- #


class NetworkError(SegXError):
    """Base for street-network construction / lookup problems."""

    code = "network.error"


class GraphError(NetworkError, ValueError):
    """A graph is missing required metadata (CRS, edge length, nodes)."""

    code = "network.graph"


class ModeNotAvailableError(NetworkError, ValueError):
    """A travel mode is unknown or reserved for a later module ('transit')."""

    code = "network.mode_unavailable"


class ModeNotRegisteredError(NetworkError, KeyError):
    """A valid mode was requested but its graph hasn't been built/registered yet."""

    code = "network.mode_not_registered"


class CacheError(NetworkError):
    """Base for OSM cache problems (reproducibility — risk #3/#4)."""

    code = "network.cache"


class CacheMissError(CacheError, FileNotFoundError):
    """A cache entry was expected but its artifacts/manifest are absent."""

    code = "network.cache_miss"


class OSMFetchError(NetworkError, RuntimeError):
    """A live OpenStreetMap download (Overpass) kept failing with a transient error.

    Raised after the configured retries (``network.fetch_retries``, exponential
    backoff from ``network.fetch_backoff_s``) for responses such as ``502 … at
    capacity``, ``429`` or a timeout. ``details.retryable`` is ``True``: the same
    run is expected to succeed later, and nothing partial was cached.
    """

    code = "network.osm_fetch"


class POIFetchError(NetworkError, RuntimeError):
    """A live POI fetch failed (timeout, rate limit, server error, bad response).

    Distinct from "the area has no POI of this category", which is a real,
    cacheable zero. A failed fetch is never turned into an empty result: it would
    be cached and read back as "no POIs here" on every later run (invariants #2
    and #4).
    """

    code = "network.poi_fetch"


class OSMExtractError(NetworkError, ValueError):
    """A local OpenStreetMap extract (``network.osm_pbf_url``) cannot serve the run.

    Raised when the ``.osm.pbf`` file does not cover the study area — its header
    bounding box misses part of the boundary, or it holds no street at all inside
    it. Never retryable: the same file gives the same answer; the user needs an
    extract of the right region.
    """

    code = "network.osm_extract"


# --------------------------------------------------------------------------- #
# Public transport (P7)                                                        #
# --------------------------------------------------------------------------- #


class TransitError(SegXError):
    """Base for public-transport (GTFS / transit-graph) problems."""

    code = "transit.error"


class GTFSError(TransitError, ValueError):
    """A GTFS feed could not be read (missing zip member, unreadable CSV, ...)."""

    code = "transit.gtfs"


class GTFSSchemaError(GTFSError):
    """A GTFS feed is readable but missing a required file or column."""

    code = "transit.gtfs_schema"


class TransitDataMissingError(TransitError, ValueError):
    """Transit was requested but no usable stops/routes exist for the study area.

    Distinct from :class:`GTFSSchemaError`: the feed (or OSM extract) parsed fine,
    it simply contains nothing inside the boundary — which is a *fault* only
    because the caller asked for a transit graph. Per invariant #2, an empty
    catchment is a result; an empty transit **network** is not.
    """

    code = "transit.data_missing"


# --------------------------------------------------------------------------- #
# Output (P8)                                                                  #
# --------------------------------------------------------------------------- #


class OutputError(SegXError, OSError):
    """A deliverable (``segments.gpkg`` / ``metrics.xlsx``) could not be written."""

    code = "output.write"


class OutputColumnCollisionError(SegXError, ValueError):
    """Two output columns resolved to the same name.

    Column names are derived from configured attribute/class names and metric
    names (invariant #3), so a collision is a configuration fact — e.g. an
    attribute named after a metric — not a writer bug. Raised rather than letting
    one column silently overwrite the other.
    """

    code = "output.column_collision"


__all__ = [
    "CRSError",
    "CacheError",
    "CacheMissError",
    "ConfigError",
    "ConfigKeyError",
    "ConfigValueError",
    "EmptyInputError",
    "GTFSError",
    "GTFSSchemaError",
    "GeometryError",
    "GraphError",
    "InputError",
    "InputFileNotFoundError",
    "ModeNotAvailableError",
    "ModeNotRegisteredError",
    "NetworkError",
    "OSMExtractError",
    "OSMFetchError",
    "OperationCancelled",
    "OutputColumnCollisionError",
    "OutputError",
    "SchemaError",
    "SegXError",
    "TransitDataMissingError",
    "TransitError",
]
