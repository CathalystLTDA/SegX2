"""Typed error hierarchy (P0; retrofitted onto P1 IO/network raise sites).

Verifies the two properties the embedding app relies on: every deliberate fault is a ``SegXError`` with a stable
``code`` and a serializable ``to_dict``; and each leaf still subclasses the
builtin it replaced, so old ``except``/`pytest.raises` keep working.
"""

from __future__ import annotations

import pytest

from segx.config import DemographicSchema, Settings
from segx.errors import (
    CacheMissError,
    ConfigKeyError,
    CRSError,
    GeometryError,
    InputFileNotFoundError,
    ModeNotAvailableError,
    ModeNotRegisteredError,
    SchemaError,
    SegXError,
)
from segx.io._crs import utm_epsg_for
from segx.io.boundary import load_boundary
from segx.network.registry import GraphRegistry


def test_segxerror_carries_code_and_serializes() -> None:
    err = SchemaError("missing columns", details={"missing": ["pop"]})
    assert err.code == "input.schema"
    assert err.to_dict() == {
        "code": "input.schema",
        "message": "missing columns",
        "details": {"missing": ["pop"]},
    }
    # str() is the clean message even for KeyError-based subclasses (no repr quoting)
    assert str(ConfigKeyError("no such key")) == "no such key"


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (InputFileNotFoundError, FileNotFoundError),
        (SchemaError, ValueError),
        (GeometryError, ValueError),
        (CRSError, ValueError),
        (ConfigKeyError, KeyError),
        (ModeNotAvailableError, ValueError),
        (ModeNotRegisteredError, KeyError),
        (CacheMissError, FileNotFoundError),
    ],
)
def test_leaf_errors_keep_builtin_compatibility(exc, builtin) -> None:
    # every leaf is both a SegXError (one base for the app to catch) and the
    # builtin it replaced (so existing call sites/tests keep catching it).
    assert issubclass(exc, SegXError)
    assert issubclass(exc, builtin)
    instance = exc("boom")
    assert isinstance(instance, SegXError)
    assert isinstance(instance, builtin)


def test_codes_are_unique_across_the_hierarchy() -> None:
    leaves = [
        CacheMissError,
        ConfigKeyError,
        CRSError,
        GeometryError,
        InputFileNotFoundError,
        ModeNotAvailableError,
        ModeNotRegisteredError,
        SchemaError,
    ]
    codes = [e.code for e in leaves]
    assert len(codes) == len(set(codes)), f"duplicate error codes: {codes}"


def test_crs_helper_raises_typed_error_with_code() -> None:
    with pytest.raises(CRSError) as ei:
        utm_epsg_for(999.0, 0.0)
    assert ei.value.code == "crs.invalid"
    assert ei.value.details["lon"] == 999.0


def test_boundary_missing_file_is_typed_but_still_filenotfound(tmp_path) -> None:
    settings = Settings(demographics=DemographicSchema.ibge_default())
    with pytest.raises(FileNotFoundError) as ei:  # builtin still matches
        load_boundary(tmp_path / "nope.gpkg", settings)
    assert isinstance(ei.value, InputFileNotFoundError)
    assert ei.value.code == "input.file_not_found"


def test_registry_distinguishes_unknown_mode_from_unregistered() -> None:
    reg = GraphRegistry()
    # unknown mode name → value error semantics + its own code
    with pytest.raises(ModeNotAvailableError) as unknown:
        reg.get("teleport")
    assert unknown.value.code == "network.mode_unavailable"
    # valid-but-unbuilt mode → key-lookup semantics + a different code.
    # 'transit' is a real mode since P7, so it lands here, not above.
    for mode in ("drive", "transit"):
        with pytest.raises(ModeNotRegisteredError) as missing:
            reg.get(mode)
        assert missing.value.code == "network.mode_not_registered"
