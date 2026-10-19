"""Generate the TypeScript mirror of the JSON-RPC protocol from the canonical
Python definitions in :mod:`segx.rpc_protocol`.

Single-source-of-truth codegen: edit the ``TypedDict``s in ``segx/rpc_protocol.py``,
then run this to regenerate ``contracts/segx-rpc.ts``. The Electron app consumes
that ``.ts`` (via the engine submodule) so the two sides can't drift.

    uv run python scripts/gen_rpc_types.py           # write contracts/segx-rpc.ts
    uv run python scripts/gen_rpc_types.py --check    # CI: fail if out of date

Handles the type constructs used by the protocol: str/int/float/bool/None/Any,
``X | Y`` unions, ``Optional``, ``Literal[...]``, ``list[X]``, ``dict[str, X]``,
``NotRequired[X]``, and references to other protocol ``TypedDict``s.
"""

from __future__ import annotations

import json
import sys
import types
import typing
from pathlib import Path

# Make `segx` importable when run as `python scripts/gen_rpc_types.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segx import rpc_protocol as P  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "contracts" / "segx-rpc.ts"

#: protocol TypedDicts, by identity, so nested references emit as interface names
_TYPED_DICTS = {t: t.__name__ for t in P.EXPORTED_TYPES}


def _ts_literal(value: object) -> str:
    if isinstance(value, bool):  # bool before int (bool is an int subclass)
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return "null"
    return str(value)


def _ts_type(tp: object) -> str:
    """Map a Python typing construct to a TypeScript type expression."""
    if tp in _TYPED_DICTS:
        return _TYPED_DICTS[tp]  # type: ignore[index]
    if tp is typing.Any:
        return "unknown"
    if tp is type(None):
        return "null"
    if tp is str:
        return "string"
    if tp in (int, float):
        return "number"
    if tp is bool:
        return "boolean"

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Literal:
        return " | ".join(_ts_literal(a) for a in args)
    if origin is typing.Union or isinstance(tp, types.UnionType):
        return " | ".join(_ts_type(a) for a in args)
    if origin in (list, typing.List):  # noqa: UP006
        inner = _ts_type(args[0]) if args else "unknown"
        # parenthesise unions so `A | B[]` reads as `(A | B)[]`
        return f"({inner})[]" if " | " in inner else f"{inner}[]"
    if origin in (dict, typing.Dict):  # noqa: UP006
        val = _ts_type(args[1]) if len(args) == 2 else "unknown"
        return f"Record<string, {val}>"

    return "unknown"


def _emit_interface(cls: type) -> str:
    hints = typing.get_type_hints(cls, include_extras=True)
    optional_keys = getattr(cls, "__optional_keys__", frozenset())
    lines = [f"export interface {cls.__name__} {{"]
    doc = (cls.__doc__ or "").strip().splitlines()
    if doc:
        lines.insert(0, f"/** {doc[0]} */")
    for name, ann in hints.items():
        if typing.get_origin(ann) in (typing.NotRequired, typing.Required):
            ann = typing.get_args(ann)[0]
        opt = "?" if name in optional_keys else ""
        lines.append(f"  {name}{opt}: {_ts_type(ann)};")
    lines.append("}")
    return "\n".join(lines)


def render() -> str:
    header = (
        "// AUTO-GENERATED — do not edit by hand.\n"
        "// Source of truth: segx/rpc_protocol.py\n"
        "// Regenerate: uv run python scripts/gen_rpc_types.py\n"
        "//\n"
        "// JSON-RPC-over-stdio protocol for the SegX engine sidecar (segx/rpc.py).\n"
    )
    aliases = [
        f"export type CorrelationId = {_ts_type(P.CorrelationId)};",
        f"export type Method = {_ts_type(P.Method)};",
    ]
    blocks = [header, "\n".join(aliases)]
    blocks += [_emit_interface(cls) for cls in P.EXPORTED_TYPES]
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    content = render()
    check = "--check" in sys.argv[1:]
    if check:
        if not OUT_PATH.exists() or OUT_PATH.read_text(encoding="utf-8") != content:
            print(f"OUT OF DATE: {OUT_PATH} — run `uv run python scripts/gen_rpc_types.py`", file=sys.stderr)
            return 1
        print(f"up to date: {OUT_PATH}")
        return 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
