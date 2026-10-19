"""Thin entry point — delegates to the Typer CLI in :mod:`segx.cli`.

Prefer the installed console script (``uv run segx ...``); this module exists so
``python main.py ...`` also works.
"""

from __future__ import annotations

from segx.cli import app

if __name__ == "__main__":
    app()
