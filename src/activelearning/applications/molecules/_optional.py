"""Helpers for optional molecules dependencies."""

from __future__ import annotations


def missing_molecules_dependency_error(
    component: str, error: ImportError
) -> ImportError:
    """Return a consistent ImportError for missing molecules extras."""
    _ = error
    return ImportError(
        f"{component} requires optional molecules dependencies that are not installed.\n"
        "Install them with:  uv sync --extra molecules\n"
        "or:                 pip install activelearning[molecules]"
    )
