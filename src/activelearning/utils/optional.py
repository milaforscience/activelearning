"""Helpers for optional-dependency boundaries."""

from __future__ import annotations


def missing_optional_dependency_error(
    component: str,
    extra: str,
    error: ImportError,
) -> ImportError:
    """Build an actionable error for a missing optional dependency.

    Parameters
    ----------
    component : str
        Component that could not be initialized.
    extra : str
        Project extra containing the optional dependencies.
    error : ImportError
        Original import error, used as the cause when the returned error is
        raised.

    Returns
    -------
    ImportError
        Error with installation instructions for the requested extra.
    """
    return ImportError(
        f"{component} requires optional {extra} dependencies that are not "
        "installed.\n"
        f"Install them with:  uv sync --extra {extra}\n"
        f"or:                 pip install activelearning[{extra}]"
    )
