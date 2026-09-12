"""Helpers for optional-dependency boundaries."""

from __future__ import annotations


def missing_optional_dependency_error(
    component: str,
    error: ImportError,
    *,
    extra: str | None = None,
    distribution: str | None = None,
) -> ImportError:
    """Build an actionable error for a missing optional dependency.

    Parameters
    ----------
    component : str
        Component that could not be initialized.
    error : ImportError
        Original import error, used as the cause when the returned error is
        raised.
    extra : str, optional
        Project extra containing the optional dependencies.
    distribution : str, optional
        Distribution containing the optional dependencies.

    Returns
    -------
    ImportError
        Error with installation instructions for the requested extra.
    """
    if distribution is not None:
        message = (
            f"{component} requires optional dependencies from "
            f"{distribution!r} that are not installed.\n"
            f"Install them with:  uv sync --package {distribution}\n"
            f"or:                 pip install {distribution}"
        )
    elif extra is not None:
        message = (
            f"{component} requires optional {extra} dependencies that are not "
            "installed.\n"
            f"Install them with:  uv sync --extra {extra}\n"
            f"or:                 pip install activelearning[{extra}]"
        )
    else:
        message = f"{component} requires an optional dependency that is not installed."
    actionable_error = ImportError(message)
    actionable_error.__cause__ = error
    return actionable_error
