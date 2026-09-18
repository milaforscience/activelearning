"""Helpers for optional molecules dependencies."""

from __future__ import annotations

from activelearning.utils.optional import missing_optional_dependency_error


def missing_molecules_dependency_error(
    component: str, error: ImportError
) -> ImportError:
    """Return a consistent error for a missing molecules extra.

    Parameters
    ----------
    component : str
        Component that could not be initialized.
    error : ImportError
        Original import error, used as the cause when the returned error is
        raised.

    Returns
    -------
    ImportError
        Actionable error explaining how to install the molecules extra.
    """
    return missing_optional_dependency_error(
        component=component,
        extra="molecules",
        error=error,
    )
