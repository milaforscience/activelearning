import warnings
from typing import Any


def warn_ignored_args(caller: object, **kwargs: Any) -> None:
    """Emit a :class:`UserWarning` for each non-``None`` keyword argument.

    Intended for ``sample()`` / ``select()`` implementations that accept
    interface arguments they do not use.

    Parameters
    ----------
    caller : object
        The object whose class name appears in the warning message.
    **kwargs : Any
        Argument names mapped to their values. A warning is emitted for
        every value that is not ``None``.

    Examples
    --------
    >>> warn_ignored_args(self, acquisition=acquisition, cost_fn=cost_fn)
    """
    for name, val in kwargs.items():
        if val is not None:
            warnings.warn(
                f"{type(caller).__name__} does not use {name}; it will be ignored.",
                UserWarning,
                stacklevel=3,
            )
