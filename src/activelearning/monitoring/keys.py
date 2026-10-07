"""Validation for component-qualified monitoring keys."""

import re


_LOG_KEY_SEGMENT = re.compile(r"^[a-z0-9_.-]+$")


def validate_log_key(key: str) -> None:
    """Require a metric or figure key to identify its owning component."""
    segments = key.split("/") if isinstance(key, str) else []
    minimum_segments = 2 if segments and segments[0] == "active_learning" else 3
    if (
        len(segments) < minimum_segments
        or any(not segment for segment in segments)
        or any(not _LOG_KEY_SEGMENT.fullmatch(segment) for segment in segments)
    ):
        raise ValueError(
            "Monitoring keys must use a component-qualified namespace such as "
            "'sampler/component/metric'; got "
            f"{key!r}."
        )
