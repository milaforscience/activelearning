"""Tests for monitoring package boundaries."""

import subprocess
import sys


def test_logger_can_be_imported_before_monitoring() -> None:
    """Logger-first imports must not cycle through monitoring orchestration."""
    command = (
        "from activelearning.logger.logger import Logger; "
        "from activelearning.monitoring.orchestration import collect_round_diagnostics; "
        "assert Logger and collect_round_diagnostics"
    )

    subprocess.run([sys.executable, "-c", command], check=True)
