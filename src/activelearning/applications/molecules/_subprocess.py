"""Process-group-aware subprocess helper shared by the molecular oracles.

Both molecular toolchain oracles launch their real work through a shell
(``bash -c 'source dockenv.sh && ligbuild ...'`` for DOCK3, ``bash -c 'module
load java && cxcalc ...'`` for cxcalc), which means the process that does the
work is a *grandchild* of this interpreter. ``subprocess`` only ever signals
the process it launched directly, so a plain ``subprocess.run(timeout=...)``
leaves the real workers running with no parent watching, burning cores in the
job's allocation for the rest of the run.

:func:`_run_subprocess` is the fix, and it is the only subprocess entry point
those oracles should use.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

# Seconds allowed for an already-killed process tree to close its pipes. The
# tree has been SIGKILLed by this point, so this only bounds a pathological
# drain rather than any real work.
_KILL_DRAIN_TIMEOUT = 10


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """SIGKILL a process along with every descendant it spawned.

    ``subprocess`` only ever signals the process it launched directly. Here
    that process is a shell, and the real work runs in its children (ligbuild,
    with AMSOL and OpenEye below that; a JVM for cxcalc), so killing the shell
    alone leaves them running with no parent watching. They then keep consuming
    cores in the job's allocation for the remainder of the run.

    Relies on the process having been started with ``start_new_session=True``,
    which makes its PID the ID of a process group containing the whole tree.

    Parameters
    ----------
    process : subprocess.Popen
        Process whose tree should be killed.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # Already reaped, or the group is not ours to signal. Killing the
        # direct child is the best that can be done.
        process.kill()


def _run_subprocess(
    command: list[str],
    *,
    timeout: int,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in its own process group, killing the tree on timeout.

    Equivalent to ``subprocess.run(capture_output=True, text=True,
    timeout=...)`` except that a timeout takes down the entire process tree
    instead of just the launched process; see :func:`_kill_process_tree`.

    Parameters
    ----------
    command : list[str]
        Command and arguments to execute.
    timeout : int
        Wall-clock seconds allowed before the process tree is killed.
    cwd : Path, optional
        Working directory for the subprocess.
    env : dict[str, str], optional
        Complete environment for the subprocess. ``None`` inherits this
        process's environment; pass a full mapping (not a delta) to override it.

    Returns
    -------
    subprocess.CompletedProcess
        Completed process with captured text ``stdout`` and ``stderr``.

    Raises
    ------
    subprocess.TimeoutExpired
        If the command did not finish within ``timeout``. Its message still
        contains "timed out", which is what the callers' failure classifiers
        key on.
    """
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            try:
                process.communicate(timeout=_KILL_DRAIN_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
