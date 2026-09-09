"""CPU-allocation helpers shared by the molecular toolchain oracles.

Both :mod:`activelearning.applications.molecules.dock3_oracle` and
:mod:`activelearning.applications.molecules.cxcalc_oracle` fan a query batch out
over a thread pool whose real work happens in external binaries. Sizing that
pool correctly is the difference between a Slurm allocation's cores all doing
chemistry and most of them sitting idle, so the sizing logic lives here once
rather than being duplicated per oracle.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Environment overrides applied to every oracle subprocess when more than one
# worker is running. The molecular toolchains are driven one molecule per
# worker, so a threaded BLAS or OpenMP component inside them would multiply the
# worker count instead of adding throughput -- and would invalidate the
# core-second cost constants, which were measured at one core per molecule.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _affinity_cpu_count() -> Optional[int]:
    """Return the number of CPUs this process is actually allowed to run on.

    ``os.cpu_count`` reports the CPUs on the *node*, which under Slurm is the
    wrong number whenever the job holds less than the whole machine: a
    ``--cpus-per-task=8`` allocation on a 64-core node would size a pool of 64
    and oversubscribe eightfold. ``os.sched_getaffinity`` reads the cpuset the
    job was actually granted.

    Returns
    -------
    int or None
        The size of this process's CPU affinity mask, or ``None`` on platforms
        without ``sched_getaffinity`` (it is Linux-only).
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return len(os.sched_getaffinity(0))
    except OSError:
        return None


def _slurm_cpus_per_task() -> Optional[int]:
    """Return ``$SLURM_CPUS_PER_TASK`` as a positive int, if it is usable.

    Returns
    -------
    int or None
        The parsed value, or ``None`` when the variable is unset, unparsable or
        not positive.
    """
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring unparsable SLURM_CPUS_PER_TASK=%r", raw)
        return None
    if value < 1:
        logger.warning("Ignoring non-positive SLURM_CPUS_PER_TASK=%r", raw)
        return None
    return value


def resolve_num_workers(num_workers: Optional[int]) -> int:
    """Resolve a worker count, defaulting to the CPUs this job actually holds.

    An explicit count is taken as given. ``None`` resolves to the job's CPU
    allocation, read from the two signals that can disagree:

    - the process CPU affinity mask, which is authoritative wherever Slurm's
      cgroup/cpuset plugins are enabled, but reports the whole node where they
      are not;
    - ``$SLURM_CPUS_PER_TASK``, which is unset under several common submission
      styles (``--ntasks-per-node`` without ``--cpus-per-task``, for one).

    When both are available the **smaller** is used, so neither an
    unconstrained cpuset nor a stale environment variable can oversubscribe the
    allocation.

    Parameters
    ----------
    num_workers : int or None
        Requested worker count, or ``None`` to size from the allocation.

    Returns
    -------
    int
        Concrete number of worker threads to use, at least 1.
    """
    if num_workers is not None:
        return num_workers

    affinity = _affinity_cpu_count()
    slurm_cpus = _slurm_cpus_per_task()

    if affinity is not None and slurm_cpus is not None:
        resolved = min(affinity, slurm_cpus)
        source = f"min(cpu affinity={affinity}, SLURM_CPUS_PER_TASK={slurm_cpus})"
    elif affinity is not None:
        resolved, source = affinity, f"cpu affinity={affinity}"
    elif slurm_cpus is not None:
        resolved, source = slurm_cpus, f"SLURM_CPUS_PER_TASK={slurm_cpus}"
    else:
        resolved, source = os.cpu_count() or 1, "os.cpu_count()"

    resolved = max(1, resolved)
    logger.info("Resolved num_workers=%d from %s.", resolved, source)
    return resolved
