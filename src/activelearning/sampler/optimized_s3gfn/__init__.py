"""Experimental optimized S3-GFN implementation.

The reference implementation remains available from
``activelearning.sampler.s3gfn``. This package is intentionally separate while
the optimized inference and prior-cache paths are benchmarked and validated.
"""

from activelearning.sampler.optimized_s3gfn.model import (
    GeneratedSequences,
    OptimizedS3GFNModel,
)
from activelearning.sampler.optimized_s3gfn.replay_buffer import (
    OptimizedReplayBatch,
    OptimizedReplayBuffer,
)
from activelearning.sampler.optimized_s3gfn.sampler import OptimizedS3GFNSampler
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from activelearning.sampler.s3gfn.replay_buffer import ReplayBatch, ReplayBuffer

__all__ = [
    "FidelityActionHead",
    "GeneratedSequences",
    "OptimizedS3GFNModel",
    "OptimizedS3GFNSampler",
    "OptimizedReplayBatch",
    "OptimizedReplayBuffer",
    "ReplayBatch",
    "ReplayBuffer",
]
