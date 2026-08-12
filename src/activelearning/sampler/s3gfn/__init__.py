"""Standalone S3-GFN algorithm core for molecular sampler integration."""

from activelearning.sampler.s3gfn._optional import S3GFNOptionalDependencyError
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from activelearning.sampler.s3gfn.losses import (
    relative_trajectory_balance_loss,
    sequence_log_probabilities,
    summed_negative_infonce_loss,
)
from activelearning.sampler.s3gfn.model import GeneratedSequences, S3GFNModel
from activelearning.sampler.s3gfn.replay_buffer import ReplayBatch, ReplayBuffer
from activelearning.sampler.s3gfn.synthesizability import (
    SAScoreSynthesizability,
    passes_sa_threshold,
)

__all__ = [
    "GeneratedSequences",
    "FidelityActionHead",
    "ReplayBatch",
    "ReplayBuffer",
    "S3GFNModel",
    "S3GFNOptionalDependencyError",
    "SAScoreSynthesizability",
    "passes_sa_threshold",
    "relative_trajectory_balance_loss",
    "sequence_log_probabilities",
    "summed_negative_infonce_loss",
]
