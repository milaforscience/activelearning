"""Loss functions for sequence-based S3-GFN training."""

from __future__ import annotations

# Adapted from hyeonahkimm/s3gfn; see THIRD_PARTY_NOTICES.md.

import math
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor


def sequence_log_probabilities_from_logits(
    shifted_logits: Tensor,
    labels: Tensor,
    pad_token_id: int,
) -> Tensor:
    """Sum next-token log probabilities over non-padding labels.

    This is the shared scoring core: ``shifted_logits[:, i]`` must be the
    distribution predicting ``labels[:, i]``. Padding labels contribute zero,
    while EOS is included because it is a regular non-padding label.

    Parameters
    ----------
    shifted_logits : Tensor
        Logits aligned to ``labels`` with shape
        ``(batch, sequence_length - 1, vocabulary_size)``.
    labels : Tensor
        Next-token targets with shape ``(batch, sequence_length - 1)``.
    pad_token_id : int
        Token id excluded from the sequence sum.

    Returns
    -------
    Tensor
        Sequence log probabilities with shape ``(batch,)``.

    Raises
    ------
    ValueError
        If the logits do not align with the shifted labels.
    """
    if shifted_logits.shape[:2] != labels.shape:
        raise ValueError(
            "The causal LM logits must align with the shifted sequence labels."
        )

    token_log_probabilities = functional.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    )
    token_log_probabilities = token_log_probabilities.squeeze(-1)
    return token_log_probabilities.masked_fill(
        labels.eq(pad_token_id),
        0.0,
    ).sum(dim=-1)


def sequence_log_probabilities(
    causal_lm: Any,
    input_ids: Tensor,
    pad_token_id: int,
) -> Tensor:
    """Compute one autoregressive log probability per padded sequence.

    The model receives all tokens except the last and each position is scored
    against the next token. Padding labels are masked, while EOS is included
    because it is a regular non-padding label.

    Parameters
    ----------
    causal_lm : Any
        Hugging Face compatible causal language model.
    input_ids : Tensor
        Integer tensor with shape ``(batch, sequence_length)``.
    pad_token_id : int
        Token id excluded from the sequence sum.

    Returns
    -------
    Tensor
        Sequence log probabilities with shape ``(batch,)``.

    Raises
    ------
    ValueError
        If ``input_ids`` is not a matrix, ``pad_token_id`` is negative, or
        the causal-model logits do not align with the shifted labels.
    TypeError
        If ``input_ids`` does not contain integer token ids.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence_length).")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must contain integer token ids.")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be nonnegative.")

    batch_size, sequence_length = input_ids.shape
    if batch_size == 0 or sequence_length < 2:
        return torch.zeros(batch_size, device=input_ids.device)

    model_input_ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    attention_mask = model_input_ids.ne(pad_token_id).long()
    outputs = causal_lm(
        input_ids=model_input_ids,
        attention_mask=attention_mask,
    )
    return sequence_log_probabilities_from_logits(
        outputs.logits,
        labels=labels,
        pad_token_id=pad_token_id,
    )


def relative_trajectory_balance_loss(
    policy_log_probabilities: Tensor,
    prior_log_probabilities: Tensor,
    reward_scores: Tensor,
    log_z: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Compute the mean squared relative trajectory-balance residual.

    For a trajectory ``tau`` ending at molecule ``x``, relative TB matches
    the log ratio ``log((Z * P_policy(tau)) / (R(x) * P_prior(tau)))``.

    This function represents the positive reward as ``R(x) = exp(beta * r(x))``
    and receives ``r(x)`` through the ``reward_scores`` argument.  Consequently,
    ``beta * reward_scores`` is ``log R(x)``.  The argument is therefore a
    scaled reward score, not ``R(x)`` itself, and must not be passed through
    ``log`` again.  This convention also allows reward scores to be negative.

    Parameters
    ----------
    policy_log_probabilities : Tensor
        Trainable policy sequence log probabilities.
    prior_log_probabilities : Tensor
        Frozen prior sequence log probabilities.
    reward_scores : Tensor
        Finite scaled reward scores ``r(x)`` aligned with the trajectories.
        They define the positive reward through ``R(x) = exp(beta * r(x))``.
    log_z : Tensor
        Trainable scalar log normalizer.
    beta : float
        Reward inverse-temperature coefficient.

    Returns
    -------
    Tensor
        Scalar mean squared RTB residual.

    Raises
    ------
    ValueError
        If tensor shapes do not align, ``log_z`` is not scalar, ``beta`` is
        not finite, or a reward score is non-finite.
    """
    if policy_log_probabilities.ndim != 1:
        raise ValueError("policy_log_probabilities must be one-dimensional.")
    if prior_log_probabilities.shape != policy_log_probabilities.shape:
        raise ValueError("prior and policy log probabilities must have equal shape.")
    if reward_scores.shape != policy_log_probabilities.shape:
        raise ValueError("Reward scores must align with policy log probabilities.")
    if log_z.numel() != 1:
        raise ValueError("log_z must be scalar.")
    if not math.isfinite(beta):
        raise ValueError("beta must be finite.")
    if not torch.isfinite(reward_scores).all():
        raise ValueError("Reward scores must be finite.")
    if policy_log_probabilities.numel() == 0:
        return log_z.sum() * 0.0

    target = prior_log_probabilities.detach() + beta * reward_scores
    residual = log_z.reshape(()) + policy_log_probabilities - target
    return residual.square().mean()


def negative_replay_contrastive_loss(
    positive_log_probabilities: Tensor,
    negative_log_probabilities: Tensor,
) -> Tensor:
    """Compute the S3-GFN contrastive auxiliary loss.

    Each positive is contrasted against the mean negative trajectory
    probability:

    ``-log(exp(pos) / (exp(pos) + mean_j exp(neg_j)))``.

    This is the normalized replay objective used by the upstream S3-GFN
    implementation. Averaging the negative probability mass keeps the
    auxiliary-loss scale independent of the number of sampled negatives, and
    averaging the positive terms keeps it independent of the positive batch
    size.

    Parameters
    ----------
    positive_log_probabilities : Tensor
        Sequence log probabilities for positive replay trajectories.
    negative_log_probabilities : Tensor
        Sequence log probabilities for negative replay trajectories.

    Raises
    ------
    ValueError
        If either input is not one-dimensional.

    Returns
    -------
    Tensor
        Scalar contrastive loss.
    """
    if positive_log_probabilities.ndim != 1:
        raise ValueError("positive_log_probabilities must be one-dimensional.")
    if negative_log_probabilities.ndim != 1:
        raise ValueError("negative_log_probabilities must be one-dimensional.")
    if (
        positive_log_probabilities.numel() == 0
        or negative_log_probabilities.numel() == 0
    ):
        return positive_log_probabilities.sum() * 0.0

    negative_log_mass = torch.logsumexp(
        negative_log_probabilities,
        dim=0,
    ) - math.log(float(negative_log_probabilities.numel()))
    per_positive_loss = (
        torch.logaddexp(positive_log_probabilities, negative_log_mass)
        - positive_log_probabilities
    )
    return per_positive_loss.mean()
