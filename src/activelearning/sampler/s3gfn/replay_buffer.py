"""Replay storage used by the S3-GFN sampler.

The reward and FIFO policies follow the upstream implementation:
https://github.com/hyeonahkimm/s3gfn/blob/43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52/src/s3gfn/replay_buffer.py
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from activelearning.sampler.s3gfn._optional import require_rdkit


@dataclass(frozen=True)
class _ReplayEntry:
    """One canonicalized trajectory held in CPU replay storage."""

    token_ids: Tensor
    smiles: str
    reward_score: float
    fidelity_index: int | None = None
    fingerprint: Any | None = None


@dataclass(frozen=True)
class ReplayBatch:
    """A padded replay minibatch with scores and optional action indices.

    Attributes
    ----------
    input_ids : Tensor
        Padded integer token ids for the sampled trajectories.
    reward_scores : Tensor
        Scaled reward scores aligned with the sampled trajectories.
    smiles : tuple[str, ...]
        Molecule strings aligned with the sampled trajectories.
    fidelity_indices : Tensor or None
        Optional zero-based terminal action indices aligned with the sampled
        trajectories.
    """

    input_ids: Tensor
    reward_scores: Tensor
    smiles: tuple[str, ...]
    fidelity_indices: Tensor | None = None

    def __len__(self) -> int:
        """Return the number of sampled trajectories.

        Returns
        -------
        int
            Number of molecule strings in the batch.
        """
        return len(self.smiles)


class ReplayBuffer:
    """Bounded reward-prioritized or FIFO replay storage.

    ``policy="reward"`` keeps high-reward, chemically diverse trajectories.
    ``policy="fifo"`` keeps unique trajectories in insertion order for the
    negative auxiliary loss.
    """

    def __init__(
        self,
        pad_token_id: int,
        capacity: int = 4096,
        similarity_threshold: float = 0.75,
        policy: Literal["reward", "fifo"] = "reward",
        seed: int = 0,
    ) -> None:
        """Initialize bounded replay storage.

        Parameters
        ----------
        pad_token_id : int
            Token id used to pad trajectories when a replay batch is sampled.
        capacity : int, optional
            Maximum number of unique trajectories retained by the buffer.
        similarity_threshold : float, optional
            Tanimoto-similarity threshold used by the reward-prioritized
            policy. A new molecule at or above this similarity to a stored
            molecule replaces it only when its reward score is higher. This
            value is ignored by the FIFO policy.
        policy : {"reward", "fifo"}, optional
            Retention strategy. ``"reward"`` keeps high-scoring, chemically
            diverse trajectories; ``"fifo"`` keeps unique trajectories in
            insertion order and evicts the oldest entry when full.
        seed : int, optional
            Seed for the Python and PyTorch random generators used when
            sampling replay entries.

        Raises
        ------
        ValueError
            If ``pad_token_id`` is negative, ``capacity`` is not positive, the
            similarity threshold is outside ``[0, 1]``, or ``policy`` is
            unsupported.
        """
        if pad_token_id < 0:
            raise ValueError("pad_token_id must be nonnegative.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one.")
        if policy not in {"reward", "fifo"}:
            raise ValueError("policy must be 'reward' or 'fifo'.")
        if policy == "fifo" and similarity_threshold != 0.75:
            warnings.warn(
                "similarity_threshold is ignored when policy='fifo'.",
                UserWarning,
                stacklevel=2,
            )

        self.pad_token_id = pad_token_id
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.policy = policy
        self._entries: list[_ReplayEntry] = []
        self._smiles: set[str] = set()
        self._random = random.Random(seed)
        self._torch_generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        """Return the number of stored trajectories.

        Returns
        -------
        int
            Number of replay entries currently retained.
        """
        return len(self._entries)

    @property
    def entries(self) -> tuple[_ReplayEntry, ...]:
        """Return a read-only snapshot of the stored trajectories.

        Returns
        -------
        tuple[_ReplayEntry, ...]
            Stored entries in the buffer's current retention order.
        """
        return tuple(self._entries)

    def add_batch(
        self,
        input_ids: Tensor,
        smiles: Sequence[str],
        reward_scores: Tensor | Sequence[float] | None = None,
        fidelity_indices: Tensor | Sequence[int] | None = None,
    ) -> int:
        """Add aligned trajectories and scaled reward scores.

        ``reward_scores`` stores the scaled acquisition score ``r(x)`` used by
        RTB. The loss eventually converts it to the log target reward
        ``log R(x) = beta * r(x)``.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids with shape ``(batch, sequence_length)``. Each
            row is stored with its corresponding SMILES string.
        smiles : Sequence[str]
            Molecule strings aligned with the rows of ``input_ids``.
        reward_scores : Tensor or Sequence[float] or None, optional
            Scaled acquisition scores ``r(x)`` aligned with ``smiles``. If
            omitted, every trajectory receives a score of zero.
        fidelity_indices : Tensor or Sequence[int] or None, optional
            Optional terminal fidelity-action indices aligned with ``smiles``.
            Omit this field for molecule-only trajectories.

        Returns
        -------
        int
            Number of trajectories accepted into the buffer. Invalid,
            duplicate, or lower-priority entries may be rejected.

        Raises
        ------
        ValueError
            If the input shapes or lengths do not match, a reward score is
            not finite, or an action index is invalid.
        TypeError
            If the token ids or action indices are not integer-valued.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if input_ids.shape[0] != len(smiles):
            raise ValueError("Replay inputs and SMILES must have equal length.")
        # Normalize incoming values in float64 before storing Python floats so
        # priority comparisons do not depend on the caller's tensor dtype.
        reward_score_values = (
            torch.zeros(len(smiles), dtype=torch.float64)
            if reward_scores is None
            else torch.as_tensor(reward_scores, dtype=torch.float64).reshape(-1)
        )
        if reward_score_values.numel() != len(smiles):
            raise ValueError("Replay reward scores and SMILES must have equal length.")
        fidelity_index_values = self._normalize_fidelity_indices(
            fidelity_indices,
            count=len(smiles),
        )

        added = 0
        for index, smiles_string in enumerate(smiles):
            if self._add(
                token_ids=input_ids[index],
                smiles=smiles_string,
                reward_score=float(reward_score_values[index]),
                fidelity_index=(
                    None
                    if fidelity_index_values is None
                    else int(fidelity_index_values[index])
                ),
            ):
                added += 1
        return added

    def _add(
        self,
        token_ids: Tensor,
        smiles: str,
        reward_score: float,
        fidelity_index: int | None,
    ) -> bool:
        """Validate and add one trajectory according to the configured policy."""
        if token_ids.ndim != 1:
            raise ValueError("Each replay trajectory must be one-dimensional.")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Replay trajectories must contain integer token ids.")
        normalized_smiles = smiles.strip()
        if not normalized_smiles or normalized_smiles in self._smiles:
            return False
        if not math.isfinite(reward_score):
            raise ValueError("Replay reward scores must be finite.")

        entry = _ReplayEntry(
            token_ids=token_ids.detach().to(device="cpu", dtype=torch.long).clone(),
            smiles=normalized_smiles,
            reward_score=reward_score,
            fidelity_index=fidelity_index,
        )
        if self.policy == "fifo":
            return self._add_fifo(entry)

        Chem, _, AllChem = require_rdkit()
        molecule = Chem.MolFromSmiles(normalized_smiles)
        if molecule is None:
            return False
        entry = _ReplayEntry(
            token_ids=entry.token_ids,
            smiles=normalized_smiles,
            reward_score=reward_score,
            fidelity_index=fidelity_index,
            fingerprint=AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                radius=2,
                nBits=2048,
            ),
        )
        return self._add_reward(entry)

    def _add_fifo(self, entry: _ReplayEntry) -> bool:
        """Append a unique entry and evict the oldest when full."""
        if len(self._entries) == self.capacity:
            evicted = self._entries.pop(0)
            self._smiles.remove(evicted.smiles)
        self._entries.append(entry)
        self._smiles.add(entry.smiles)
        return True

    def _add_reward(self, entry: _ReplayEntry) -> bool:
        """Keep high-score diverse entries using Tanimoto similarity."""
        if self._entries:
            _, DataStructs, _ = require_rdkit()
            similarities = DataStructs.BulkTanimotoSimilarity(
                entry.fingerprint,
                [stored.fingerprint for stored in self._entries],
            )
            nearest_index, nearest_similarity = max(
                enumerate(similarities),
                key=lambda item: item[1],
            )
            if nearest_similarity >= self.similarity_threshold:
                if self._entries[nearest_index].reward_score >= entry.reward_score:
                    return False
                self._smiles.remove(self._entries[nearest_index].smiles)
                self._entries[nearest_index] = entry
                self._smiles.add(entry.smiles)
                return True

        if len(self._entries) < self.capacity:
            self._entries.append(entry)
            self._smiles.add(entry.smiles)
            return True

        lowest_index = min(
            range(len(self._entries)),
            key=lambda index: self._entries[index].reward_score,
        )
        if self._entries[lowest_index].reward_score >= entry.reward_score:
            return False
        self._smiles.remove(self._entries[lowest_index].smiles)
        self._entries[lowest_index] = entry
        self._smiles.add(entry.smiles)
        return True

    def sample(
        self,
        count: int,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
        reward_prioritized: bool = False,
        replace: bool = True,
    ) -> ReplayBatch:
        """Sample trajectories and pad them on the requested device.

        Parameters
        ----------
        count : int
            Maximum number of trajectories to sample.
        device : str or torch.device
            Device for the returned token ids and reward scores.
        dtype : torch.dtype or None, optional
            Floating-point dtype for returned reward scores and priority
            weights. Defaults to PyTorch's default floating-point dtype.
        reward_prioritized : bool, optional
            Whether to sample according to stored reward scores.
        replace : bool, optional
            Whether reward-prioritized sampling may select an entry more than
            once. This option has no effect for uniform sampling.

        Returns
        -------
        ReplayBatch
            Sampled token ids padded with ``pad_token_id``, reward scores in
            ``dtype``, corresponding SMILES strings, and optional terminal
            fidelity-action indices. The batch may be smaller than ``count``
            when the buffer contains fewer entries.

        Raises
        ------
        ValueError
            If ``count`` is negative.
        TypeError
            If ``dtype`` is not a floating-point torch dtype.
        """
        if count < 0:
            raise ValueError("count must be nonnegative.")
        score_dtype = torch.get_default_dtype() if dtype is None else dtype
        if not torch.empty((), dtype=score_dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point torch dtype.")
        sample_size = min(count, len(self._entries))
        if sample_size == 0:
            return ReplayBatch(
                input_ids=torch.empty((0, 0), dtype=torch.long, device=device),
                reward_scores=torch.empty(0, dtype=score_dtype, device=device),
                smiles=(),
            )

        if reward_prioritized:
            weights = torch.tensor(
                [entry.reward_score for entry in self._entries],
                dtype=score_dtype,
            )
            minimum = float(weights.min())
            if minimum <= 0.0:
                weights = weights - minimum
            weights = weights + torch.finfo(weights.dtype).eps
            if float(weights.sum()) == 0.0:
                weights.fill_(1.0)
            indices = torch.multinomial(
                weights,
                num_samples=sample_size,
                replacement=replace,
                generator=self._torch_generator,
            ).tolist()
            selected = [self._entries[index] for index in indices]
        else:
            selected = self._random.sample(self._entries, sample_size)

        return ReplayBatch(
            input_ids=pad_sequence(
                [entry.token_ids for entry in selected],
                batch_first=True,
                padding_value=self.pad_token_id,
            ).to(device),
            reward_scores=torch.tensor(
                [entry.reward_score for entry in selected],
                dtype=score_dtype,
                device=device,
            ),
            smiles=tuple(entry.smiles for entry in selected),
            fidelity_indices=self._sampled_fidelity_indices(selected, device),
        )

    def _normalize_fidelity_indices(
        self,
        fidelity_indices: Tensor | Sequence[int] | None,
        *,
        count: int,
    ) -> Tensor | None:
        """Validate optional action indices and keep replay mode consistent."""
        if fidelity_indices is None:
            if any(entry.fidelity_index is not None for entry in self._entries):
                raise ValueError(
                    "Fidelity indices are required after action trajectories "
                    "have been added."
                )
            return None

        values = torch.as_tensor(fidelity_indices)
        if values.ndim != 1 or values.numel() != count:
            raise ValueError(
                "Replay fidelity indices and SMILES must have equal length."
            )
        if values.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("Replay fidelity indices must be integers.")
        if values.numel() and bool(torch.any(values < 0)):
            raise ValueError("Replay fidelity indices must be nonnegative.")
        if any(entry.fidelity_index is None for entry in self._entries):
            raise ValueError(
                "Fidelity indices must be omitted after sequence-only "
                "trajectories have been added."
            )
        return values.to(dtype=torch.long, device="cpu")

    @staticmethod
    def _sampled_fidelity_indices(
        entries: Sequence[_ReplayEntry],
        device: str | torch.device,
    ) -> Tensor | None:
        """Return selected action indices when the batch stores them."""
        values = [entry.fidelity_index for entry in entries]
        if not values:
            return None
        if any(value is None for value in values):
            if not all(value is None for value in values):
                raise RuntimeError("Replay entries mix action and sequence modes.")
            return None
        return torch.tensor(values, dtype=torch.long, device=device)
