"""GFlowNet environments for molecule generation."""

from __future__ import annotations

from typing import List, Optional, Union

from gflownet.envs.sequences.selfies import Selfies
from torchtyping import TensorType

from activelearning_molecules.samplers.molecule_utils import (
    canonicalize_connected_smiles,
)
from activelearning_molecules.samplers.random_sampler import _require_selfies
from activelearning_molecules.samplers.s3gfn._optional import require_rdkit


class SelfiesSmiles(Selfies):
    """SELFIES token environment whose proxy states are canonical SMILES.

    The GFlowNet policy builds SELFIES sequences, while candidates, surrogate
    encoders, and oracles receive SMILES. Sequences that do not canonicalize to
    one connected molecule keep their raw decoded SMILES; the xTB oracle scores
    those as ``NaN`` and the dataset drops them.
    """

    def _get_seq_length(
        self,
        state: Optional[TensorType["max_length"]] = None,  # noqa: F821
    ) -> int:
        """Return the effective length of a state, ignoring padding.

        Copies the state to host memory once instead of the per-element reads
        in :meth:`SequenceBase._get_seq_length`, each of which forces a device
        sync. Returns an ``int`` rather than a 0-dim tensor.

        Parameters
        ----------
        state : tensor, optional
            The input sequence. If None, ``self.state`` is used.

        Returns
        -------
        int
            Number of non-padding tokens in the state.
        """
        seq = self._get_state(state).tolist()
        try:
            return seq.index(self.pad_idx)
        except ValueError:
            return len(seq)

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[TensorType["max_length"]] = None,  # noqa: F821
        done: Optional[bool] = None,
    ) -> List[bool]:
        """Return the mask of invalid forward actions.

        Same result as :meth:`SequenceBase.get_mask_invalid_actions_forward`,
        but derived from a single :meth:`_get_seq_length` call.

        Parameters
        ----------
        state : tensor, optional
            The input sequence. If None, ``self.state`` is used.
        done : bool, optional
            Whether the trajectory is done. If None, ``self.done`` is used.

        Returns
        -------
        list of bool
            One entry per action, True where the action is invalid.
        """
        dim = self.action_space_dim
        if self._get_done(done):
            return [True] * dim

        length = self._get_seq_length(state)
        eos_index = self.action_space.index(self.eos)
        if length < self.min_length:
            mask = [False] * dim
            mask[eos_index] = True
            return mask
        if length < self.max_length:
            return [False] * dim
        mask = [True] * dim
        mask[eos_index] = False
        return mask

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]],  # noqa: F821
            TensorType["batch", "max_length"],  # noqa: F821
        ],
    ) -> List[str]:
        """Decode a batch of SELFIES states to canonical SMILES strings."""
        selfies_module = _require_selfies()
        molecule_chem, _, _ = require_rdkit()
        smiles_strings = []
        for selfies_string in super().states2proxy(states):
            decoded = selfies_module.decoder(selfies_string)
            canonical = canonicalize_connected_smiles(
                decoded, molecule_chem=molecule_chem
            )
            smiles_strings.append(canonical if canonical is not None else decoded)
        return smiles_strings


__all__ = ["SelfiesSmiles"]
