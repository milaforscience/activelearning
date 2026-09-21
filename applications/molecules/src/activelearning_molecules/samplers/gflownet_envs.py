"""GFlowNet environments for molecule generation."""

from __future__ import annotations

from typing import List, Union

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
