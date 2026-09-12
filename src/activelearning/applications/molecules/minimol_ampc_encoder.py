"""MiniMol AmpC checkpoint support for molecular surrogate encoders."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor

from activelearning.applications.molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.applications.molecules.minimol_encoder import (
    MINIMOL_FINGERPRINT_DIM,
    MiniMolSmilesFixedEncoder,
    MiniMolSmilesEncoder,
    _graphium_float32_compatibility,
    _minimol_molecule_transform,
)

__all__ = ["MiniMolAmpcSmilesEncoder", "MiniMolAmpcSmilesFixedEncoder"]


@contextmanager
def _temporary_sys_path(path: Path | None) -> Iterator[None]:
    """Temporarily make a local collaborator package importable."""
    if path is None:
        yield
        return

    path_text = str(path)
    added = path_text not in sys.path
    if added:
        sys.path.insert(0, path_text)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path_text)


def _resolve_package_path(
    checkpoint_path: Path,
    package_path: str | Path | None,
) -> Path | None:
    """Resolve the directory containing the shared ``minimol_ampc`` package."""
    if package_path is not None:
        resolved = Path(package_path).expanduser().resolve()
        if not resolved.is_dir() or not (resolved / "minimol_ampc").is_dir():
            raise FileNotFoundError(
                "MiniMol AmpC package_path must contain a minimol_ampc directory: "
                f"{resolved}"
            )
        return resolved

    for candidate in (checkpoint_path.parent, checkpoint_path.parent.parent):
        if (candidate / "minimol_ampc").is_dir():
            return candidate.resolve()
    return None


def _load_ampc_encoder(
    checkpoint_path: Path,
    package_path: Path | None,
    device: str | torch.device | None,
) -> Any:
    """Load the collaborator's full-trunk MiniMol encoder."""
    try:
        with _temporary_sys_path(package_path):
            package = importlib.import_module("minimol_ampc")
    except ImportError as error:
        raise missing_molecules_dependency_error(
            "MiniMol AmpC encoder",
            error,
        ) from error

    encoder_class = getattr(package, "MiniMolAmpcEncoder", None)
    if encoder_class is None:
        raise TypeError("The MiniMol AmpC package must expose MiniMolAmpcEncoder.")
    with _graphium_float32_compatibility():
        encoder = encoder_class.load(checkpoint_path, device=device)

    # Graphium may featurize in worker processes, so a parent-process patch
    # alone does not reach the transform used by later encode calls.
    datamodule = encoder.model.trunk.datamodule
    datamodule.smiles_transformer = partial(
        _minimol_molecule_transform,
        **datamodule.featurization,
    )
    return encoder


class MiniMolAmpcSmilesFixedEncoder(MiniMolSmilesFixedEncoder):
    """Encode SMILES as fixed ``pooled512`` AmpC representations."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        package_path: str | Path | None = None,
        device: str | torch.device | None = "cpu",
        batch_size: int = 100,
        cache_size: int = 4096,
    ) -> None:
        """Initialize the fine-tuned MiniMol AmpC fixed encoder.

        Parameters
        ----------
        checkpoint_path : Path or str
            Full ``MiniMolAmpcEncoder`` checkpoint.
        package_path : Path or str, optional
            Directory containing the sibling ``minimol_ampc`` package. If
            omitted, it is inferred from the checkpoint path when possible.
        device : str or torch.device, default="cpu"
            Device used by the frozen checkpoint encoder. ``None`` delegates
            device selection to the shared package.
        batch_size : int, default=100
            Maximum number of SMILES encoded per inference batch.
        cache_size : int, default=4096
            Maximum number of detached CPU fingerprints retained by the LRU
            cache. Zero disables caching.
        """
        resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.package_path = _resolve_package_path(
            resolved_checkpoint_path,
            package_path,
        )
        self.backend_device = device
        super().__init__(
            batch_size=batch_size,
            cache_size=cache_size,
            checkpoint_path=resolved_checkpoint_path,
        )

    def _build_minimol(self, checkpoint_path: Path | None) -> Any:
        """Load the full AmpC checkpoint through its bundled implementation."""
        if checkpoint_path is None:
            raise ValueError("MiniMol AmpC requires a checkpoint_path.")
        return _load_ampc_encoder(
            checkpoint_path,
            self.package_path,
            self.backend_device,
        )

    def _extract_fingerprints(self, smiles: list[str]) -> list[Tensor]:
        """Extract and validate the shared encoder's ``pooled512`` vectors."""
        with _graphium_float32_compatibility():
            outputs = np.asarray(
                self._minimol.encode(smiles, batch_size=self.batch_size),
                dtype=np.float32,
            )
        expected_shape = (len(smiles), MINIMOL_FINGERPRINT_DIM)
        if outputs.shape != expected_shape:
            raise ValueError(
                "MiniMol AmpC pooled512 output must have shape "
                f"{expected_shape}, got {outputs.shape}."
            )
        if not np.isfinite(outputs).all():
            raise ValueError(
                "MiniMol AmpC pooled512 output contains non-finite values."
            )

        return [
            torch.from_numpy(np.asarray(output, dtype=np.float32).copy())
            for output in outputs
        ]


class MiniMolAmpcSmilesEncoder(MiniMolSmilesEncoder):
    """Encode SMILES with the collaborator's fine-tuned MiniMol AmpC trunk.

    The fixed ``pooled512`` representation is passed through the same trainable
    DKL projection used by :class:`MiniMolSmilesEncoder`. The checkpoint's
    prediction head is not used.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        package_path: str | Path | None = None,
        device: str | torch.device | None = "cpu",
        batch_size: int = 100,
        latent_dim: int = 32,
        cache_size: int = 4096,
    ) -> None:
        """Initialize the fine-tuned MiniMol AmpC encoder."""
        self.package_path = package_path
        self.backend_device = device
        super().__init__(
            batch_size=batch_size,
            latent_dim=latent_dim,
            cache_size=cache_size,
            checkpoint_path=checkpoint_path,
        )
        self.package_path = self.fixed_encoder.package_path

    def _build_fixed_encoder(
        self,
        *,
        batch_size: int,
        cache_size: int,
        checkpoint_path: str | Path | None,
    ) -> MiniMolAmpcSmilesFixedEncoder:
        """Construct the fixed AmpC encoder used by DKL."""
        if checkpoint_path is None:
            raise ValueError("MiniMol AmpC requires a checkpoint_path.")
        return MiniMolAmpcSmilesFixedEncoder(
            checkpoint_path=checkpoint_path,
            package_path=self.package_path,
            device=self.backend_device,
            batch_size=batch_size,
            cache_size=cache_size,
        )
