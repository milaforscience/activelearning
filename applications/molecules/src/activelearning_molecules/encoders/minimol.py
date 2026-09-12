"""MiniMol-backed SMILES encoders for molecular surrogate models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from activelearning_molecules._optional import (
    missing_molecules_dependency_error,
)
from activelearning.surrogate.encoder import FixedEncoder, LatentEncoder

__all__ = ["MiniMolSmilesEncoder", "MiniMolSmilesFixedEncoder"]

MINIMOL_FINGERPRINT_DIM = 512


@contextmanager
def _graphium_float32_compatibility() -> Iterator[None]:
    """Use a SciPy-compatible dtype while Graphium featurizes a molecule.

    Graphium 2.4.7's graph-dict helper drops its configured dtype before
    calling the adjacency helper. The resulting ``float16`` sparse matrix is
    rejected by SciPy. The patch is scoped to the active featurization call
    and is applied independently in each process used by Graphium's
    featurizer.
    """
    import numpy as np
    from graphium.features import featurizer

    original = featurizer.mol_to_adj_and_features

    def _featurize_with_float32(*args: Any, **kwargs: Any) -> Any:
        kwargs["dtype"] = np.float32
        return original(*args, **kwargs)

    featurizer.mol_to_adj_and_features = _featurize_with_float32
    try:
        yield
    finally:
        featurizer.mol_to_adj_and_features = original


def _minimol_molecule_transform(molecule: Any, **kwargs: Any) -> Any:
    """Featurize one molecule with Graphium's sparse dtype workaround."""
    from graphium.features import featurizer

    with _graphium_float32_compatibility():
        return featurizer.mol_to_pyggraph(molecule, **kwargs)


def _load_minimol_checkpoint(model: Any, checkpoint_path: Path) -> None:
    """Load a predictor state dict into a constructed MiniMol model."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict: Any = checkpoint
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(key, str) for key in state_dict
    ):
        raise TypeError(
            "MiniMol checkpoint must be a state dict or a mapping containing "
            "a string-keyed 'state_dict'."
        )

    fingerprinter = getattr(model, "predictor", None)
    predictor = getattr(fingerprinter, "predictor", None)
    if not isinstance(predictor, nn.Module):
        raise TypeError(
            "MiniMol does not expose the expected Graphium predictor for "
            "checkpoint loading."
        )

    target_keys = set(predictor.state_dict())
    if not target_keys.intersection(state_dict):
        raise ValueError(
            f"MiniMol checkpoint {checkpoint_path} contains no parameters "
            "matching the constructed predictor."
        )
    predictor.load_state_dict(state_dict, strict=False)


def _build_compatible_minimol(
    minimol_class: type[Any],
    *,
    batch_size: int,
    checkpoint_path: Path | None = None,
) -> Any:
    """Construct MiniMol with the Graphium sparse dtype workaround."""
    with _graphium_float32_compatibility():
        model = minimol_class(batch_size=batch_size)

    if checkpoint_path is not None:
        _load_minimol_checkpoint(model, checkpoint_path)

    model.datamodule.smiles_transformer = partial(
        _minimol_molecule_transform,
        **model.datamodule.featurization,
    )
    return model


def _load_minimol() -> Callable[..., Any]:
    """Load a MiniMol constructor on demand."""
    try:
        from minimol import Minimol
    except ImportError as error:  # pragma: no cover - optional dependency
        raise missing_molecules_dependency_error(
            "MiniMol SMILES encoder",
            error,
        ) from error

    return partial(_build_compatible_minimol, Minimol)


class MiniMolSmilesFixedEncoder(FixedEncoder):
    """Encode SMILES as fixed 512-dimensional MiniMol fingerprints."""

    feature_dim = MINIMOL_FINGERPRINT_DIM

    def __init__(
        self,
        *,
        batch_size: int = 100,
        cache_size: int = 4096,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialize the fixed MiniMol encoder.

        Parameters
        ----------
        batch_size : int, default=100
            Maximum number of SMILES sent to MiniMol per extraction batch.
        cache_size : int, default=4096
            Maximum number of detached CPU fingerprints retained by the LRU
            cache. Zero disables caching.
        checkpoint_path : Path or str, optional
            Optional predictor state-dict checkpoint to load over MiniMol's
            bundled pretrained weights.

        Raises
        ------
        ValueError
            If ``batch_size`` is not positive or ``cache_size`` is negative.
        FileNotFoundError
            If ``checkpoint_path`` is provided but does not point to a file.
        ImportError
            If MiniMol is not installed.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")

        resolved_checkpoint_path = (
            Path(checkpoint_path).expanduser() if checkpoint_path is not None else None
        )
        if (
            resolved_checkpoint_path is not None
            and not resolved_checkpoint_path.is_file()
        ):
            raise FileNotFoundError(
                "MiniMol checkpoint_path does not point to a file: "
                f"{resolved_checkpoint_path}"
            )

        self.batch_size = batch_size
        self.cache_size = cache_size
        self.checkpoint_path = resolved_checkpoint_path
        self._minimol = self._build_minimol(resolved_checkpoint_path)
        self._fingerprint_cache: OrderedDict[str, Tensor] = OrderedDict()

    def _build_minimol(self, checkpoint_path: Path | None) -> Any:
        """Construct stock MiniMol and optionally load its predictor state dict."""
        return _load_minimol()(
            batch_size=self.batch_size,
            checkpoint_path=checkpoint_path,
        )

    def encode(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert raw SMILES into MiniMol fingerprint tensors.

        Parameters
        ----------
        values : Sequence[Any]
            Raw molecular strings. Every item must be a string.
        device : torch.device
            Device on which the returned fingerprint tensor is allocated.

        Returns
        -------
        Tensor
            Tensor of shape ``(B, 512)`` containing frozen MiniMol
            fingerprints.

        Raises
        ------
        ValueError
            If an input is not a string, or if MiniMol returns malformed or
            non-finite fingerprints.
        TypeError
            If MiniMol does not return a sequence of tensors.
        """
        strings: list[str] = []
        for value in values:
            if not isinstance(value, str):
                raise ValueError(
                    "MiniMol SMILES fixed encoders require string inputs, got "
                    f"{type(value).__name__}."
                )
            strings.append(value)

        if not strings:
            return torch.empty(
                (0, MINIMOL_FINGERPRINT_DIM),
                dtype=torch.float32,
                device=device,
            )

        resolved: dict[str, Tensor] = {}
        missing: list[str] = []
        missing_set: set[str] = set()
        for string in strings:
            if string in resolved:
                continue
            cached = self._fingerprint_cache.get(string) if self.cache_size else None
            if cached is not None:
                self._fingerprint_cache.move_to_end(string)
                resolved[string] = cached
            elif string not in missing_set:
                missing.append(string)
                missing_set.add(string)

        if missing:
            missing_features = self._extract_fingerprints(missing)
            for string, feature in zip(missing, missing_features):
                resolved[string] = feature
                if self.cache_size:
                    self._fingerprint_cache[string] = feature
                    self._fingerprint_cache.move_to_end(string)
                    while len(self._fingerprint_cache) > self.cache_size:
                        self._fingerprint_cache.popitem(last=False)

        return torch.stack(
            [resolved[string] for string in strings],
            dim=0,
        ).to(device=device, dtype=torch.float32)

    def _extract_fingerprints(self, smiles: list[str]) -> list[Tensor]:
        """Run frozen MiniMol inference and validate its fingerprint output."""
        with torch.inference_mode():
            outputs = self._minimol(smiles)
        if not isinstance(outputs, (list, tuple)):
            raise TypeError("MiniMol must return a list of fingerprint tensors.")
        if len(outputs) != len(smiles):
            raise ValueError(
                "MiniMol returned "
                f"{len(outputs)} fingerprints for {len(smiles)} SMILES."
            )

        features: list[Tensor] = []
        for index, output in enumerate(outputs):
            if not isinstance(output, Tensor):
                raise TypeError(
                    f"MiniMol fingerprint at index {index} is not a tensor."
                )
            if output.ndim != 1 or output.numel() != MINIMOL_FINGERPRINT_DIM:
                raise ValueError(
                    f"MiniMol fingerprint at index {index} must have shape "
                    f"({MINIMOL_FINGERPRINT_DIM},), got {tuple(output.shape)}."
                )
            feature = output.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(feature).all():
                raise ValueError(
                    f"MiniMol fingerprint at index {index} contains non-finite values."
                )
            features.append(feature)
        return features


class MiniMolSmilesEncoder(LatentEncoder):
    """Encode SMILES with frozen MiniMol fingerprints and a trainable head.

    MiniMol returns fixed-width graph fingerprints rather than token IDs or
    hidden states from a PyTorch module. The fingerprints are kept frozen and
    passed through a trainable linear projection so the DKL surrogate can
    adapt the representation during fitting.
    """

    def __init__(
        self,
        *,
        batch_size: int = 100,
        latent_dim: int = 32,
        cache_size: int = 4096,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialize the MiniMol encoder and trainable projection."""
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")

        self.fixed_encoder = self._build_fixed_encoder(
            batch_size=batch_size,
            cache_size=cache_size,
            checkpoint_path=checkpoint_path,
        )
        self.batch_size = self.fixed_encoder.batch_size
        self.latent_dim = latent_dim
        self.cache_size = self.fixed_encoder.cache_size
        self.checkpoint_path = self.fixed_encoder.checkpoint_path
        self.projection = nn.Linear(MINIMOL_FINGERPRINT_DIM, latent_dim)

    def _build_fixed_encoder(
        self,
        *,
        batch_size: int,
        cache_size: int,
        checkpoint_path: str | Path | None,
    ) -> MiniMolSmilesFixedEncoder:
        """Construct the fixed MiniMol encoder used by DKL."""
        return MiniMolSmilesFixedEncoder(
            batch_size=batch_size,
            cache_size=cache_size,
            checkpoint_path=checkpoint_path,
        )

    def prepare_inputs(
        self,
        values: Sequence[Any],
        *,
        device: torch.device,
    ) -> Tensor:
        """Convert raw SMILES into frozen MiniMol fingerprint tensors."""
        return self.fixed_encoder.encode(values, device=device)

    def forward(self, model_inputs: Tensor) -> Tensor:
        """Project MiniMol fingerprints into the DKL latent space."""
        if model_inputs.ndim != 2:
            raise ValueError(
                "MiniMol fingerprints must be 2-D (B, 512), got "
                f"{tuple(model_inputs.shape)}."
            )
        if model_inputs.shape[-1] != MINIMOL_FINGERPRINT_DIM:
            raise ValueError(
                "MiniMol fingerprints must have width "
                f"{MINIMOL_FINGERPRINT_DIM}, got {model_inputs.shape[-1]}."
            )
        projection_inputs = model_inputs.to(
            device=self.projection.weight.device,
            dtype=self.projection.weight.dtype,
        )
        return self.projection(projection_inputs)
