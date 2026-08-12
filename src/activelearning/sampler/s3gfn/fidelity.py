"""Terminal fidelity action for multi-fidelity S3-GFN trajectories."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FidelityActionHead(nn.Module):
    """Predict and sample a categorical fidelity action after a molecule.

    The head consumes the causal language model hidden state at the molecule's
    terminal token. Its zero initialization makes the initial policy uniform
    over the configured fidelity levels; the fixed terminal prior is uniform.
    """

    def __init__(self, hidden_size: int, n_fidelities: int) -> None:
        """Initialize a terminal fidelity-action head.

        Parameters
        ----------
        hidden_size : int
            Size of the terminal language-model hidden state.
        n_fidelities : int
            Number of fidelity actions.

        Raises
        ------
        ValueError
            If ``hidden_size`` is not positive or ``n_fidelities`` is not
            greater than one.
        """
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if n_fidelities <= 1:
            raise ValueError("n_fidelities must be greater than one.")

        self.hidden_size = hidden_size
        self.n_fidelities = n_fidelities
        self.projection = nn.Linear(hidden_size, n_fidelities)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, terminal_hidden_states: Tensor) -> Tensor:
        """Return one logit per fidelity action for each terminal state.

        Parameters
        ----------
        terminal_hidden_states : Tensor
            Floating-point hidden states at the molecule terminal token with
            shape ``(batch, hidden_size)``.

        Returns
        -------
        Tensor
            Fidelity logits with shape ``(batch, n_fidelities)``.

        Raises
        ------
        ValueError
            If the hidden states do not have the expected rank or width.
        """
        if terminal_hidden_states.ndim != 2:
            raise ValueError(
                "terminal_hidden_states must have shape (batch, hidden_size)."
            )
        if terminal_hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                "terminal_hidden_states has an unexpected hidden-size dimension."
            )
        return self.projection(terminal_hidden_states)

    def log_prob(
        self,
        terminal_hidden_states: Tensor,
        fidelity_indices: Tensor,
    ) -> Tensor:
        """Return policy log probabilities for selected fidelity actions.

        Parameters
        ----------
        terminal_hidden_states : Tensor
            Floating-point hidden states at the molecule terminal token with
            shape ``(batch, hidden_size)``.
        fidelity_indices : Tensor
            Zero-based integer action indices with shape ``(batch,)``.

        Returns
        -------
        Tensor
            Log probability of the selected action for each batch item, with
            shape ``(batch,)``.

        Raises
        ------
        ValueError
            If the hidden states or action indices are misaligned, or an
            action index is outside the configured range.
        TypeError
            If ``fidelity_indices`` is not integer-valued.
        """
        logits = self(terminal_hidden_states)
        indices = self._validate_indices(
            fidelity_indices,
            batch_size=logits.shape[0],
            device=logits.device,
        )
        return (
            torch.log_softmax(logits, dim=-1)
            .gather(
                dim=-1,
                index=indices.unsqueeze(-1),
            )
            .squeeze(-1)
        )

    def sample(self, terminal_hidden_states: Tensor) -> Tensor:
        """Sample one zero-based fidelity action for each terminal state.

        Parameters
        ----------
        terminal_hidden_states : Tensor
            Floating-point hidden states at the molecule terminal token with
            shape ``(batch, hidden_size)``.

        Returns
        -------
        Tensor
            Sampled integer action indices with shape ``(batch,)`` and dtype
            ``torch.long``.

        Raises
        ------
        ValueError
            If the hidden states do not have the expected rank or width.
        """
        probabilities = torch.softmax(self(terminal_hidden_states), dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    def uniform_prior_log_prob(
        self,
        fidelity_indices: Tensor,
        *,
        batch_size: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return fixed uniform-prior log probabilities for actions.

        Parameters
        ----------
        fidelity_indices : Tensor
            Zero-based integer action indices with shape ``(batch,)``.
        batch_size : int or None, optional
            Expected number of action indices. When provided, the value is
            checked against the first dimension of ``fidelity_indices``.
        device : torch.device
            Device for the returned tensor.
        dtype : torch.dtype
            Floating-point dtype for the returned tensor.

        Returns
        -------
        Tensor
            Constant log probabilities equal to
            ``-log(n_fidelities)`` with shape ``(batch,)``.

        Raises
        ------
        ValueError
            If the action indices are misaligned or outside the configured
            range.
        TypeError
            If ``fidelity_indices`` is not integer-valued.
        """
        indices = self._validate_indices(
            fidelity_indices,
            batch_size=batch_size,
            device=device,
        )
        return torch.full(
            (indices.numel(),),
            -math.log(self.n_fidelities),
            device=device,
            dtype=dtype,
        )

    def _validate_indices(
        self,
        fidelity_indices: Tensor,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Validate and normalize a batch of categorical action indices."""
        if fidelity_indices.ndim != 1:
            raise ValueError("fidelity_indices must be one-dimensional.")
        if batch_size is not None and fidelity_indices.shape[0] != batch_size:
            raise ValueError("Fidelity indices must align with terminal states.")
        if fidelity_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("fidelity_indices must contain integer indices.")

        indices = fidelity_indices.to(
            device=device if device is not None else fidelity_indices.device,
            dtype=torch.long,
        )
        if indices.numel() and (
            bool(torch.any(indices < 0))
            or bool(torch.any(indices >= self.n_fidelities))
        ):
            raise ValueError("fidelity_indices contain an unknown action.")
        return indices
