"""GP-MoLFormer policy/prior wrapper for S3-GFN molecule training.

This wrapper treats each tokenized SMILES string, optionally followed by a
terminal fidelity action, as a trajectory. It keeps a trainable policy and a
frozen copy of the pretrained model as the prior. RTB trains the policy by
comparing their trajectory probabilities with a reward-weighted target:

``log((Z * P_policy(tau)) / (R(x) * P_prior(tau)))``.

The API accepts scaled reward scores ``r(x)`` and converts them to positive
rewards with ``R(x) = exp(beta * r(x))``. The implementation follows the
generation and loss flow of the upstream S3-GFN trainer:
https://github.com/hyeonahkimm/s3gfn/blob/43aa7b310e9e03ef71ea0bd0cce501a48b6e2d52/src/s3gfn/train.py
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from activelearning.sampler.s3gfn._optional import require_transformers
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from activelearning.sampler.s3gfn.losses import (
    negative_replay_contrastive_loss,
    relative_trajectory_balance_loss,
    sequence_log_probabilities,
    sequence_log_probabilities_from_logits,
)


DEFAULT_GP_MOLFORMER_MODEL = "ibm-research/GP-MoLFormer-Uniq"
DEFAULT_GP_MOLFORMER_TOKENIZER = "ibm-research/MoLFormer-XL-both-10pct"


@dataclass(frozen=True)
class GeneratedSequences:
    """Generated token ids and their decoded SMILES.

    Attributes
    ----------
    input_ids : Tensor
        Token ids returned by the policy. Rows correspond to ``smiles`` and may
        contain padding.
    smiles : tuple[str, ...]
        Strings decoded from ``input_ids`` in the same order.
    fidelity_indices : Tensor or None
        Optional terminal fidelity-action indices aligned with ``smiles``.
    """

    input_ids: Tensor
    smiles: tuple[str, ...]
    fidelity_indices: Tensor | None = None


class S3GFNModel(nn.Module):
    """Wrap a trainable SMILES policy and a frozen pretrained prior.

    The policy and prior are separate causal language models, usually loaded
    from the same GP-MoLFormer checkpoint. The policy generates SMILES and is
    trained with RTB and, optionally, the replay auxiliary loss. The prior is
    frozen, kept in evaluation mode, and supplies the reference sequence
    probabilities used by RTB.

    The trainable ``log_z`` parameter estimates the RTB normalizer ``log Z``.
    Tokenization, generation, and sequence likelihoods are provided here so
    callers can work directly with SMILES batches. An optional fidelity action
    head adds one categorical action after the molecule's terminal token.
    """

    def __init__(
        self,
        policy: nn.Module,
        prior: nn.Module,
        tokenizer: Any,
        initial_log_z: float = 0.0,
        fidelity_head: FidelityActionHead | None = None,
    ) -> None:
        """Create an S3-GFN model from a policy, prior, and tokenizer.

        Parameters
        ----------
        policy : nn.Module
            Causal language model to train as the S3-GFN policy.
        prior : nn.Module
            Causal language model used as the frozen RTB reference.
        tokenizer : Any
            Hugging Face-compatible tokenizer with pad and EOS token ids.
        initial_log_z : float, optional
            Initial value for the trainable RTB normalizer ``log Z``.
        fidelity_head : FidelityActionHead or None, optional
            Optional head for a terminal fidelity action. Omit it for
            molecule-only trajectories.

        Raises
        ------
        ValueError
            If ``initial_log_z`` is non-finite or the tokenizer lacks a pad
            or EOS token id.
        """
        super().__init__()
        if not math.isfinite(initial_log_z):
            raise ValueError("initial_log_z must be finite.")
        if tokenizer.pad_token_id is None:
            raise ValueError("The S3-GFN tokenizer must define pad_token_id.")
        if tokenizer.eos_token_id is None:
            raise ValueError("The S3-GFN tokenizer must define eos_token_id.")
        if int(tokenizer.pad_token_id) == int(tokenizer.eos_token_id):
            raise ValueError("The S3-GFN tokenizer must use distinct pad and EOS ids.")
        if getattr(tokenizer, "padding_side", None) != "right":
            raise ValueError("The S3-GFN tokenizer must use right-side padding.")

        self.policy = policy
        self.prior = prior
        self.tokenizer = tokenizer
        self.fidelity_head = fidelity_head
        if self.fidelity_head is not None:
            policy_parameter = next(self.policy.parameters(), None)
            if policy_parameter is not None:
                self.fidelity_head.to(
                    device=policy_parameter.device,
                    dtype=policy_parameter.dtype,
                )
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.log_z = nn.Parameter(torch.tensor(float(initial_log_z)))
        self._last_auxiliary_loss: Tensor | None = None

        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)
        self.prior.eval()

    @classmethod
    def from_pretrained(
        cls,
        policy_model_name_or_path: str = DEFAULT_GP_MOLFORMER_MODEL,
        prior_model_name_or_path: str | None = None,
        tokenizer_name_or_path: str = DEFAULT_GP_MOLFORMER_TOKENIZER,
        *,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        initial_log_z: float = 0.0,
        n_fidelities: int | None = None,
        deterministic_eval: bool | None = True,
    ) -> "S3GFNModel":
        """Load a policy, prior, and tokenizer from Hugging Face or disk.

        This creates one tokenizer and two independent language-model
        instances. If no prior path is given, both models use the policy
        checkpoint. The returned wrapper freezes the prior and keeps it in
        evaluation mode.

        Parameters
        ----------
        policy_model_name_or_path : str
            Model identifier or local path for the trainable policy.
        prior_model_name_or_path : str or None, optional
            Model identifier or local path for the frozen prior. Defaults to
            the policy checkpoint.
        tokenizer_name_or_path : str
            Tokenizer identifier or local path.
        trust_remote_code : bool, optional
            Whether Transformers may load custom checkpoint code.
        cache_dir : str or None, optional
            Directory for Hugging Face's downloaded files.
        device : str or torch.device, optional
            Device on which to place the model.
        dtype : torch.dtype or None, optional
            Optional dtype for both language models.
        initial_log_z : float, optional
            Initial value for the trainable RTB normalizer ``log Z``.
        n_fidelities : int or None, optional
            Number of configured fidelity levels. A terminal action head is
            created only when this is greater than one.
        deterministic_eval : bool or None, optional
            GP-MoLFormer-specific flag forwarded to Transformers. It defaults
            to ``True`` because GP-MoLFormer approximates attention with
            random features that are otherwise redrawn on every forward pass,
            even in evaluation mode. Without it the frozen prior would return
            a different likelihood for the same molecule on each call, making
            the RTB target noisy. Note that the checkpoint's own config sets
            it to ``False``, so omitting it is not equivalent to leaving it
            unset. Pass ``None`` only for checkpoints that do not define this
            custom argument.

        Returns
        -------
        S3GFNModel
            A model with a trainable policy and frozen prior.

        Raises
        ------
        ValueError
            If ``n_fidelities`` is provided but is not positive, or if the
            policy configuration does not expose a supported hidden-size
            field when a terminal action head is required.
        S3GFNOptionalDependencyError
            If the optional Transformers dependency is unavailable.
        """
        AutoModelForCausalLM, AutoTokenizer = require_transformers()
        if n_fidelities is not None and n_fidelities <= 0:
            raise ValueError("n_fidelities must be positive when provided.")
        prior_name = prior_model_name_or_path or policy_model_name_or_path
        if prior_name is None:
            raise ValueError("A policy or prior model path must be provided.")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
        }
        if deterministic_eval is not None:
            model_kwargs["deterministic_eval"] = deterministic_eval
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        prior = AutoModelForCausalLM.from_pretrained(prior_name, **model_kwargs)
        policy = AutoModelForCausalLM.from_pretrained(
            policy_model_name_or_path,
            **model_kwargs,
        )
        fidelity_head = None
        if n_fidelities is not None and n_fidelities > 1:
            fidelity_head = FidelityActionHead(
                hidden_size=_model_hidden_size(policy),
                n_fidelities=n_fidelities,
            )
        return cls(
            policy=policy,
            prior=prior,
            tokenizer=tokenizer,
            initial_log_z=initial_log_z,
            fidelity_head=fidelity_head,
        ).to(device)

    @property
    def device(self) -> torch.device:
        """Return the device used for model inputs and computations.

        The property follows the ``log_z`` parameter, which moves with the
        model when :meth:`torch.nn.Module.to` is called.
        """
        return self.log_z.device

    @property
    def last_auxiliary_loss(self) -> float | None:
        """Return the most recently evaluated replay auxiliary loss."""
        if self._last_auxiliary_loss is None:
            return None
        return float(self._last_auxiliary_loss.item())

    def train(self, mode: bool = True) -> "S3GFNModel":
        """Set the policy's training mode and keep the prior in evaluation.

        Parameters
        ----------
        mode : bool, optional
            ``True`` for training mode or ``False`` for evaluation mode.

        Returns
        -------
        S3GFNModel
            This model, allowing calls such as ``model.train()`` to be chained.
        """
        super().train(mode)
        self.prior.eval()
        return self

    def encode_smiles(
        self,
        smiles: Sequence[str],
    ) -> Tensor:
        """Tokenize a batch of SMILES for model input.

        Special tokens are added and sequences are padded to the longest item
        in the batch. The method never truncates: dropping EOS would change the
        trajectory probability used by RTB. The result is moved to
        :attr:`device`. An empty input returns a ``(0, 0)`` tensor.

        Parameters
        ----------
        smiles : Sequence[str]
            SMILES strings to tokenize. The method does not canonicalize or
            validate them.

        Returns
        -------
        Tensor
            Integer token ids with shape ``(batch, sequence_length)``.

        Raises
        ------
        ValueError
            If the tokenizer returns non-matrix input ids.
        """
        if not smiles:
            return torch.empty((0, 0), dtype=torch.long, device=self.device)

        arguments: dict[str, Any] = {
            "add_special_tokens": True,
            "padding": True,
            "return_tensors": "pt",
        }
        encoded = self.tokenizer(list(smiles), **arguments)
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError("The tokenizer must return two-dimensional input_ids.")
        return input_ids.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        count: int,
        max_length: int,
        temperature: float = 1.0,
        **generation_kwargs: Any,
    ) -> GeneratedSequences:
        """Generate and decode a batch of SMILES with the current policy.

        Generation always samples and uses the tokenizer's pad and EOS token
        ids. Additional keyword arguments are passed to the Hugging Face
        ``generate`` method.

        Parameters
        ----------
        count : int
            Number of sequences to generate.
        max_length : int
            Maximum sequence length, including special tokens.
        temperature : float, optional
            Sampling temperature. Higher values produce more varied outputs.
        **generation_kwargs : Any
            Additional keyword arguments for ``generate``.

        Returns
        -------
        GeneratedSequences
            Generated token ids and their decoded SMILES, in matching order.

        Raises
        ------
        ValueError
            If ``count`` is negative, ``max_length`` is less than two, or
            ``temperature`` is not finite and positive.
        """
        if count < 0:
            raise ValueError("count must be nonnegative.")
        if max_length < 2:
            raise ValueError("max_length must be at least two.")
        if temperature <= 0.0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive.")
        if count == 0:
            return GeneratedSequences(
                input_ids=torch.empty((0, 0), dtype=torch.long, device=self.device),
                smiles=(),
            )

        generated_ids = self.policy.generate(
            do_sample=True,
            max_length=max_length,
            num_return_sequences=count,
            temperature=temperature,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            **generation_kwargs,
        )
        fidelity_indices = None
        if self.fidelity_head is not None:
            fidelity_indices = self.fidelity_head.sample(
                self._terminal_hidden_states(generated_ids),
                temperature=temperature,
            )
        return GeneratedSequences(
            input_ids=generated_ids,
            smiles=tuple(
                self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            ),
            fidelity_indices=fidelity_indices,
        )

    def policy_sequence_log_probabilities(self, input_ids: Tensor) -> Tensor:
        """Compute one sequence log probability per input under the policy.

        Padding is ignored and EOS is included. Gradients flow through the
        policy, so the result can be used for policy optimization.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids with shape ``(batch, sequence_length)``.

        Returns
        -------
        Tensor
            Sequence log probabilities with shape ``(batch,)``.

        Raises
        ------
        ValueError
            If ``input_ids`` does not have shape
            ``(batch, sequence_length)`` or the causal-model logits do not
            align with the shifted labels.
        TypeError
            If ``input_ids`` does not contain integer token ids.
        """
        return sequence_log_probabilities(
            causal_lm=self.policy,
            input_ids=input_ids.to(self.device),
            pad_token_id=self.pad_token_id,
        )

    def prior_sequence_log_probabilities(self, input_ids: Tensor) -> Tensor:
        """Compute one detached sequence log probability per input under the prior.

        Padding is ignored and EOS is included. The prior is evaluated without
        gradients because it is the fixed reference distribution for RTB.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids with shape ``(batch, sequence_length)``.

        Returns
        -------
        Tensor
            Detached sequence log probabilities with shape ``(batch,)``.

        Raises
        ------
        ValueError
            If ``input_ids`` does not have shape
            ``(batch, sequence_length)`` or the causal-model logits do not
            align with the shifted labels.
        TypeError
            If ``input_ids`` does not contain integer token ids.
        """
        with torch.no_grad():
            return sequence_log_probabilities(
                causal_lm=self.prior,
                input_ids=input_ids.to(self.device),
                pad_token_id=self.pad_token_id,
            ).detach()

    def policy_trajectory_log_probabilities(
        self,
        input_ids: Tensor,
        fidelity_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute policy log probability for a complete trajectory.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids with shape ``(batch, sequence_length)``.
        fidelity_indices : Tensor or None, optional
            Terminal fidelity-action indices aligned with ``input_ids``.

        Returns
        -------
        Tensor
            One complete-trajectory log probability per input row.

        Raises
        ------
        ValueError
            If fidelity indices are supplied without an action head, omitted
            for an action-aware model, or misaligned with the input batch.
        TypeError
            If fidelity indices are not integer-valued.
        """
        if self.fidelity_head is None:
            sequence_log_probabilities_ = self.policy_sequence_log_probabilities(
                input_ids
            )
            if fidelity_indices is not None:
                raise ValueError(
                    "Fidelity indices require a model with multiple fidelities."
                )
            return sequence_log_probabilities_
        if fidelity_indices is None:
            raise ValueError(
                "Fidelity indices are required for a multi-fidelity trajectory."
            )
        sequence_log_probabilities_, terminal_hidden_states = (
            self._policy_sequence_log_probabilities_and_terminal_hidden_states(
                input_ids
            )
        )
        return sequence_log_probabilities_ + self.fidelity_head.log_prob(
            terminal_hidden_states,
            fidelity_indices,
        )

    def prior_trajectory_log_probabilities(
        self,
        input_ids: Tensor,
        fidelity_indices: Tensor | None = None,
    ) -> Tensor:
        """Compute prior log probability for a complete trajectory.

        Parameters
        ----------
        input_ids : Tensor
            Integer token ids with shape ``(batch, sequence_length)``.
        fidelity_indices : Tensor or None, optional
            Terminal fidelity-action indices aligned with ``input_ids``.

        Returns
        -------
        Tensor
            One detached complete-trajectory log probability per input row.

        Raises
        ------
        ValueError
            If fidelity indices are supplied without an action head, omitted
            for an action-aware model, or misaligned with the input batch.
        TypeError
            If fidelity indices are not integer-valued.
        """
        sequence_log_probabilities_ = self.prior_sequence_log_probabilities(input_ids)
        if self.fidelity_head is None:
            if fidelity_indices is not None:
                raise ValueError(
                    "Fidelity indices require a model with multiple fidelities."
                )
            return sequence_log_probabilities_
        if fidelity_indices is None:
            raise ValueError(
                "Fidelity indices are required for a multi-fidelity trajectory."
            )
        return sequence_log_probabilities_ + self.fidelity_head.uniform_prior_log_prob(
            fidelity_indices,
            batch_size=input_ids.shape[0],
            device=sequence_log_probabilities_.device,
            dtype=sequence_log_probabilities_.dtype,
        )

    def _positive_rtb(
        self,
        positive_input_ids: Tensor,
        reward_scores: Tensor,
        beta: float,
        fidelity_indices: Tensor | None = None,
    ) -> Tensor:
        """Evaluate RTB for a non-empty batch of positive trajectories.

        The input scores are scaled scores ``r(x)``, not positive rewards. RTB
        therefore uses ``log R(x) = beta * reward_scores``. This method
        computes the policy and prior sequence log probabilities and then
        evaluates the mean-squared RTB residual.

        Parameters
        ----------
        positive_input_ids : Tensor
            Positive trajectory token ids with shape ``(batch, sequence_length)``.
        reward_scores : Tensor
            One finite scaled score ``r(x)`` per trajectory.
        beta : float
            Coefficient used to convert scores into ``log R(x)``.
        fidelity_indices : Tensor or None, optional
            Optional terminal fidelity-action indices aligned with the
            trajectories.

        Returns
        -------
        Tensor
            Scalar RTB loss.
        """
        positive_input_ids = positive_input_ids.to(self.device)
        reward_scores = reward_scores.to(
            device=self.device,
            dtype=self.log_z.dtype,
        ).reshape(-1)
        if positive_input_ids.ndim != 2:
            raise ValueError(
                "positive_input_ids must have shape (batch, sequence_length)."
            )
        if positive_input_ids.shape[0] != reward_scores.numel():
            raise ValueError("Positive trajectories and reward scores must align.")

        policy_log_probabilities = self.policy_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=fidelity_indices,
        )
        prior_log_probabilities = self.prior_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=fidelity_indices,
        )
        return relative_trajectory_balance_loss(
            policy_log_probabilities=policy_log_probabilities,
            prior_log_probabilities=prior_log_probabilities,
            reward_scores=reward_scores,
            log_z=self.log_z,
            beta=beta,
        )

    def on_policy_loss(
        self,
        positive_input_ids: Tensor,
        reward_scores: Tensor,
        beta: float,
        fidelity_indices: Tensor | None = None,
    ) -> Tensor | None:
        """Evaluate RTB for newly generated positive trajectories.

        This is the positive-only loss used before replay sampling. The method
        only evaluates the loss; it does not generate sequences or update
        parameters. It returns ``None`` for an empty batch.

        Parameters
        ----------
        positive_input_ids : Tensor
            Positive trajectory token ids with shape ``(batch, sequence_length)``.
        reward_scores : Tensor
            One finite scaled score ``r(x)`` per trajectory.
        beta : float
            Coefficient used to convert scores into
            ``log R(x) = beta * r(x)``.
        fidelity_indices : Tensor or None, optional
            Optional terminal fidelity-action indices aligned with the
            trajectories.

        Returns
        -------
        Tensor or None
            Scalar RTB loss, or ``None`` when the input batch is empty.

        Raises
        ------
        ValueError
            If the input ids do not have matrix shape or the reward scores
            and trajectories are not aligned.
        """
        if positive_input_ids.ndim != 2:
            raise ValueError(
                "positive_input_ids must have shape (batch, sequence_length)."
            )
        if positive_input_ids.shape[0] == 0:
            return None
        return self._positive_rtb(
            positive_input_ids,
            reward_scores,
            beta,
            fidelity_indices=fidelity_indices,
        )

    def replay_loss(
        self,
        positive_input_ids: Tensor,
        reward_scores: Tensor,
        beta: float,
        negative_input_ids: Tensor | None = None,
        aux_coefficient: float = 0.0,
        positive_fidelity_indices: Tensor | None = None,
        negative_fidelity_indices: Tensor | None = None,
    ) -> Tensor | None:
        """Evaluate RTB on replay data with an optional contrastive loss.

        The returned value is
        ``L_RTB + aux_coefficient * L_aux``. RTB uses the positive replay
        scores through ``log R(x) = beta * reward_scores``. If enabled, the
        auxiliary loss contrasts each positive sequence with the combined
        probability of the negative replay sequences. Negative sequences do
        not need reward scores. The method returns ``None`` for an empty
        positive batch.

        Parameters
        ----------
        positive_input_ids : Tensor
            Positive replay token ids with shape ``(batch, sequence_length)``.
        reward_scores : Tensor
            One finite scaled score ``r(x)`` per positive trajectory.
        beta : float
            Coefficient used to convert scores into ``log R(x)``.
        negative_input_ids : Tensor or None, optional
            Optional negative replay token ids with shape
            ``(batch, sequence_length)``. They are evaluated only when
            ``aux_coefficient`` is positive.
        aux_coefficient : float, optional
            Nonnegative weight for the contrastive loss. Zero disables
            negative replay evaluation.
        positive_fidelity_indices : Tensor or None, optional
            Optional terminal fidelity-action indices aligned with positive
            replay trajectories.
        negative_fidelity_indices : Tensor or None, optional
            Optional terminal fidelity-action indices aligned with negative
            replay trajectories.

        Returns
        -------
        Tensor or None
            Scalar replay loss, or ``None`` when the positive batch is empty.

        Raises
        ------
        ValueError
            If the auxiliary coefficient is invalid, input ids do not have
            matrix shape, or replay scores/actions are misaligned.
        TypeError
            If fidelity action indices are not integer-valued.
        """
        if not math.isfinite(aux_coefficient) or aux_coefficient < 0.0:
            raise ValueError("aux_coefficient must be finite and nonnegative.")
        self._last_auxiliary_loss = None
        if positive_input_ids.ndim != 2:
            raise ValueError(
                "positive_input_ids must have shape (batch, sequence_length)."
            )
        if positive_input_ids.shape[0] == 0:
            return None

        positive_input_ids = positive_input_ids.to(self.device)
        reward_scores = reward_scores.to(
            device=self.device,
            dtype=self.log_z.dtype,
        ).reshape(-1)
        if positive_input_ids.shape[0] != reward_scores.numel():
            raise ValueError("Positive trajectories and reward scores must align.")
        positive_log_probabilities = self.policy_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=positive_fidelity_indices,
        )
        prior_log_probabilities = self.prior_trajectory_log_probabilities(
            positive_input_ids,
            fidelity_indices=positive_fidelity_indices,
        )
        rtb_loss = relative_trajectory_balance_loss(
            policy_log_probabilities=positive_log_probabilities,
            prior_log_probabilities=prior_log_probabilities,
            reward_scores=reward_scores,
            log_z=self.log_z,
            beta=beta,
        )

        # ``_last_auxiliary_loss`` stays ``None`` unless the contrastive branch runs,
        # so reporting can distinguish "not computed" from a measured zero.
        total_loss = rtb_loss
        if aux_coefficient > 0.0 and negative_input_ids is not None:
            if negative_input_ids.ndim != 2:
                raise ValueError(
                    "negative_input_ids must have shape (batch, sequence_length)."
                )
            if negative_input_ids.shape[0] > 0:
                negative_log_probabilities = self.policy_trajectory_log_probabilities(
                    negative_input_ids,
                    fidelity_indices=negative_fidelity_indices,
                )
                auxiliary_loss = negative_replay_contrastive_loss(
                    positive_log_probabilities,
                    negative_log_probabilities,
                )
                self._last_auxiliary_loss = auxiliary_loss.detach()
                total_loss = rtb_loss + aux_coefficient * auxiliary_loss
        return total_loss

    def _select_terminal_hidden_states(
        self,
        outputs: Any,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return final-layer hidden states at each row's last non-padding token.

        Parameters
        ----------
        outputs : Any
            Causal language-model output carrying ``hidden_states``.
        input_ids : Tensor
            Integer tensor with shape ``(batch, sequence_length)``.
        attention_mask : Tensor
            Non-padding mask covering the full ``input_ids`` sequence.

        Returns
        -------
        Tensor
            Terminal hidden states with shape ``(batch, hidden_size)``.

        Raises
        ------
        ValueError
            If hidden states are missing, misaligned with ``input_ids``, or a
            row contains no non-padding token.
        """
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError(
                "The policy must return hidden states for terminal fidelity actions."
            )
        final_hidden_states = hidden_states[-1]
        if final_hidden_states.shape[:2] != input_ids.shape:
            raise ValueError(
                "The policy hidden states must align with input_ids for "
                "terminal fidelity actions."
            )
        if not torch.all(attention_mask.sum(dim=1) > 0):
            raise ValueError(
                "Terminal fidelity actions require at least one non-padding token."
            )
        terminal_positions = attention_mask.sum(dim=1) - 1
        batch_positions = torch.arange(
            input_ids.shape[0],
            device=input_ids.device,
        )
        return final_hidden_states[batch_positions, terminal_positions]

    def _terminal_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Return policy hidden states at the final non-padding tokens."""
        input_ids = input_ids.to(self.device)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return self._select_terminal_hidden_states(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _policy_sequence_log_probabilities_and_terminal_hidden_states(
        self,
        input_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute policy sequence probabilities and terminal states in one pass."""
        input_ids = input_ids.to(self.device)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence_length).")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids.")
        if input_ids.shape[1] < 2:
            raise ValueError("input_ids must contain at least two tokens.")

        labels = input_ids[:, 1:]
        attention_mask = input_ids.ne(self.pad_token_id).long()
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if outputs.logits.shape[:2] != input_ids.shape:
            raise ValueError(
                "The causal LM logits must align with the shifted sequence labels."
            )
        # The full sequence is scored in one pass so the terminal hidden state is
        # available; causal masking makes positions 0..L-2 independent of the last
        # token, so slicing here matches ``sequence_log_probabilities``.
        sequence_log_probabilities_ = sequence_log_probabilities_from_logits(
            outputs.logits[:, :-1],
            labels=labels,
            pad_token_id=self.pad_token_id,
        )
        terminal_hidden_states = self._select_terminal_hidden_states(
            outputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return sequence_log_probabilities_, terminal_hidden_states


def _model_hidden_size(model: nn.Module) -> int:
    """Return a causal model hidden size across common config names."""
    config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        hidden_size = getattr(config, name, None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size
    raise ValueError(
        "The policy configuration must expose hidden_size, n_embd, or d_model "
        "for terminal fidelity actions."
    )
