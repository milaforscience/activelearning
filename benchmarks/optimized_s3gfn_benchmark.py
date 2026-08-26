"""Benchmark the reference and experimental S3-GFN implementations.

The default benchmark uses a small HF-shaped causal model so it can run without
downloading GP-MoLFormer:

    .venv/bin/python benchmarks/optimized_s3gfn_benchmark.py

Use ``--real-model`` to load the cached or remote GP-MoLFormer checkpoint. CUDA
measurements synchronize before and after every timed section, and ``--json-output``
writes a structured report for experiment orchestration.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from activelearning.runtime import RuntimeContext
from activelearning.sampler.optimized_s3gfn.model import OptimizedS3GFNModel
from activelearning.sampler.optimized_s3gfn.sampler import OptimizedS3GFNSampler
from activelearning.sampler.s3gfn.fidelity import FidelityActionHead
from activelearning.sampler.s3gfn.model import GeneratedSequences, S3GFNModel
from activelearning.sampler.s3gfn.replay_buffer import ReplayBuffer
from activelearning.sampler.s3gfn.sampler import S3GFNSampler

_IMPLEMENTATIONS = ("reference", "optimized")
_TRAINING_AUX_COEFFICIENT = 1.0e-4


@dataclass
class _ModelTemplates:
    """CPU-side templates used to build identical benchmark models."""

    tokenizer: Any
    policy: nn.Module
    prior: nn.Module
    fidelity_head: FidelityActionHead


@dataclass
class _IterationSample:
    """One timed benchmark sample."""

    wall_ms: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    payload: Any = None


class _BenchmarkTokenizer:
    """Minimal right-padded tokenizer interface used by the model wrappers."""

    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"

    def __init__(self, recorder: _PhaseRecorder | None = None) -> None:
        self.recorder = recorder

    def batch_decode(
        self,
        input_ids: Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Return deterministic placeholder strings for generated rows."""
        del skip_special_tokens
        row_sums = input_ids.detach().sum(dim=1).cpu().tolist()
        smiles = ["C" * (1 + int(row_sum) % 16) for row_sum in row_sums]
        if self.recorder is not None:
            self.recorder.generated_smiles.extend(smiles)
        return smiles

    def __call__(
        self,
        smiles: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, Tensor]:
        """Encode placeholder SMILES into short, right-padded trajectories."""
        del add_special_tokens, padding
        if self.recorder is None:
            return self._encode(smiles, return_tensors=return_tensors)
        with self.recorder.phase("tokenization"):
            return self._encode(smiles, return_tensors=return_tensors)

    @staticmethod
    def _encode(
        smiles: list[str],
        *,
        return_tensors: str,
    ) -> dict[str, Tensor]:
        """Encode placeholder strings without external tokenizer state."""
        if return_tensors != "pt":
            raise ValueError("The benchmark tokenizer only supports PyTorch tensors.")
        rows = [
            torch.tensor(
                [1, 3 + (sum(ord(char) for char in smile) % 20), 2],
                dtype=torch.long,
            )
            for smile in smiles
        ]
        if not rows:
            return {"input_ids": torch.empty((0, 0), dtype=torch.long)}
        return {"input_ids": torch.stack(rows)}


class _TimedTokenizerProxy:
    """Wrap an arbitrary tokenizer so benchmark phases include tokenization."""

    def __init__(self, tokenizer: Any, recorder: _PhaseRecorder) -> None:
        self._tokenizer = tokenizer
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        """Decode strings while retaining them for benchmark quality metrics."""
        decoded = self._tokenizer.batch_decode(*args, **kwargs)
        self._recorder.generated_smiles.extend(str(value) for value in decoded)
        return decoded

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._recorder.phase("tokenization"):
            return self._tokenizer(*args, **kwargs)


class _PhaseRecorder:
    """Collect nested wall-clock timings and model-forward counts."""

    def __init__(
        self,
        device: torch.device,
        *,
        profile_timings: bool = False,
    ) -> None:
        self.device = device
        self.profile_timings = profile_timings
        self.times: dict[str, float] = {}
        self.forward_counts: dict[str, int] = {}
        self.forward_times: dict[str, float] = {}
        self.forward_phase_counts: dict[str, int] = {}
        self.forward_phase_times: dict[str, float] = {}
        self.generated_count = 0
        self.valid_count = 0
        self.synthesizable_count = 0
        self.online_update_performed = False
        self.replay_update_performed = False
        self.generated_smiles: list[str] = []
        self.valid_smiles: list[str] = []
        self.acquisition_scores: list[float] = []
        self.fidelity_values: list[int] = []
        self._active_phase = "unattributed"
        self._forward_starts: dict[int, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Record one phase while restoring any enclosing phase."""
        previous_phase = self._active_phase
        self._active_phase = name
        if not self.profile_timings:
            try:
                yield
            finally:
                self._active_phase = previous_phase
            return
        _synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            _synchronize(self.device)
            self.times[name] = self.times.get(name, 0.0) + (
                time.perf_counter() - started
            )
            self._active_phase = previous_phase

    def forward_pre_hook(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Start timing one policy or prior forward."""
        del args, kwargs
        if not self.profile_timings:
            return
        _synchronize(self.device)
        self._forward_starts[id(module)] = time.perf_counter()

    def forward_hook(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        outputs: object,
    ) -> None:
        """Finish timing one policy or prior forward."""
        del args, kwargs, outputs
        if not self.profile_timings:
            return
        _synchronize(self.device)
        module_name = (
            "prior" if getattr(module, "_benchmark_is_prior", False) else "policy"
        )
        self.forward_counts[module_name] = self.forward_counts.get(module_name, 0) + 1
        phase_key = f"{module_name}_forward/{self._active_phase}"
        self.forward_phase_counts[phase_key] = (
            self.forward_phase_counts.get(phase_key, 0) + 1
        )
        started = self._forward_starts.pop(id(module), None)
        if started is not None:
            duration = time.perf_counter() - started
            self.forward_times[module_name] = (
                self.forward_times.get(module_name, 0.0) + duration
            )
            self.forward_phase_times[phase_key] = (
                self.forward_phase_times.get(phase_key, 0.0) + duration
            )


class _BenchmarkChem:
    """Minimal molecule parser/canonicalizer with optional phase timing."""

    def __init__(self, recorder: _PhaseRecorder | None = None) -> None:
        self.recorder = recorder

    def MolFromSmiles(self, smiles: str) -> str:
        """Treat every nonempty benchmark string as a molecule."""
        if self.recorder is not None:
            with self.recorder.phase("rdkit"):
                return smiles
        return smiles

    def MolToSmiles(
        self,
        molecule: str,
        *,
        canonical: bool,
        isomericSmiles: bool,
    ) -> str:
        """Return the benchmark molecule unchanged."""
        del canonical, isomericSmiles
        if self.recorder is not None:
            with self.recorder.phase("rdkit"):
                return molecule
        return molecule


class _BenchmarkSynthesizability:
    """Accept every benchmark molecule while recording classification time."""

    def __init__(self, recorder: _PhaseRecorder) -> None:
        self.recorder = recorder

    def classify_batch(self, smiles: list[str]) -> list[bool]:
        """Return one positive classification for every string."""
        with self.recorder.phase("synthesizability"):
            self.recorder.valid_smiles.extend(smiles)
            return [True] * len(smiles)


class _BenchmarkAcquisition:
    """Singleton acquisition surrogate used by the train-step benchmark."""

    supports_singleton_scoring = True

    def __init__(self, recorder: _PhaseRecorder) -> None:
        self.recorder = recorder

    def score(self, candidates, cost_weighting=None) -> list[float]:
        """Return deterministic fidelity-dependent scores."""
        with self.recorder.phase("acquisition"):
            scores = [
                float(index + candidate.fidelity)
                for index, candidate in enumerate(candidates)
            ]
            if cost_weighting is not None:
                scores = cost_weighting(scores, candidates)
            self.recorder.acquisition_scores.extend(float(score) for score in scores)
            self.recorder.fidelity_values.extend(
                int(candidate.fidelity) for candidate in candidates
            )
            return scores


class _ProfileSampler(S3GFNSampler):
    """Reference sampler subclass that times each training-step boundary."""

    def __init__(
        self,
        recorder: _PhaseRecorder,
        *,
        batch_size: int,
        max_length: int,
        training_steps: int,
    ) -> None:
        super().__init__(
            n_samples=batch_size,
            fidelities=(1, 2, 3),
            max_length=max_length,
            batch_size=batch_size,
            replay_batch_size=batch_size,
            n_train_steps=training_steps,
            num_warmup_steps=0,
            aux_coefficient=_TRAINING_AUX_COEFFICIENT,
        )
        self.recorder = recorder

    def _prepare_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time canonicalization, scoring, tokenization, and classification."""
        with self.recorder.phase("preparation"):
            return super()._prepare_batch(*args, **kwargs)

    def _update_generated_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time the online replay insertion and optimizer update."""
        with self.recorder.phase("online_update"):
            return super()._update_generated_batch(*args, **kwargs)

    def _update_replay_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time replay sampling, scoring, and the replay optimizer update."""
        with self.recorder.phase("replay_update"):
            return super()._update_replay_batch(*args, **kwargs)

    def _optimize(self, *args: Any, **kwargs: Any) -> Any:
        """Time backward, gradient clipping, and the optimizer step."""
        with self.recorder.phase("backward_optimizer"):
            return super()._optimize(*args, **kwargs)


class _ProfileOptimizedSampler(OptimizedS3GFNSampler):
    """Optimized sampler subclass that records the same phase boundaries."""

    def __init__(
        self,
        recorder: _PhaseRecorder,
        optimized_options: Mapping[str, Any],
        *,
        batch_size: int,
        max_length: int,
        training_steps: int,
    ) -> None:
        """Initialize the profiling sampler with one ablation configuration."""
        super().__init__(
            n_samples=batch_size,
            fidelities=(1, 2, 3),
            max_length=max_length,
            batch_size=batch_size,
            replay_batch_size=batch_size,
            n_train_steps=training_steps,
            num_warmup_steps=0,
            aux_coefficient=_TRAINING_AUX_COEFFICIENT,
            prior_cache_enabled=bool(
                optimized_options.get("prior_cache_enabled", True)
            ),
            prior_cache_capacity=int(
                optimized_options.get("prior_cache_capacity", 8192)
            ),
            **{
                key: value
                for key, value in optimized_options.items()
                if key not in {"prior_cache_enabled", "prior_cache_capacity"}
            },
        )
        self.recorder = recorder

    def _prepare_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time canonicalization, scoring, tokenization, and classification."""
        with self.recorder.phase("preparation"):
            return super()._prepare_batch(*args, **kwargs)

    def _update_generated_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time online replay insertion and the first optimizer update."""
        with self.recorder.phase("online_update"):
            return super()._update_generated_batch(*args, **kwargs)

    def _update_replay_batch(self, *args: Any, **kwargs: Any) -> Any:
        """Time replay sampling and the second optimizer update."""
        with self.recorder.phase("replay_update"):
            return super()._update_replay_batch(*args, **kwargs)

    def _optimize(self, *args: Any, **kwargs: Any) -> Any:
        """Time backward, gradient clipping, and the optimizer step."""
        with self.recorder.phase("backward_optimizer"):
            return super()._optimize(*args, **kwargs)


@dataclass
class _TrainStepState:
    """Reusable state for one implementation's training-step benchmark."""

    sampler: _ProfileSampler | _ProfileOptimizedSampler
    positive_buffer: ReplayBuffer
    negative_buffer: ReplayBuffer | None
    optimizer: torch.optim.Optimizer


class _BenchmarkLanguageModel(nn.Module):
    """Small causal model with the tuple KV-cache shape used by HF models."""

    def __init__(self, hidden_size: int, vocabulary_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, bos_token_id=1)
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocabulary_size)
        self.vocabulary_size = vocabulary_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
        use_cache: bool | None = None,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        """Compute logits and append a single-layer tuple KV cache."""
        del attention_mask, use_cache, return_dict
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        positions = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
        )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = (
            hidden_states + positions.to(dtype=hidden_states.dtype)[None, :, None]
        )
        logits = self.output_projection(hidden_states)
        key = hidden_states.unsqueeze(1)
        value = key.clone()
        if past_key_values is not None:
            key = torch.cat((past_key_values[0][0], key), dim=2)
            value = torch.cat((past_key_values[0][1], value), dim=2)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            past_key_values=((key, value),),
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        do_sample: bool,
        max_length: int,
        num_return_sequences: int,
        temperature: float,
        pad_token_id: int,
        eos_token_id: int,
        **kwargs: object,
    ) -> Tensor:
        """Run a simple HF-compatible cached sampling loop."""
        del do_sample, kwargs
        sequences = torch.full(
            (num_return_sequences, 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=self.token_embedding.weight.device,
        )
        finished = torch.zeros(
            num_return_sequences,
            dtype=torch.bool,
            device=sequences.device,
        )
        past_key_values = None
        while sequences.shape[1] < max_length:
            outputs = self(
                input_ids=(sequences if past_key_values is None else sequences[:, -1:]),
                attention_mask=torch.ones_like(sequences),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            probabilities = torch.softmax(
                outputs.logits[:, -1] / temperature,
                dim=-1,
            )
            next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )
            sequences = torch.cat((sequences, next_tokens[:, None]), dim=1)
            finished |= next_tokens.eq(eos_token_id)
            if bool(finished.all()):
                break
        return sequences


def _load_fake_templates(
    *,
    hidden_size: int,
    vocabulary_size: int,
) -> _ModelTemplates:
    """Build small CPU-side model templates for local benchmarking."""
    policy = _BenchmarkLanguageModel(hidden_size, vocabulary_size)
    prior = _BenchmarkLanguageModel(hidden_size, vocabulary_size)
    tokenizer = _BenchmarkTokenizer()
    head = FidelityActionHead(hidden_size=hidden_size, n_fidelities=3)
    return _ModelTemplates(
        tokenizer=tokenizer,
        policy=policy,
        prior=prior,
        fidelity_head=head,
    )


def _load_real_templates(
    *,
    model_name: str,
    tokenizer_name: str,
    cache_dir: str | None,
) -> _ModelTemplates:
    """Load one CPU-side GP-MoLFormer template for repeated cloning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    load_kwargs = {
        "trust_remote_code": True,
        "deterministic_eval": True,
        "torch_dtype": torch.float32,
        "cache_dir": cache_dir,
    }
    policy = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    prior = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    hidden_size = getattr(policy.config, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("The real policy configuration must expose hidden_size.")
    head = FidelityActionHead(hidden_size=hidden_size, n_fidelities=3)
    return _ModelTemplates(
        tokenizer=tokenizer,
        policy=policy,
        prior=prior,
        fidelity_head=head,
    )


def _build_model(
    templates: _ModelTemplates,
    *,
    implementation: str,
    device: torch.device,
    optimized_options: Mapping[str, Any] | None = None,
) -> S3GFNModel | OptimizedS3GFNModel:
    """Clone one reference or optimized model onto the target device."""
    policy = deepcopy(templates.policy).to(device)
    prior = deepcopy(templates.prior).to(device)
    fidelity_head = deepcopy(templates.fidelity_head).to(device)
    if implementation == "reference":
        return S3GFNModel(
            policy=policy,
            prior=prior,
            tokenizer=templates.tokenizer,
            fidelity_head=fidelity_head,
        ).to(device)
    if implementation == "optimized":
        return OptimizedS3GFNModel(
            policy=policy,
            prior=prior,
            tokenizer=templates.tokenizer,
            fidelity_head=fidelity_head,
            **dict(optimized_options or {}),
        ).to(device)
    raise ValueError(f"Unsupported implementation: {implementation}")


def _build_models(
    *,
    device: torch.device,
    hidden_size: int,
    vocabulary_size: int,
) -> tuple[S3GFNModel, OptimizedS3GFNModel]:
    """Build reference and optimized wrappers with identical initial weights."""
    templates = _load_fake_templates(
        hidden_size=hidden_size,
        vocabulary_size=vocabulary_size,
    )
    reference = _build_model(templates, implementation="reference", device=device)
    optimized = _build_model(templates, implementation="optimized", device=device)
    return reference, optimized


def _build_real_models(
    *,
    device: torch.device,
    model_name: str,
    tokenizer_name: str,
    cache_dir: str | None = None,
) -> tuple[S3GFNModel, OptimizedS3GFNModel]:
    """Load reference and optimized GP-MoLFormer wrappers with shared weights."""
    templates = _load_real_templates(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
    )
    reference = _build_model(templates, implementation="reference", device=device)
    optimized = _build_model(templates, implementation="optimized", device=device)
    return reference, optimized


def _synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before measuring wall time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def _gpu_utilization_trace(path: str | None) -> Iterator[None]:
    """Record one ``nvidia-smi dmon`` stream while the benchmark runs."""
    if path is None:
        yield
        return
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("--gpu-utilization-trace requires nvidia-smi on the PATH.")
    output = open(path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "pucm", "-d", "1"],
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        output.close()


def _read_gpu_utilization_trace(path: str | None) -> dict[str, Any]:
    """Summarize SM and memory utilization from an ``nvidia-smi dmon`` file."""
    if path is None:
        return {
            "available": False,
            "samples": 0,
            "sm_average_percent": None,
            "memory_average_percent": None,
        }
    trace_path = Path(path)
    if not trace_path.exists():
        return {
            "available": False,
            "samples": 0,
            "sm_average_percent": None,
            "memory_average_percent": None,
        }
    header: list[str] | None = None
    sm_values: list[float] = []
    memory_values: list[float] = []
    for raw_line in trace_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        fields = raw_line.lstrip("#").split()
        lowered = [field.lower() for field in fields]
        if "sm" in lowered:
            header = lowered
            continue
        if header is None or not fields or not fields[0].lstrip("-").isdigit():
            continue
        for name, values in (("sm", sm_values), ("mem", memory_values)):
            if name not in header:
                continue
            index = header.index(name)
            if index >= len(fields):
                continue
            try:
                values.append(float(fields[index]))
            except ValueError:
                continue
    return {
        "available": bool(sm_values or memory_values),
        "samples": max(len(sm_values), len(memory_values)),
        "sm_average_percent": (None if not sm_values else statistics.fmean(sm_values)),
        "memory_average_percent": (
            None if not memory_values else statistics.fmean(memory_values)
        ),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a simple linear-interpolated percentile."""
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _summarize_samples(values: Sequence[float]) -> dict[str, float | int | None]:
    """Compute stable summary statistics for JSON serialization."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "cv": None,
        }
    data = [float(value) for value in values]
    median = statistics.median(data)
    stdev = statistics.stdev(data) if len(data) > 1 else 0.0
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": median,
        "p95": _percentile(data, 0.95),
        "min": min(data),
        "max": max(data),
        "stdev": stdev,
        "cv": None if median == 0.0 else stdev / median,
    }


def _summarize_optional_ints(
    values: Sequence[int | None],
) -> dict[str, float | int | None]:
    """Summarize optional integer samples when an accelerator exposes memory."""
    numeric_values = [int(value) for value in values if value is not None]
    if not numeric_values:
        return {
            "count": len(values),
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "stdev": None,
            "cv": None,
        }
    return _summarize_samples(numeric_values)


def _extract_memory_sample(device: torch.device) -> tuple[int | None, int | None]:
    """Return peak CUDA allocator counters for the finished iteration."""
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _measure_iterations(
    function: Callable[[int], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[_IterationSample]:
    """Run warmups, then collect timed and synchronized iteration samples."""
    for index in range(warmup):
        function(-(index + 1))
    _synchronize(device)
    samples: list[_IterationSample] = []
    for index in range(iterations):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        payload = function(index)
        _synchronize(device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated, peak_reserved = _extract_memory_sample(device)
        samples.append(
            _IterationSample(
                wall_ms=wall_ms,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                payload=payload,
            )
        )
    return samples


def _measure_iteration_blocks(
    function: Callable[[int], Any],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    block_size: int = 20,
) -> list[_IterationSample]:
    """Measure steady-state iteration time with synchronization between blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    for index in range(warmup):
        function(-(index + 1))
    _synchronize(device)

    samples: list[_IterationSample] = []
    completed = 0
    while completed < iterations:
        current_block_size = min(block_size, iterations - completed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        payloads = [
            function(completed + offset) for offset in range(current_block_size)
        ]
        _synchronize(device)
        block_wall_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated, peak_reserved = _extract_memory_sample(device)
        samples.append(
            _IterationSample(
                wall_ms=block_wall_ms / current_block_size,
                peak_allocated_bytes=peak_allocated,
                peak_reserved_bytes=peak_reserved,
                payload=payloads,
            )
        )
        completed += current_block_size
    return samples


def _summarize_timed_operation(
    samples: Sequence[_IterationSample],
    *,
    units_per_iteration: int | float,
) -> dict[str, Any]:
    """Summarize wall time, throughput, and memory for one benchmark section."""
    wall_ms_values = [sample.wall_ms for sample in samples]
    throughputs = [
        float(units_per_iteration) / (sample.wall_ms / 1000.0)
        for sample in samples
        if sample.wall_ms > 0.0
    ]
    return {
        "wall_ms": _summarize_samples(wall_ms_values),
        "throughput_per_s": _summarize_samples(throughputs),
        "peak_allocated_bytes": _summarize_optional_ints(
            [sample.peak_allocated_bytes for sample in samples]
        ),
        "peak_reserved_bytes": _summarize_optional_ints(
            [sample.peak_reserved_bytes for sample in samples]
        ),
    }


def _serialize_generated_sequences(generated: GeneratedSequences) -> dict[str, Any]:
    """Convert a generated sample into JSON-safe structures."""
    payload: dict[str, Any] = {
        "input_ids": generated.input_ids.detach().cpu().tolist(),
        "smiles": list(generated.smiles),
    }
    fidelity_indices = getattr(generated, "fidelity_indices", None)
    payload["fidelity_indices"] = (
        None if fidelity_indices is None else fidelity_indices.detach().cpu().tolist()
    )
    return payload


def _serialize_phase_recorder(recorder: _PhaseRecorder) -> dict[str, Any]:
    """Convert one train-step phase recorder into JSON-safe metrics."""
    return {
        "online_update_performed": recorder.online_update_performed,
        "replay_update_performed": recorder.replay_update_performed,
        "times_ms": {
            name: seconds * 1000.0 for name, seconds in recorder.times.items()
        },
        "forward_counts": dict(recorder.forward_counts),
        "forward_times_ms": {
            name: seconds * 1000.0 for name, seconds in recorder.forward_times.items()
        },
        "forward_phase_counts": dict(recorder.forward_phase_counts),
        "forward_phase_times_ms": {
            name: seconds * 1000.0
            for name, seconds in recorder.forward_phase_times.items()
        },
    }


def _summarize_quality_payloads(
    recorders: Sequence[_PhaseRecorder],
) -> dict[str, Any]:
    """Summarize task-level quality signals collected during training steps."""
    validity_rates: list[float] = []
    uniqueness_rates: list[float] = []
    synthesizable_rates: list[float] = []
    candidate_yields: list[float] = []
    acquisition_means: list[float] = []
    acquisition_maxima: list[float] = []
    fidelity_values: dict[int, list[float]] = {}
    for recorder in recorders:
        generated_count = len(recorder.generated_smiles)
        valid_count = len(recorder.valid_smiles)
        if generated_count > 0:
            validity_rates.append(valid_count / generated_count)
            candidate_yields.append(len(set(recorder.valid_smiles)) / generated_count)
        if valid_count > 0:
            uniqueness_rates.append(len(set(recorder.valid_smiles)) / valid_count)
            synthesizable_rates.append(recorder.synthesizable_count / valid_count)
        if recorder.acquisition_scores:
            acquisition_means.append(statistics.fmean(recorder.acquisition_scores))
            acquisition_maxima.append(max(recorder.acquisition_scores))
        total_fidelities = len(recorder.fidelity_values)
        if total_fidelities > 0:
            for fidelity in set(recorder.fidelity_values):
                fidelity_values.setdefault(fidelity, []).append(
                    recorder.fidelity_values.count(fidelity) / total_fidelities
                )
    return {
        "validity_rate": _summarize_samples(validity_rates),
        "uniqueness_rate": _summarize_samples(uniqueness_rates),
        "synthesizable_rate": _summarize_samples(synthesizable_rates),
        "candidate_yield": _summarize_samples(candidate_yields),
        "acquisition_mean": _summarize_samples(acquisition_means),
        "acquisition_max": _summarize_samples(acquisition_maxima),
        "fidelity_frequencies": {
            str(fidelity): _summarize_samples(values)
            for fidelity, values in sorted(fidelity_values.items())
        },
    }


def _summarize_recorder_payloads(recorders: Sequence[_PhaseRecorder]) -> dict[str, Any]:
    """Summarize per-step phase and forward metrics across measured samples."""
    phase_names = sorted({name for recorder in recorders for name in recorder.times})
    forward_count_names = sorted(
        {name for recorder in recorders for name in recorder.forward_counts}
    )
    forward_time_names = sorted(
        {name for recorder in recorders for name in recorder.forward_times}
    )
    forward_phase_count_names = sorted(
        {name for recorder in recorders for name in recorder.forward_phase_counts}
    )
    forward_phase_time_names = sorted(
        {name for recorder in recorders for name in recorder.forward_phase_times}
    )
    return {
        "raw_samples": [_serialize_phase_recorder(recorder) for recorder in recorders],
        "generated_count": _summarize_samples(
            [float(recorder.generated_count) for recorder in recorders]
        ),
        "valid_count": _summarize_samples(
            [float(recorder.valid_count) for recorder in recorders]
        ),
        "synthesizable_count": _summarize_samples(
            [float(recorder.synthesizable_count) for recorder in recorders]
        ),
        "online_update_rate": statistics.fmean(
            float(recorder.online_update_performed) for recorder in recorders
        ),
        "replay_update_rate": statistics.fmean(
            float(recorder.replay_update_performed) for recorder in recorders
        ),
        "phases_ms": {
            name: _summarize_samples(
                [recorder.times.get(name, 0.0) * 1000.0 for recorder in recorders]
            )
            for name in phase_names
        },
        "forward_counts": {
            name: _summarize_samples(
                [float(recorder.forward_counts.get(name, 0)) for recorder in recorders]
            )
            for name in forward_count_names
        },
        "forward_times_ms": {
            name: _summarize_samples(
                [
                    recorder.forward_times.get(name, 0.0) * 1000.0
                    for recorder in recorders
                ]
            )
            for name in forward_time_names
        },
        "forward_phase_counts": {
            name: _summarize_samples(
                [
                    float(recorder.forward_phase_counts.get(name, 0))
                    for recorder in recorders
                ]
            )
            for name in forward_phase_count_names
        },
        "forward_phase_times_ms": {
            name: _summarize_samples(
                [
                    recorder.forward_phase_times.get(name, 0.0) * 1000.0
                    for recorder in recorders
                ]
            )
            for name in forward_phase_time_names
        },
        "quality": _summarize_quality_payloads(recorders),
    }


def _cleanup_device_model(model: nn.Module, device: torch.device) -> None:
    """Release accelerator memory between implementation runs."""
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _create_train_step_state(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    training_steps: int,
    optimized_options: Mapping[str, Any] | None = None,
) -> _TrainStepState:
    """Create reusable train-step state for one implementation."""
    recorder = _PhaseRecorder(device)
    if isinstance(model, OptimizedS3GFNModel):
        sampler = _ProfileOptimizedSampler(
            recorder,
            optimized_options or _DEFAULT_OPTIMIZED_OPTIONS,
            batch_size=batch_size,
            max_length=max_length,
            training_steps=training_steps,
        )
        positive_buffer, negative_buffer = sampler._create_replay_buffers(
            pad_token_id=model.pad_token_id,
        )
    else:
        sampler = _ProfileSampler(
            recorder,
            batch_size=batch_size,
            max_length=max_length,
            training_steps=training_steps,
        )
        positive_buffer, negative_buffer = sampler._create_replay_buffers(
            pad_token_id=model.pad_token_id,
        )
    sampler.bind_runtime_context(
        RuntimeContext(
            device=device,
            dtype=next(model.policy.parameters()).dtype,
        )
    )
    policy_parameters = list(model.policy.parameters())
    if model.fidelity_head is not None:
        policy_parameters.extend(model.fidelity_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": sampler.learning_rate},
            {"params": [model.log_z], "lr": sampler.log_z_learning_rate},
        ]
    )
    return _TrainStepState(
        sampler=sampler,
        positive_buffer=positive_buffer,
        negative_buffer=negative_buffer,
        optimizer=optimizer,
    )


def _run_profiled_train_step(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    device: torch.device,
    state: _TrainStepState,
    seed: int,
    profile_timings: bool,
) -> _PhaseRecorder:
    """Run one actual training-mode step and return its phase recorder."""
    recorder = _PhaseRecorder(device, profile_timings=profile_timings)
    state.sampler.recorder = recorder
    original_tokenizer = model.tokenizer
    if isinstance(original_tokenizer, _BenchmarkTokenizer):
        original_tokenizer.recorder = recorder
    else:
        model.tokenizer = _TimedTokenizerProxy(original_tokenizer, recorder)
    model.policy.train()
    if model.fidelity_head is not None:
        model.fidelity_head.train()
    model.prior.eval()
    model.policy._benchmark_is_prior = False
    model.prior._benchmark_is_prior = True

    policy_pre_handle = None
    policy_post_handle = None
    prior_pre_handle = None
    prior_post_handle = None
    if profile_timings:
        policy_pre_handle = model.policy.register_forward_pre_hook(
            recorder.forward_pre_hook,
            with_kwargs=True,
        )
        policy_post_handle = model.policy.register_forward_hook(
            recorder.forward_hook,
            with_kwargs=True,
        )
        prior_pre_handle = model.prior.register_forward_pre_hook(
            recorder.forward_pre_hook,
            with_kwargs=True,
        )
        prior_post_handle = model.prior.register_forward_hook(
            recorder.forward_hook,
            with_kwargs=True,
        )
    original_generate = model.generate
    original_fidelity_sample = (
        None if model.fidelity_head is None else model.fidelity_head.sample
    )

    def timed_generate(*args: Any, **kwargs: Any) -> Any:
        with recorder.phase("generation"):
            return original_generate(*args, **kwargs)

    model.generate = timed_generate
    if original_fidelity_sample is not None:

        def timed_fidelity_sample(*args: Any, **kwargs: Any) -> Any:
            with recorder.phase("fidelity"):
                return original_fidelity_sample(*args, **kwargs)

        model.fidelity_head.sample = timed_fidelity_sample

    try:
        torch.manual_seed(seed)
        with recorder.phase("train_step"):
            step_result = state.sampler._train_step(
                model=model,
                synthesizability=_BenchmarkSynthesizability(recorder),
                positive_buffer=state.positive_buffer,
                negative_buffer=state.negative_buffer,
                molecule_chem=_BenchmarkChem(recorder),
                acquisition=_BenchmarkAcquisition(recorder),
                cost_fn=None,
                optimizer=state.optimizer,
            )
        recorder.generated_count = int(step_result[0])
        recorder.valid_count = int(step_result[1])
        recorder.synthesizable_count = int(step_result[2])
        recorder.online_update_performed = step_result[3] is not None
        recorder.replay_update_performed = step_result[4] is not None
    finally:
        model.generate = original_generate
        if original_fidelity_sample is not None:
            model.fidelity_head.sample = original_fidelity_sample
        if isinstance(original_tokenizer, _BenchmarkTokenizer):
            original_tokenizer.recorder = None
        else:
            model.tokenizer = original_tokenizer
        for handle in (
            policy_post_handle,
            policy_pre_handle,
            prior_post_handle,
            prior_pre_handle,
        ):
            if handle is not None:
                handle.remove()
    return recorder


def _benchmark_generation(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Time fidelity-aware generation for one implementation."""

    def generation_step(iteration_index: int) -> None:
        torch.manual_seed(seed + 1000 + max(iteration_index, 0))
        model.generate(batch_size, max_length)

    model.policy.eval()
    if model.fidelity_head is not None:
        model.fidelity_head.eval()
    model.prior.eval()
    samples = _measure_iterations(
        generation_step,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    summary = _summarize_timed_operation(samples, units_per_iteration=batch_size)
    summary["units"] = "sequences"
    return summary


def _benchmark_prior(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Time uncached prior scoring for one implementation."""
    torch.manual_seed(seed + 2000)
    input_batches = torch.randint(
        3,
        32,
        (warmup + iterations, batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    input_batches[:, :, -1] = 2

    def prior_step(iteration_index: int) -> None:
        batch_index = (
            iteration_index
            if iteration_index >= 0
            else iterations + abs(iteration_index) - 1
        )
        model.prior_sequence_log_probabilities(input_batches[batch_index])

    model.prior.eval()
    samples = _measure_iterations(
        prior_step,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    summary = _summarize_timed_operation(samples, units_per_iteration=batch_size)
    summary["units"] = "sequences"
    summary["cache_state"] = "cold"
    return summary


def _benchmark_prior_reuse(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Time repeated scoring of identical trajectories, including cache hits."""
    torch.manual_seed(seed + 2500)
    input_ids = torch.randint(
        3,
        32,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, -1] = 2

    def prior_step(iteration_index: int) -> None:
        del iteration_index
        model.prior_sequence_log_probabilities(input_ids)

    clear_cache = getattr(model, "clear_prior_cache", None)
    if callable(clear_cache):
        clear_cache()
    model.prior.eval()
    samples = _measure_iterations(
        prior_step,
        device=device,
        warmup=max(1, warmup),
        iterations=iterations,
    )
    summary = _summarize_timed_operation(samples, units_per_iteration=batch_size)
    summary["units"] = "sequences"
    summary["cache_state"] = "reuse"
    return summary


def _benchmark_policy_vectorization(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Compare one batched policy forward with one forward per sequence."""
    torch.manual_seed(seed + 3000)
    input_ids = torch.randint(
        3,
        32,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, -1] = model.eos_token_id
    model.policy.eval()

    def batched_step(iteration_index: int) -> None:
        del iteration_index
        with torch.no_grad():
            model.policy_sequence_log_probabilities(input_ids)

    def per_sequence_step(iteration_index: int) -> None:
        del iteration_index
        with torch.no_grad():
            for row in input_ids:
                model.policy_sequence_log_probabilities(row.unsqueeze(0))

    batched_samples = _measure_iterations(
        batched_step,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    per_sequence_samples = _measure_iterations(
        per_sequence_step,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    batched_summary = _summarize_timed_operation(
        batched_samples,
        units_per_iteration=batch_size,
    )
    per_sequence_summary = _summarize_timed_operation(
        per_sequence_samples,
        units_per_iteration=batch_size,
    )
    batched_median = batched_summary["wall_ms"]["median"]
    per_sequence_median = per_sequence_summary["wall_ms"]["median"]
    speedup = None
    if (
        isinstance(batched_median, float)
        and batched_median > 0.0
        and isinstance(per_sequence_median, float)
    ):
        speedup = per_sequence_median / batched_median
    return {
        "batched": batched_summary,
        "per_sequence": per_sequence_summary,
        "speedup_per_sequence_over_batched": speedup,
    }


def _benchmark_training_step(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    optimized_options: Mapping[str, Any] | None = None,
    warmup: int,
    iterations: int,
    seed: int,
    profile_timings: bool,
) -> dict[str, Any]:
    """Benchmark one actual training-mode optimization step."""
    state = _create_train_step_state(
        model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        training_steps=warmup + iterations,
        optimized_options=optimized_options,
    )

    def train_step(iteration_index: int) -> _PhaseRecorder:
        seed_offset = (
            iteration_index
            if iteration_index >= 0
            else iterations + abs(iteration_index)
        )
        return _run_profiled_train_step(
            model,
            device=device,
            state=state,
            seed=seed + 4000 + seed_offset,
            profile_timings=profile_timings,
        )

    optimized_sampler = (
        state.sampler if isinstance(state.sampler, OptimizedS3GFNSampler) else None
    )
    deferred_sync = bool(
        optimized_sampler is not None and optimized_sampler.deferred_sync
    )
    if deferred_sync:
        optimized_sampler._deferred_training_steps.clear()
        optimized_sampler._collect_deferred_metrics = True
    if optimized_sampler is not None:
        optimized_sampler._active_grad_scaler = optimized_sampler._make_grad_scaler()
        precision_scope = optimized_sampler._tf32_scope()
    else:
        precision_scope = nullcontext()
    try:
        with precision_scope:
            samples = _measure_iteration_blocks(
                train_step,
                device=device,
                warmup=warmup,
                iterations=iterations,
            )
    finally:
        if deferred_sync:
            optimized_sampler._flush_deferred_metrics()
            optimized_sampler._collect_deferred_metrics = False
        if optimized_sampler is not None:
            optimized_sampler._active_grad_scaler = None
    summary = _summarize_timed_operation(
        samples,
        units_per_iteration=batch_size,
    )
    summary["units"] = "trajectories"
    summary["steps_per_s"] = _summarize_samples(
        [1.0 / (sample.wall_ms / 1000.0) for sample in samples if sample.wall_ms > 0.0]
    )
    summary["batch_size"] = batch_size
    summary["replay_batch_size"] = batch_size
    summary["max_length"] = max_length
    summary["warmup_steps"] = warmup
    summary["measured_steps"] = iterations
    summary["measurement_block_size"] = min(20, iterations)
    summary["measurement_mode"] = (
        "intrusive_phase_profile"
        if profile_timings
        else "steady_state_synchronized_wall"
    )
    recorders = [recorder for sample in samples for recorder in sample.payload]
    phase_summary = _summarize_recorder_payloads(recorders)
    summary["quality"] = phase_summary.pop("quality")
    summary["phases"] = phase_summary
    summary["generated_trajectories_per_s"] = _summarize_samples(
        [
            sum(recorder.generated_count for recorder in sample.payload)
            / (sample.wall_ms * len(sample.payload) / 1000.0)
            for sample in samples
            if sample.wall_ms > 0.0 and sample.payload
        ]
    )
    summary["valid_trajectories_per_s"] = _summarize_samples(
        [
            sum(recorder.valid_count for recorder in sample.payload)
            / (sample.wall_ms * len(sample.payload) / 1000.0)
            for sample in samples
            if sample.wall_ms > 0.0 and sample.payload
        ]
    )
    summary["replay_update_rate"] = statistics.fmean(
        float(getattr(recorder, "replay_update_performed", False))
        for recorder in recorders
    )
    return summary


def _run_throughput_smoke_test(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    max_length: int,
    steps: int,
    device: torch.device,
    seed: int,
) -> float:
    """Run a short repeated workload to catch throughput regressions."""
    if steps <= 0:
        return 0.0
    model.policy.eval()
    if model.fidelity_head is not None:
        model.fidelity_head.eval()
    model.prior.eval()
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(steps):
            torch.manual_seed(seed + 5000 + step)
            generated = model.generate(batch_size, max_length)
            if generated.input_ids.shape[0] > 0:
                model.prior_sequence_log_probabilities(generated.input_ids)
    _synchronize(device)
    return steps / (time.perf_counter() - started)


def _collect_environment_metadata(device: torch.device) -> dict[str, Any]:
    """Capture lightweight environment metadata for benchmark reports."""
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        current_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        metadata["cuda"] = {
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": current_index,
            "device_name": torch.cuda.get_device_name(current_index),
        }
    return metadata


def _collect_correctness_sample(
    model: S3GFNModel | OptimizedS3GFNModel,
    *,
    batch_size: int,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    """Collect one deterministic generation sample for equivalence checks."""
    model.policy.eval()
    if model.fidelity_head is not None:
        model.fidelity_head.eval()
    model.prior.eval()
    torch.manual_seed(seed)
    generated = model.generate(batch_size, max_length)
    return _serialize_generated_sequences(generated)


def _compare_correctness(
    reference_sample: dict[str, Any] | None,
    optimized_sample: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare deterministic generation outputs between implementations."""
    if reference_sample is None or optimized_sample is None:
        return {
            "checked": False,
            "generation_matches": None,
            "fidelity_matches": None,
        }
    generation_matches = reference_sample["input_ids"] == optimized_sample["input_ids"]
    fidelity_matches = reference_sample.get("fidelity_indices") == optimized_sample.get(
        "fidelity_indices"
    )
    return {
        "checked": True,
        "generation_matches": generation_matches,
        "fidelity_matches": fidelity_matches,
    }


def _compute_speedup(reference_value: Any, candidate_value: Any) -> float | None:
    """Return a reference-over-candidate speedup when both medians exist."""
    if not isinstance(reference_value, (float, int)) or not isinstance(
        candidate_value, (float, int)
    ):
        return None
    reference_float = float(reference_value)
    candidate_float = float(candidate_value)
    if candidate_float <= 0.0:
        return None
    return reference_float / candidate_float


def _build_comparison(result: dict[str, Any]) -> dict[str, Any]:
    """Build structured reference-versus-optimized comparisons."""
    reference = result["implementations"].get("reference")
    optimized = result["implementations"].get("optimized")
    if reference is None or optimized is None:
        return {}
    comparison: dict[str, Any] = {}
    for section_name in ("generation", "prior", "prior_reuse", "training_step"):
        comparison[section_name] = {
            "speedup_reference_over_optimized": _compute_speedup(
                reference[section_name]["wall_ms"]["median"],
                optimized[section_name]["wall_ms"]["median"],
            )
        }
    comparison["policy_vectorization"] = {
        "reference_per_sequence_over_batched": reference["policy_vectorization"][
            "speedup_per_sequence_over_batched"
        ],
        "optimized_per_sequence_over_batched": optimized["policy_vectorization"][
            "speedup_per_sequence_over_batched"
        ],
    }
    return comparison


def _benchmark_optimized_options(args: argparse.Namespace) -> dict[str, Any]:
    """Read the optimized-model switches from a benchmark namespace."""
    return {
        "precision": getattr(args, "precision", "fp32"),
        "fixed_feature_maps": bool(getattr(args, "fixed_feature_maps", False)),
        "parallel_cuda_rollout": bool(getattr(args, "parallel_cuda_rollout", False)),
        "compile_mode": getattr(args, "compile_mode", "eager"),
        "deferred_sync": bool(getattr(args, "deferred_sync", False)),
        "carried_prior_scores": bool(getattr(args, "carried_prior_scores", False)),
        "overlap_online_prior": bool(getattr(args, "overlap_online_prior", False)),
        "combined_aux_policy_batch": bool(
            getattr(args, "combined_aux_policy_batch", False)
        ),
        "stop_check_interval": int(getattr(args, "stop_check_interval", 1)),
        "prior_cache_enabled": bool(getattr(args, "prior_cache_enabled", True)),
        "prior_cache_capacity": int(getattr(args, "prior_cache_capacity", 8192)),
    }


_DEFAULT_OPTIMIZED_OPTIONS = {
    "precision": "fp32",
    "fixed_feature_maps": False,
    "parallel_cuda_rollout": False,
    "compile_mode": "eager",
    "deferred_sync": False,
    "carried_prior_scores": False,
    "overlap_online_prior": False,
    "combined_aux_policy_batch": False,
    "stop_check_interval": 1,
    "prior_cache_enabled": True,
    "prior_cache_capacity": 8192,
}


def _benchmark_implementation(
    templates: _ModelTemplates,
    *,
    implementation: str,
    optimized_options: Mapping[str, Any] | None = None,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    max_length: int,
    warmup: int,
    iterations: int,
    smoke_steps: int,
    seed: int,
    profile_timings: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark one implementation and return metrics plus a correctness sample."""
    model = _build_model(
        templates,
        implementation=implementation,
        device=device,
        optimized_options=optimized_options,
    )
    try:
        correctness_sample = _collect_correctness_sample(
            model,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
        )
        result: dict[str, Any] = {
            "generation": _benchmark_generation(
                model,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                warmup=warmup,
                iterations=iterations,
                seed=seed,
            ),
            "prior": _benchmark_prior(
                model,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                warmup=warmup,
                iterations=iterations,
                seed=seed,
            ),
            "prior_reuse": _benchmark_prior_reuse(
                model,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                warmup=warmup,
                iterations=iterations,
                seed=seed,
            ),
            "policy_vectorization": _benchmark_policy_vectorization(
                model,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                warmup=warmup,
                iterations=iterations,
                seed=seed,
            ),
            "training_step": _benchmark_training_step(
                model,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                optimized_options=optimized_options,
                warmup=warmup,
                iterations=iterations,
                seed=seed,
                profile_timings=profile_timings,
            ),
        }
        if smoke_steps > 0:
            result["smoke_test"] = {
                "steps": smoke_steps,
                "steps_per_second": _run_throughput_smoke_test(
                    model,
                    batch_size=batch_size,
                    max_length=max_length,
                    steps=smoke_steps,
                    device=device,
                    seed=seed,
                ),
            }
        else:
            result["smoke_test"] = None
        result["prior_cache"] = {
            "hits": int(getattr(model, "prior_cache_hits", 0)),
            "misses": int(getattr(model, "prior_cache_misses", 0)),
            "enabled": bool(getattr(model, "prior_cache_enabled", False)),
        }
        result["optimization"] = {
            "precision": getattr(model, "precision", "fp32"),
            "precision_dtype": (
                None
                if getattr(model, "precision_dtype", None) is None
                else str(model.precision_dtype).replace("torch.", "")
            ),
            "fixed_feature_maps": bool(getattr(model, "fixed_feature_maps", False)),
            "feature_map_redraw_count": int(
                getattr(model, "feature_map_redraw_count", 0)
            ),
            "parallel_cuda_rollout": bool(
                getattr(model, "parallel_cuda_rollout", False)
            ),
            "compile_mode": getattr(model, "compile_mode", "eager"),
            "compile_cold_start_s": getattr(model, "compile_cold_start_s", None),
            "compile_graph_count": int(getattr(model, "compile_graph_count", 0)),
            "compile_graph_breaks": int(getattr(model, "compile_graph_breaks", 0)),
            "compile_recompilations": int(getattr(model, "compile_recompilations", 0)),
            "compile_fallback_reason": getattr(
                model,
                "compile_fallback_reason",
                None,
            ),
        }
        return result, correctness_sample
    finally:
        _cleanup_device_model(model, device)


def build_parser() -> argparse.ArgumentParser:
    """Create the benchmark CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda", "mps"),
    )
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument(
        "--implementation",
        choices=("reference", "optimized", "comparison"),
        default="comparison",
        help="Implementation to benchmark. The default preserves the old side-by-side CLI.",
    )
    parser.add_argument(
        "--model-name",
        default="ibm-research/GP-MoLFormer-Uniq",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="ibm-research/MoLFormer-XL-both-10pct",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocabulary-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "cuda_auto"),
        default="fp32",
        help="Optimized CUDA precision mode.",
    )
    parser.add_argument("--fixed-feature-maps", action="store_true")
    parser.add_argument("--parallel-cuda-rollout", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
            "max-autotune",
        ),
        default="eager",
    )
    parser.add_argument("--deferred-sync", action="store_true")
    parser.add_argument("--carried-prior-scores", action="store_true")
    parser.add_argument("--overlap-online-prior", action="store_true")
    parser.add_argument("--combined-aux-policy-batch", action="store_true")
    parser.add_argument("--stop-check-interval", type=int, default=1)
    parser.add_argument(
        "--no-prior-cache",
        dest="prior_cache_enabled",
        action="store_false",
    )
    parser.set_defaults(prior_cache_enabled=True)
    parser.add_argument("--prior-cache-capacity", type=int, default=8192)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Run this many generation/prior steps after measurements.",
    )
    parser.add_argument(
        "--profile-step",
        action="store_true",
        help=(
            "Enable intrusive synchronized phase timings. Disabled by default "
            "so primary wall-time measurements represent steady-state training."
        ),
    )
    parser.add_argument(
        "--gpu-utilization-trace",
        default=None,
        metavar="PATH",
        help="Save an nvidia-smi dmon trace while benchmark measurements run.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        metavar="PATH",
        help="Write the structured benchmark result as JSON.",
    )
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected benchmark workload and return a structured report."""
    device = torch.device(args.device)
    if args.gpu_utilization_trace is not None and device.type != "cuda":
        raise ValueError("--gpu-utilization-trace requires --device cuda.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requires CUDA availability.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requires MPS availability.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least two.")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least two.")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be nonnegative and --iterations positive.")

    torch.manual_seed(args.seed)

    templates = (
        _load_real_templates(
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            cache_dir=args.cache_dir,
        )
        if args.real_model
        else _load_fake_templates(
            hidden_size=args.hidden_size,
            vocabulary_size=args.vocabulary_size,
        )
    )
    implementations = (
        _IMPLEMENTATIONS
        if args.implementation == "comparison"
        else (args.implementation,)
    )
    result: dict[str, Any] = {
        "benchmark_version": 3,
        "implementation_mode": args.implementation,
        "config": {
            "real_model": bool(args.real_model),
            "device": str(device),
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "max_length": args.max_length,
            "hidden_size": args.hidden_size,
            "vocabulary_size": args.vocabulary_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "smoke_steps": args.smoke_steps,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "cache_dir": args.cache_dir,
            "ablations": _benchmark_optimized_options(args),
            "profile_timings": bool(args.profile_step),
        },
        "environment": _collect_environment_metadata(device),
        "implementations": {},
        "correctness_samples": {},
    }
    with _gpu_utilization_trace(args.gpu_utilization_trace):
        for implementation in implementations:
            benchmark_kwargs = {
                "implementation": implementation,
                "device": device,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "max_length": args.max_length,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "smoke_steps": args.smoke_steps,
                "seed": args.seed,
                "profile_timings": bool(args.profile_step),
            }
            optimized_options = _benchmark_optimized_options(args)
            if implementation == "optimized" and (
                optimized_options != _DEFAULT_OPTIMIZED_OPTIONS
            ):
                benchmark_kwargs["optimized_options"] = optimized_options
            implementation_result, correctness_sample = _benchmark_implementation(
                templates,
                **benchmark_kwargs,
            )
            result["implementations"][implementation] = implementation_result
            result["correctness_samples"][implementation] = correctness_sample
    result["gpu_utilization"] = _read_gpu_utilization_trace(args.gpu_utilization_trace)
    for implementation_result in result["implementations"].values():
        implementation_result["gpu_utilization"] = result["gpu_utilization"]
    if args.implementation == "optimized":
        reference_model = _build_model(
            templates,
            implementation="reference",
            device=device,
        )
        try:
            result["correctness_samples"]["reference"] = _collect_correctness_sample(
                reference_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                seed=args.seed,
            )
        finally:
            _cleanup_device_model(reference_model, device)
    result["quality"] = _compare_correctness(
        result["correctness_samples"].get("reference"),
        result["correctness_samples"].get("optimized"),
    )
    result["comparisons"] = _build_comparison(result)
    return result


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write one structured JSON artifact."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _print_phase_profile(name: str, training_step: dict[str, Any]) -> None:
    """Print detailed train-step phase and forward summaries."""
    print(f"profile[{name}]")
    for phase, summary in sorted(training_step["phases"]["phases_ms"].items()):
        print(f"  {phase}_median_ms_per_step={summary['median']:.3f}")
    for phase, summary in sorted(
        training_step["phases"]["forward_phase_counts"].items()
    ):
        forward_time = training_step["phases"]["forward_phase_times_ms"].get(phase, {})
        print(
            f"  {phase}_count_median={summary['median']:.1f} "
            f"{phase}_median_ms={forward_time.get('median', 0.0):.3f}"
        )
    for module_name, count_summary in sorted(
        training_step["phases"]["forward_counts"].items()
    ):
        module_time = training_step["phases"]["forward_times_ms"].get(module_name, {})
        print(
            f"  {module_name}_forwards_median={count_summary['median']:.1f} "
            f"{module_name}_forward_median_ms={module_time.get('median', 0.0):.3f}"
        )


def _print_cli_summary(result: dict[str, Any], *, profile_step: bool) -> None:
    """Print the legacy human-readable CLI summary."""
    config = result["config"]
    implementations = result["implementations"]
    print(f"device={config['device']} batch={config['batch_size']}")
    if "reference" in implementations and "optimized" in implementations:
        generation_reference = implementations["reference"]["generation"]["wall_ms"][
            "median"
        ]
        generation_optimized = implementations["optimized"]["generation"]["wall_ms"][
            "median"
        ]
        prior_reference = implementations["reference"]["prior"]["wall_ms"]["median"]
        prior_optimized = implementations["optimized"]["prior"]["wall_ms"]["median"]
        prior_reuse_reference = implementations["reference"]["prior_reuse"]["wall_ms"][
            "median"
        ]
        prior_reuse_optimized = implementations["optimized"]["prior_reuse"]["wall_ms"][
            "median"
        ]
        print(
            "generation_ms: "
            f"reference={generation_reference:.3f} "
            f"optimized={generation_optimized:.3f} "
            f"speedup={result['comparisons']['generation']['speedup_reference_over_optimized']:.2f}x"
        )
        print(
            "prior_ms: "
            f"reference={prior_reference:.3f} "
            f"optimized={prior_optimized:.3f} "
            f"speedup={result['comparisons']['prior']['speedup_reference_over_optimized']:.2f}x"
        )
        print(
            "prior_reuse_ms: "
            f"reference={prior_reuse_reference:.3f} "
            f"optimized={prior_reuse_optimized:.3f} "
            f"speedup={result['comparisons']['prior_reuse']['speedup_reference_over_optimized']:.2f}x"
        )
        print(
            "training_step_ms: "
            f"reference={implementations['reference']['training_step']['wall_ms']['median']:.3f} "
            f"optimized={implementations['optimized']['training_step']['wall_ms']['median']:.3f} "
            f"speedup={result['comparisons']['training_step']['speedup_reference_over_optimized']:.2f}x"
        )
        print(
            "policy_vectorization_ms: "
            f"batched={implementations['reference']['policy_vectorization']['batched']['wall_ms']['median']:.3f} "
            f"per_sequence={implementations['reference']['policy_vectorization']['per_sequence']['wall_ms']['median']:.3f} "
            f"speedup={implementations['reference']['policy_vectorization']['speedup_per_sequence_over_batched']:.2f}x"
        )
        optimized_smoke = implementations["optimized"].get("smoke_test")
        if optimized_smoke is not None:
            print(
                f"smoke_test: steps={optimized_smoke['steps']} "
                f"steps_per_second={optimized_smoke['steps_per_second']:.3f}"
            )
        print(
            "prior_cache: "
            f"hits={implementations['optimized']['prior_cache']['hits']} "
            f"misses={implementations['optimized']['prior_cache']['misses']}"
        )
        quality = result["quality"]
        if quality["checked"]:
            print(
                "correctness: "
                f"generation_match={quality['generation_matches']} "
                f"fidelity_match={quality['fidelity_matches']}"
            )
        if profile_step:
            _print_phase_profile(
                "reference", implementations["reference"]["training_step"]
            )
            _print_phase_profile(
                "optimized", implementations["optimized"]["training_step"]
            )
        return

    implementation_name = next(iter(implementations))
    implementation = implementations[implementation_name]
    print(
        f"{implementation_name}_generation_ms="
        f"{implementation['generation']['wall_ms']['median']:.3f}"
    )
    print(
        f"{implementation_name}_prior_ms="
        f"{implementation['prior']['wall_ms']['median']:.3f}"
    )
    print(
        f"{implementation_name}_prior_reuse_ms="
        f"{implementation['prior_reuse']['wall_ms']['median']:.3f}"
    )
    print(
        f"{implementation_name}_training_step_ms="
        f"{implementation['training_step']['wall_ms']['median']:.3f}"
    )
    print(
        f"{implementation_name}_policy_vectorization_ms: "
        f"batched={implementation['policy_vectorization']['batched']['wall_ms']['median']:.3f} "
        f"per_sequence={implementation['policy_vectorization']['per_sequence']['wall_ms']['median']:.3f} "
        f"speedup={implementation['policy_vectorization']['speedup_per_sequence_over_batched']:.2f}x"
    )
    smoke_test = implementation.get("smoke_test")
    if smoke_test is not None:
        print(
            f"{implementation_name}_smoke_test: steps={smoke_test['steps']} "
            f"steps_per_second={smoke_test['steps_per_second']:.3f}"
        )
    print(
        f"{implementation_name}_prior_cache: "
        f"hits={implementation['prior_cache']['hits']} "
        f"misses={implementation['prior_cache']['misses']}"
    )
    if profile_step:
        _print_phase_profile(implementation_name, implementation["training_step"])


def main() -> None:
    """Run correctness checks and print reference/optimized timings."""
    args = build_parser().parse_args()
    result = run_benchmark(args)
    if args.json_output is not None:
        _write_json(args.json_output, result)
    _print_cli_summary(result, profile_step=args.profile_step)


if __name__ == "__main__":
    main()
