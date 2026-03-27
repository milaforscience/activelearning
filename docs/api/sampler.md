# Sampler API

Samplers generate proposal pools. In multi-fidelity runs, those proposals can be candidate-fidelity queries `(x, m)` rather than only candidate locations. Selectors consume those pools and choose the final queries.

## Modules at a glance

| Module | Main symbols | Role |
| --- | --- | --- |
| `activelearning.sampler.sampler` | `Sampler` | Abstract sampler interface. |
| `activelearning.sampler.hypercube_sampler` | `HypercubeSampler` | Random or LHS sampling in bounded boxes. |
| `activelearning.sampler.pool_uniform_sampler` | `PoolUniformSampler` | Uniform sampling from a fixed candidate pool. |
| `activelearning.sampler.pool_score_sampler` | `PoolScoreSampler` | Softmax-weighted sampling from a fixed candidate pool. |
| `activelearning.sampler.gflownet.gflownet_sampler` | `GFlowNetSampler` | GFlowNet-based candidate generation. |
| `activelearning.sampler.gflownet.grid_sampler` | `GFlowNetGridSampler` | GFlowNet sampler with optional coordinate rescaling. |
| `activelearning.sampler.gflownet.proxy` | `AcquisitionProxy` | Bridge from acquisition-like rewards to the GFlowNet proxy interface. |
| `activelearning.sampler.gflownet.config_utils` | `compose_gflownet_conf` | Composes bundled GFlowNet configs. |
| `activelearning.sampler.config` | sampler config models | YAML-facing builders. |

## Core abstraction: `Sampler`

Defined in `activelearning.sampler.sampler`.

| Method | Purpose | Notes |
| --- | --- | --- |
| `sample(acquisition=None, observations=None)` | Return a sequence of `Candidate` objects to be passed to a selector | Samplers may ignore `acquisition` and `observations`, or may use them for score-aware or stateful sampling. |

At the public API boundary, samplers produce `Candidate` objects rather than tensors or BoTorch-specific inputs, so fidelity assignment stays explicit when a sampler emits candidate-fidelity queries.

## Implementations

| Class | Module | Config-exposed | Needs acquisition? | Notes |
| --- | --- | --- | --- | --- |
| `HypercubeSampler` | `hypercube_sampler.py` | yes | no | Samples points inside user-provided bounds. |
| `GFlowNetSampler` | `gflownet/gflownet_sampler.py` | yes | yes | Trains a GFlowNet agent and samples terminating states each call. |
| `GFlowNetGridSampler` | `gflownet/grid_sampler.py` | yes | yes | Extends `GFlowNetSampler` with optional coordinate rescaling. |
| `PoolUniformSampler` | `pool_uniform_sampler.py` | no | no | Uniformly samples from a fixed pool without replacement. |
| `PoolScoreSampler` | `pool_score_sampler.py` | no | yes | Uses acquisition scores and a softmax distribution to sample from a fixed pool. |

## `HypercubeSampler`

`HypercubeSampler` is the primary YAML-facing sampler for bounded-box proposal generation.

| Parameter | Meaning |
| --- | --- |
| `bounds` | Per-dimension `(lower, upper)` pairs. |
| `num_samples` | Number of candidates returned per call. |
| `point_strategy` | `"uniform"` for i.i.d. draws or `"lhs"` for Latin hypercube sampling. |
| `fidelities` | `None`, a list of fidelity ids, or a `dict[int, float]` of fidelity costs. |

Fidelity handling is worth noting:

- `None` means single-fidelity candidates with `candidate.fidelity = None`
- a list means uniform sampling across the listed fidelity ids
- a dict means sampling **inversely proportional to cost**, so cheaper fidelities appear more often

`HypercubeSampler` is therefore the built-in sampler that most directly exposes paper-aligned candidate-fidelity query proposals.

## GFlowNet samplers

### `GFlowNetSampler`

`GFlowNetSampler` stores a Hydra and OmegaConf config tree and, on each `sample()` call:

1. instantiates an `AcquisitionProxy`
2. builds a GFlowNet agent
3. trains the agent
4. samples terminating states
5. converts those states to `Candidate` objects

`observations` are currently accepted for interface compatibility but not used by the implementation. The bundled GFlowNet samplers generate candidate locations; fidelity-specific query framing remains with the rest of the configured stack.

### `GFlowNetGridSampler`

`GFlowNetGridSampler` keeps the same training flow but remaps the sampled grid coordinates from the GFlowNet environment domain (`conf.env.cell_min` to `conf.env.cell_max`) into `output_bounds` when that argument is configured.

### `compose_gflownet_conf`

`activelearning.sampler.gflownet.config_utils.compose_gflownet_conf(...)` loads bundled YAML fragments from `config/gflownet/`, merges any overrides into a single `DictConfig`, and ensures a log directory exists.

## Configuration

Defined in `activelearning.sampler.config`.

| Config model | Builds | Notes |
| --- | --- | --- |
| `HypercubeSamplerConfig` | `HypercubeSampler(...)` | Direct parameter mapping. |
| `GFlowNetSamplerConfig` | `GFlowNetSampler(...)` | Resolves device and float precision from `RuntimeConfig` when those fields were not explicitly set. |
| `GFlowNetGridSamplerConfig` | `GFlowNetGridSampler(...)` | Same as above, plus `output_bounds`. |
| `SamplerConfig` | discriminated union | Currently only the three configs above. |

Config-time behavior for the GFlowNet builders:

- `build(runtime=cfg.runtime)` can inherit `runtime.device` and `runtime.precision`
- `conf` is a raw nested override dict that is merged over the bundled GFlowNet defaults
- `log_dir=None` creates a temporary log root automatically

## Interaction with acquisitions and selectors

- `PoolScoreSampler` requires an acquisition that supports singleton scoring.
- The current GFlowNet bridge is `AcquisitionProxy`, which adapts acquisition-like values to the proxy interface expected by the external GFlowNet library.
- The built-in selectors operate on the `Candidate` objects returned here; samplers do not choose final queries themselves.

If you want to add a new YAML-selectable sampling strategy, the extension points are a new `Sampler` subclass and a new branch in `activelearning.sampler.config.SamplerConfig`.

## Class Reference

::: activelearning.sampler.sampler.Sampler
    options:
      show_source: false
      heading_level: 3

::: activelearning.sampler.hypercube_sampler.HypercubeSampler
    options:
      show_source: false
      heading_level: 3

::: activelearning.sampler.pool_score_sampler.PoolScoreSampler
    options:
      show_source: false
      heading_level: 3

::: activelearning.sampler.pool_uniform_sampler.PoolUniformSampler
    options:
      show_source: false
      heading_level: 3
