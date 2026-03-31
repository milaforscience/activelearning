# GFlowNet Sampler Setup

The GFlowNet sampler trains a generative flow network as a proposal mechanism, guided by the current acquisition function as a reward signal. This page covers the config surface for `GFlowNetSampler` and `GFlowNetGridSampler`.

## What's in the repo

- `config/branin_gflownet_toy.yaml` — top-level activelearning config using
  `GFlowNetGridSampler` for acquisition-guided candidate sampling
- `config/gflownet/sampler/conf/*` — nested defaults for `agent`, `policy`, `proxy`, and `logger`
- `src/activelearning/sampler/gflownet/config_utils.py` — helper that composes those defaults and merges `sampler.conf` overrides
- `src/activelearning/sampler/gflownet/gflownet_sampler.py` — base GFlowNet sampler trained against acquisition-derived rewards
- `src/activelearning/sampler/gflownet/grid_sampler.py` — grid sampler with `output_bounds` rescaling
- `tests/sampler/test_gflownet_config.py` and `tests/test_gflownet_branin.py` — primary code-level references for composition and Branin wiring

## Choose the sampler flavor

| Sampler | Use it when... | Extra field |
| --- | --- | --- |
| `GFlowNetSampler` | your acquisition and objective oracle already work in the coordinates produced by the GFlowNet environment | none |
| `GFlowNetGridSampler` | the GFlowNet samples on a grid but the expensive objective oracle operates on a natural domain such as Branin bounds | `output_bounds` |

`GFlowNetGridSampler` is appropriate for Branin because the
underlying grid environment operates in `[cell_min, cell_max]^d`, while the
oracle expects `[-5, 10] x [0, 15]`.

## Minimal sampler block from `config/branin_gflownet_toy.yaml`

```yaml
sampler:
  type: GFlowNetGridSampler
  n_samples: 100
  output_bounds:
    - [-5.0, 10.0]
    - [0.0, 15.0]
  conf:
    env:
      n_dim: 2
      length: 50
      cell_min: 0.0
      cell_max: 1.0
    agent:
      optimizer:
        n_train_steps: 500
        lr: 5.0e-4
    proxy:
      reward_function: power
      reward_min: 1.0e-8
      reward_function_kwargs:
        beta: 1.0
```

This block is merged over the bundled GFlowNet defaults before the sampler is
constructed. It specifies the candidate-sampling side of the method: the GFlowNet
is trained on acquisition-derived rewards and proposes candidate points on a grid,
then rescales them into the oracle domain.

## What `sampler.conf` controls

| Key | What it changes | Repo default lives in |
| --- | --- | --- |
| `env` | grid dimensionality and resolution | bundled env config + local override |
| `agent` | optimizer, loss, train steps, random action rate | `config/gflownet/sampler/conf/agent/base.yaml` |
| `policy` | forward/backward policy network definitions | `config/gflownet/sampler/conf/policy/base.yaml` |
| `proxy` | how acquisition scores become GFlowNet rewards | `config/gflownet/sampler/conf/proxy/base.yaml` + `proxy/acquisition.yaml` |
| `logger` | GFlowNet-internal logging location and online/offline mode | `config/gflownet/sampler/conf/logger/base.yaml` |
| `state_flow` | optional state-flow model | composed sampler config |

The default proxy used by the activelearning integration is
`AcquisitionProxy`, so the GFlowNet is trained against the current acquisition
function rather than a separate standalone reward model.

## Override paths

These are the exact override paths accepted by the current config model:

```sh
uv run activelearning config/branin_gflownet_toy.yaml \
  sampler.n_samples=200 \
  sampler.conf.env.length=80 \
  sampler.conf.agent.optimizer.n_train_steps=1000 \
  sampler.conf.proxy.reward_function=power
```

The sampler inherits the top-level runtime when local `device` and `float_precision` fields are omitted; pin them locally if the GFlowNet stack requires different values.

## How `config/gflownet/branin.yaml` fits in

`config/gflownet/branin.yaml` is not a full `activelearning` CLI config. It is
better read as an upstream-style Hydra reference that shows the same nested
`sampler.conf` ideas at a lower level.

Use it to understand:

- the Branin-specific `env` override,
- example agent and proxy tweaks, and
- the naming of nested GFlowNet sections.

Use `config/branin_gflownet_toy.yaml` when you want the top-level activelearning shape.
