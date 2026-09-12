# Active Learning Molecules

`activelearning-molecules` contains reusable molecular components for the
[`activelearning`](https://github.com/milaforscience/activelearning) framework.
It is an application package, not a campaign-specific project: the core
framework remains domain-neutral and this package owns molecular encoders,
tokenization, the xTB oracle, and S3-GFN.

## Installation

For an installed environment:

```sh
pip install activelearning-molecules
```

For this repository, install the complete workspace and development tools from
the repository root:

```sh
uv sync --all-packages --group dev
```

Molecular oracle runs also require the external
[xTB](https://xtb-docs.readthedocs.io/en/latest/) executable on `PATH`.
Python dependencies such as RDKit, SELFIES, MiniMol, and Transformers are
installed by this distribution; the xTB executable is not bundled.

Run molecular experiments through the application-owned command:

```sh
uv run activelearning-molecules applications/molecules/config/exact.yaml
```

The command imports the molecular catalogs once and supplies them to the core
configuration loader. No package declaration is needed in experiment YAML.

## Example configurations

The examples are ordinary YAML files intended to be copied and adapted in a
consuming project. Run them from the repository root with the paths below:

| File (relative to `applications/molecules/`) | Pipeline | Purpose |
| --- | --- | --- |
| `config/exact.yaml` | Pool + exact SELFIES DKL | Smallest pool-based baseline |
| `config/exact_multi_fidelity.yaml` | Pool + multi-fidelity DKL | Cost-aware pool search |
| `config/gflownet_exact.yaml` | SELFIES GFlowNet + exact DKL | Single-fidelity generative search |
| `config/gflownet_exact_multi_fidelity.yaml` | SELFIES GFlowNet + exact DKL | Joint molecule-fidelity search |
| `config/gflownet_variational_multi_fidelity.yaml` | SELFIES GFlowNet + variational DKL | Larger sparse-GP workflow |
| `config/s3gfn_exact.yaml` | S3-GFN + GP-MoLFormer | Single-fidelity SMILES search |
| `config/s3gfn_exact_multi_fidelity.yaml` | S3-GFN + GP-MoLFormer | Multi-fidelity SMILES search |
| `config/s3gfn_minimol_exact.yaml` | S3-GFN + MiniMol | Single-fidelity fingerprint search |
| `config/s3gfn_minimol_variational_multi_fidelity.yaml` | S3-GFN + MiniMol | Sparse-GP SMILES workflow |
| `config/s3gfn_minimol_fixed_variational_multi_fidelity.yaml` | S3-GFN + fixed MiniMol | Sparse-GP workflow without a trainable projection |

The pool-based examples use `config/data/molecules.txt`. If an example is
copied elsewhere, update `sampler.candidate_pool_file` to the new pool path.
The YAML location is not coupled to the installed Python package.

## Configuration ownership

There are two different meanings of “configuration”:

- Python `*Config` schemas belong to the distribution that implements their
  component. Molecular schemas live in each component subpackage, such as
  `activelearning_molecules.encoders.config`, and are aggregated by
  `activelearning_molecules.config_catalogs`.
- Experiment YAML belongs to the user. Keep it in a project-specific
  configuration directory, pass its path to `activelearning-molecules`, and use normal
  OmegaConf overrides. The framework does not require a global config directory
  or package-owned runtime files.

This keeps application schemas extensible without adding molecular imports or
application-owned schemas to the core package.

## Adding a molecular component

Implement the runtime component and its schema in this package, keeping heavy
imports inside `build()` or runtime methods where possible:

```python
from typing import Literal

from activelearning.config_registry import BuildableConfig


class MyMolecularSamplerConfig(BuildableConfig):
    type: Literal["MyMolecularSampler"] = "MyMolecularSampler"

    def build(self) -> object:
        from activelearning_molecules.samplers.my_sampler import MyMolecularSampler

        return MyMolecularSampler()


MOLECULE_SAMPLER_CONFIGS = (MyMolecularSamplerConfig,)
```

Add additional sampler schemas to this tuple. Aggregate the package's
categories in the conventional module
`activelearning_molecules.config_catalogs`:

```python
from activelearning_molecules.samplers.config import MOLECULE_SAMPLER_CONFIGS

CONFIG_CATALOGS = {
    "sampler": MOLECULE_SAMPLER_CONFIGS,
}
```

Do not edit core component registries or unions. The application command
composes this catalog with the core catalogs before parsing the experiment.
For a future ecosystem of independently developed third-party extensions,
package entry points would be a separate discovery mechanism; they are not
needed for this known application package.

## S3-GFN

S3-GFN is owned by this package because it generates molecular SMILES rather
than domain-neutral candidate coordinates. Its Python implementation is under
`activelearning_molecules.samplers.s3gfn`, while its stable telemetry
namespace remains `sampler/s3gfn/...` so existing dashboards describe the same
algorithm.

The sampler exposes independent controls for model execution dtype, training
and replay batch sizes, final generation batch size, and optional Torch
compilation. Start with `compile_strategy: none` when validating a new
environment, then enable compilation only after the uncompiled workflow is
working.

S3-GFN is based on the upstream implementation and its required attribution is
shipped in
[`THIRD_PARTY_NOTICES.md`](https://github.com/milaforscience/activelearning/blob/main/applications/molecules/src/activelearning_molecules/THIRD_PARTY_NOTICES.md).

## Development

From the repository root:

```sh
uv sync --all-packages --group dev
uv run pytest applications/molecules/tests
```

Use `make test-core` for the domain-neutral suite, `make test-molecules` for
this package, and `make test-all` for both.
