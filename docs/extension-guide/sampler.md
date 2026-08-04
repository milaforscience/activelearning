# **Adding a New Sampler**

Samplers generate the pool of candidates that the selector then trims to the
final query set. Implement a new sampler to:

- Propose candidates from a new search space geometry (graph, sequence, grid).
- Use an acquisition-guided or model-based proposal (MCMC, GFlowNet-style).
- Integrate an external candidate generator.
- Apply domain constraints that [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler) cannot express.

Before implementing a new sampler, verify that [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler),
[`PoolUniformSampler`](../reference/activelearning/sampler/pool_uniform_sampler/#activelearning.sampler.pool_uniform_sampler.PoolUniformSampler), or [`PoolScoreSampler`](../reference/activelearning/sampler/pool_score_sampler/#activelearning.sampler.pool_score_sampler.PoolScoreSampler) does not already cover your use case.

## **What to implement**

Subclass `activelearning.sampler.sampler.Sampler` and implement one method:

| Method | Required? | Notes |
|---|---|---|
| `sample(acquisition=None, observations=None)` | **Yes** | Returns `list[Candidate]` — the proposed candidate pool for the selector |

Both arguments are optional — your sampler may ignore either or both.

## **Reference implementations**

Review the built-in samplers as concrete examples before writing your own:

- [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler) — uniform sampling over a hypercube; the default sampler and simplest possible implementation.
- [`PoolScoreSampler`](../reference/activelearning/sampler/pool_score_sampler/#activelearning.sampler.pool_score_sampler.PoolScoreSampler) — scores a fixed pool with the acquisition function and returns the top-scoring candidates; a good example of acquisition-guided sampling.

Source: `src/activelearning/sampler/`.

## **Config model and registration**

Add a Pydantic config model in `src/activelearning/sampler/config.py` and extend the `SamplerConfig` union. See the existing models in that file as reference.

```python
class MySamplerConfig(BaseModel):
    type: Literal["MySampler"] = "MySampler"
    # your parameters here

    def build(self) -> Sampler:
        return MySampler(...)

SamplerConfig = Annotated[
    Union[..., MySamplerConfig],
    Field(discriminator="type"),
]
```

Then reference it in your YAML:

```yaml
sampler:
  type: MySampler
```

## **Fidelity handling**

The sampler is responsible for setting `Candidate.fidelity`. If your oracle
requires explicit fidelity ids (i.e. it is multi-fidelity), stamp each
[`Candidate`](../reference/activelearning/utils/types/#activelearning.utils.types.Candidate) before returning it:

```python
# uniform random assignment across two fidelity levels
import random
fidelity = random.choice([0, 1])
candidates.append(Candidate(x=x, fidelity=fidelity))
```

The convention is to keep fidelity ids consistent in config:
define the oracle-supported ids under `oracle.fidelity_costs` (or your oracle's
equivalent config) and configure the sampler to emit that same set (for
example via `sampler.fidelities` or custom sampler init/config fields). This
keeps `Candidate.fidelity` aligned with the ids declared by the oracle's
`get_fidelity_confidences()`; mismatches raise a `ValueError` at query time.
The bundled multi-fidelity configs follow this pattern (e.g.,
`config/branin_benchmark/base.yaml` composed with `config/branin_benchmark/mf_gfn.yaml`, and
`config/hartmann/multi_fidelity.yaml`).
See the [Oracle guide](oracle.md#fidelity-id-alignment-with-the-sampler) for
details.

For single-fidelity setups, single-fidelity is a special case of multi-fidelity
with exactly one declared level.  The oracle's `fidelity_costs` map has a single
entry (e.g. `{1: 1.0}`), the sampler emits that same level on every candidate,
and the surrogate operates in single-fidelity mode (no fidelity column appended
to inputs).  The top-level config validator automatically derives the fidelity
level from the oracle and fills in `sampler.fidelities` when it is omitted.

## **Using acquisition scores in sampling**

If your sampler uses the acquisition function to weight proposals, call
`acquisition.score()` inside `sample()`:

```python
def sample(self, acquisition=None, observations=None):
    candidates = self._generate_pool()
    if acquisition is not None and acquisition.supports_singleton_scoring:
        scores = acquisition.score(candidates)
        # use scores to weight selection from pool
        ...
    return candidates
```

Always guard with `acquisition is not None` — the sampler may be called before
the surrogate has been fitted.

**Cost weighting in multi-fidelity acquisitions.** Some acquisition classes
(notably BoTorch multi-fidelity acquisitions) accept a `cost_aware_utility`
that is wired into the acquisition object at `update()` time. When this is
configured, the scores returned by `acquisition.score()` already incorporate
labeling-cost penalties — meaning the sampler's score-guided proposals will
implicitly favour cheaper candidates. This is not something the sampler
controls; it is a consequence of calling `score()` on a cost-weighted
acquisition instance. See the
[Acquisition guide](acquisition.md#common-pitfalls) for a full explanation of
how this interacts with selector-side cost weighting.

## **Common pitfalls**

**Return [`Candidate`](../reference/activelearning/utils/types/#activelearning.utils.types.Candidate) objects, not tensors.** The selector and oracle expect
`Candidate` instances. Wrapping tensors as `Candidate.x` values is fine, but
the outer type must be `Candidate`.

**Do not query the oracle** inside the sampler. Sampling is purely generative;
evaluation happens in the oracle step.

**Avoid side effects from `observations`.** The observations iterable may be
a lazy generator. Never hold a reference to the generator and iterate it later.

## **Related pages**

- [Oracle guide](oracle.md) — aligning fidelity ids
- [Selector guide](selector.md) — what happens to the candidate pool after sampling
- [Extension guide overview](overview.md)
- [Sampler API](../reference/activelearning/sampler/sampler/)
