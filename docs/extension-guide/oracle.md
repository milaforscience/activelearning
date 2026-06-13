# **Adding a New Oracle**

An oracle implements the ground-truth evaluation function $f(x, m)$ for one or more supported fidelity levels $m \subseteq \mathcal{M}$. Implement a new oracle subclass to:

- Wrap a new simulator, benchmark function, or lab instrument.
- Define a different fidelity structure (different levels, costs, or confidences).
- Implement a custom routing rule for multi-fidelity queries.

A single oracle does not need to cover every fidelity level. The experiment's
top-level `oracle` configuration, however, must collectively support all fidelity
levels used during the run. You can achieve this with a single oracle that covers
all levels, or by using a `CompositeOracle` to combine several oracles that each
cover a subset. If you only need to **combine** existing oracles across fidelity
levels, see
[Using CompositeOracle](#using-compositeoracle-instead-of-building-from-scratch)
before writing a new class.

## **What to implement**

Subclass `activelearning.oracle.oracle.Oracle` and implement three methods:

| Method | Required? | Notes |
|---|---|---|
| `get_fidelity_confidences()` | **Yes** | Returns `dict[int, float]` — maps each fidelity id to a confidence in `[0, 1]` |
| `get_costs(candidates)` | **Yes** | Returns `list[float]` — one cost per input candidate, in the same order |
| `query(candidates)` | **Yes** | Returns `list[Observation]` — one [`Observation`](../reference/activelearning/utils/types/#activelearning.utils.types.Observation) per input candidate, in the same order |

## **Reference implementations**

Review the built-in oracles as concrete examples before writing your own:

- [`BraninOracle`](../reference/activelearning/oracle/augmented_function_oracle/#activelearning.oracle.augmented_function_oracle.BraninOracle) — 2D benchmark with configurable single and multi-fidelity costs.
- [`Hartmann6DOracle`](../reference/activelearning/oracle/augmented_function_oracle/#activelearning.oracle.augmented_function_oracle.Hartmann6DOracle) — 6-dimensional benchmark with the same structure.
- [`CompositeOracle`](../reference/activelearning/oracle/composite_oracle/#activelearning.oracle.composite_oracle.CompositeOracle) — routes queries across sub-oracles by fidelity level.

Source: `src/activelearning/oracle/augmented_function_oracle.py` and `src/activelearning/oracle/composite_oracle.py`.

## **Config model and registration**

Add a Pydantic config model in `src/activelearning/oracle/config.py` and extend the `OracleConfig` union. See the existing models in that file as reference.

```python
class MyOracleConfig(BaseModel):
    type: Literal["MyOracle"] = "MyOracle"
    # your parameters here

    def build(self) -> Oracle:
        return MyOracle(...)

OracleConfig = Annotated[
    Union[..., MyOracleConfig],
    Field(discriminator="type"),
]
```

Then specify it in your YAML:

```yaml
oracle:
  type: MyOracle
```

## **Fidelity id alignment with the sampler**

The fidelity ids used in `get_fidelity_confidences()` must match the ids your
sampler stamps onto `Candidate.fidelity`. If the sampler emits fidelity `0` or
`1` but your oracle only declares fidelity `2`, every query will raise a
`ValueError` (e.g., the [`HypercubeSampler`](../reference/activelearning/sampler/hypercube_sampler/#activelearning.sampler.hypercube_sampler.HypercubeSampler) `fidelities` parameter controls which ids
it assigns — keep them consistent).

## **Common pitfalls**

**Order alignment** — `get_costs()` and `query()` must return one item per
input candidate, in the exact same order. Never sort, group, or filter the
input list before building your return list.

**Budget** — `query()` must not check or modify the budget. Budget tracking
happens in the loop; your oracle only observes.

**Runtime tensors** — build tensors inside `query()`, not in `__init__()`.
Use `self.dtype` and `self.device` so the runtime binding takes effect.

## **Using CompositeOracle instead of building from scratch**

When different oracle implementations each handle a subset of fidelity levels, use [`CompositeOracle`](../reference/activelearning/oracle/composite_oracle/#activelearning.oracle.composite_oracle.CompositeOracle). It merges `get_fidelity_confidences()` from all
sub-oracles, routes candidates to the cheapest sub-oracle that handles each
fidelity, and assembles results in the original candidate order.

```yaml
oracle:
  type: CompositeOracle
  sub_oracles:
    - type: BraninOracle
      fidelity_costs: {0: 1.0}
    - type: BraninOracle
      fidelity_costs: {1: 5.0}
```

Use [`CompositeOracle`](../reference/activelearning/oracle/composite_oracle/#activelearning.oracle.composite_oracle.CompositeOracle) when fidelity levels are cleanly separated across
implementations. Write a new oracle class when the routing logic cannot be
expressed as independent sub-oracles.

## **Related pages**

- [Sampler guide](sampler.md) — aligning fidelity ids with the sampler
- [Extension guide overview](index.md)
- [Oracle API](../reference/activelearning/oracle/oracle/)
