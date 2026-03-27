# Add a new Selector

The complete guide is in the [Extension Guide → Adding a Selector](../extension-guide/selector.md). This page provides a concise implementation checklist.

## Implementation Checklist

- Extend `activelearning.selector.selector.Selector`.
- Implement `__call__(candidates, acquisition=None, cost_fn=None, round_budget=None)`.
- Register `<Name>Config` in `src/activelearning/selector/config.py` and add it to `SelectorConfig`.
- Return a subset of `Candidate` objects; do not spend budget or query the oracle here.
- Call `cost_fn(...)` yourself when the policy is cost-aware and preserve candidate fidelities.
- Return `[]` when nothing is feasible so the loop can stop cleanly.

If `TopKAcquisitionSelector`, `CostAwareSelector`, or `KnapsackSelector` already matches the policy, reuse it.

## References

- [Full guide](../extension-guide/selector.md)
- [Selector API](../api/selector.md)
- [Active learning loop](../concepts/active_learning_loop.md)
