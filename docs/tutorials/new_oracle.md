# Add a new Oracle

The complete guide is in the [Extension Guide → Adding an Oracle](../extension-guide/oracle.md). This page provides a concise implementation checklist.

## Implementation Checklist

- Extend `activelearning.oracle.oracle.Oracle`.
- Implement `get_fidelity_confidences()`, `get_costs()`, and `query()`.
- Register `<Name>Config` in `src/activelearning/oracle/config.py` and add it to `OracleConfig`.
- Keep returned costs and observations aligned with the input candidate order.
- Validate fidelities and make sure the sampler emits the same fidelity ids.
- Build runtime-aware tensors inside methods, not in `__init__()`.

If you only need to combine existing fidelities, prefer `CompositeOracle` over a brand-new oracle.

## References

- [Full guide](../extension-guide/oracle.md)
- [Oracle API](../api/oracle.md)
- [Runtime and configuration](../concepts/runtime_and_configuration.md)
