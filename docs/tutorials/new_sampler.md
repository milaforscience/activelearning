# Add a new Sampler

The complete guide is in the [Extension Guide → Adding a Sampler](../extension-guide/sampler.md). This page provides a concise implementation checklist.

## Implementation Checklist

- Extend `activelearning.sampler.sampler.Sampler`.
- Implement `sample(acquisition=None, observations=None)`.
- Register `<Name>Config` in `src/activelearning/sampler/config.py` and add it to `SamplerConfig`.
- Ensure the config model's `build()` method accepts `runtime: RuntimeConfig | None = None`, even if the sampler does not use it.
- Return `Candidate` objects and stamp `Candidate.fidelity` when the oracle needs explicit fidelity ids.
- Materialize `observations` yourself if the implementation needs more than one pass.

If `HypercubeSampler`, `GFlowNetSampler`, or `GFlowNetGridSampler` already matches the proposal strategy, reuse it.

## References

- [Full guide](../extension-guide/sampler.md)
- [Sampler API](../api/sampler.md)
- [Runtime and configuration](../concepts/runtime_and_configuration.md)
