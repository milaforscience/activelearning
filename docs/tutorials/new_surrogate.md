# Add a new Surrogate

The complete guide is in the [Extension Guide → Adding a Surrogate](../extension-guide/surrogate.md). This page provides a concise implementation checklist.

## Implementation Checklist

- Extend `activelearning.surrogate.surrogate.Surrogate`.
- Usually implement `updates_from_latest()`, `fit()`, `predict()`, and `is_fitted()`.
- Only add `update(...)` when incremental updates are genuinely supported.
- Register `<Name>Config` in `src/activelearning/surrogate/config.py` and add it to `SurrogateConfig`.
- Override `set_fidelity_confidences(...)` if the model uses oracle confidence metadata.
- Confirm that the chosen acquisition understands the surrogate interface you expose.

If `DummyMeanSurrogate` or `BoTorchGPSurrogate` already matches the modeling path, reuse it.

## References

- [Full guide](../extension-guide/surrogate.md)
- [Surrogate API](../api/surrogate.md)
- [Runtime and types](../api/runtime_and_types.md)
