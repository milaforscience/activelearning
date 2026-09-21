# **S3-GFN Performance**

S3-GFN exposes performance controls for GPU-backed molecule experiments. The
controls affect policy execution, frozen-prior scoring, model precision, and
the independent batch sizes used during training and final candidate
generation.

## **Performance presets**

`sampler.performance_mode` selects a preset:

| Value | Policy execution | Model dtype | Attention-mask adapter | Prior scorer |
|-------|------------------|-------------|------------------------|--------------|
| `optimized` (default) | Compiled for training and final generation | `bfloat16` | Enabled | Compiled |
| `eager` | Eager execution | `float32` | Disabled | Eager |

The preset fills only low-level fields that are omitted from the YAML. Explicit
low-level values take precedence, so individual controls can be tuned without
losing the rest of a preset. To use the eager path:

```yaml
sampler:
  type: S3GFNSampler
  performance_mode: eager
```

The optimized preset is intended for supported CUDA hardware. Compilation is
lazy: the first training step includes TorchInductor warm-up. Use
`torch_compile_mode` and `torch_compile_dynamic` to control the
`torch.compile` invocation, or set `compile_strategy` to `none` or
`training_only` when final-generation compilation is not appropriate.

The low-level controls are:

| Field | Meaning |
|-------|---------|
| `compile_strategy` | `none`, `training_only`, or `training_and_generation`. |
| `torch_compile_mode` | TorchInductor mode passed to `torch.compile`. |
| `torch_compile_dynamic` | Dynamic-shape policy passed to `torch.compile`; `null` lets PyTorch choose. |
| `attention_mask_adapter` | Enables the GP-MoLFormer attention-mask adapter used by the compiled path. |
| `compile_prior_scorer` | Compiles frozen-prior sequence scoring as a separate no-gradient graph. |
| `model_dtype` | `float32` or `bfloat16` for S3-GFN model and loss tensors. |

## **Batch sizes and generation attempts**

The three batch-size controls have separate roles:

| Field | Role |
|-------|------|
| `batch_size` | Molecules generated for each on-policy training step. |
| `replay_batch_size` | Maximum positive and negative trajectories sampled for a replay update. |
| `generation_batch_size` | Molecules generated for each final candidate-generation attempt. |

When `generation_batch_size` is omitted, it inherits `batch_size`. Final
generation truncates its last batch to the remaining attempt budget, so
`max_generation_attempts` counts attempted samples strictly. If that limit is
omitted, it is derived from `n_samples` and the effective final-generation
batch size.

## **Reference benchmark**

The following measurements were reported for a production-path workload on an
NVIDIA A100-SXM4-40GB. Each variant used 100 measured training steps and 100
measured generation batches after warm-up.

| Phase | FP32 eager baseline | Optimized | Speedup |
|-------|---------------------|-----------|---------|
| Training | 28.08 trajectories/s (2.279 s/step) | 46.03 trajectories/s (1.390 s/step) | 1.64x |
| Final generation | 2,336 tokens/s at batch 64 | 3,549 tokens/s at batch 128 | 1.52x |

Peak allocated training memory was 30.58 GB for the baseline and 16.14 GB for
the optimized run. Optimized generation allocated 2.91 GB versus 3.20 GB for
the baseline while using twice the generation batch size. These measurements
are hardware- and workload-specific; benchmark the chosen settings on the
target GPU before increasing batch sizes.
