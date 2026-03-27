# References and Citation

This page collects the references most directly tied to the repository's current paper-aligned documentation and benchmark path.

## Primary paper

[*Multi-Fidelity Active Learning with GFlowNets*](http://arxiv.org/abs/2306.11715)
Alex Hernandez-Garcia, Nikita Saxena, Moksh Jain, Cheng-Hao Liu, and Yoshua Bengio.
Transactions on Machine Learning Research, 2024.

This paper provides the theoretical foundation for the repository's budget-constrained multi-fidelity active-learning workflow.

## Talks and Slides

The following presentation materials accompany the paper:

- **Talk (video)**: [Multi-Fidelity Active Learning with GFlowNets — TMLR presentation](https://www.dailymotion.com/video/k1k8KKYS67DgFCB516w) by Alex Hernandez-Garcia. Covers the motivation, methodology, and experimental results of the MF-AL framework.

- **Slides**: [MF-AL GFlowNets — Presentation slides](https://alexhernandezgarcia.com/slides/mfgfn-tmlr) (Alex Hernandez-Garcia). The slide deck accompanying the TMLR paper presentation.

These materials provide context for the research framing and are recommended for understanding the paper prior to examining the implementation.

## Suggested citation

If you use this codebase in academic work, cite the original paper:

```bibtex
@article{hernandezgarcia2024multifidelity,
  title={Multi-Fidelity Active Learning with {GF}lowNets},
  author={Alex Hernandez-Garcia and Nikita Saxena and Moksh Jain and Cheng-Hao Liu and Yoshua Bengio},
  journal={Transactions on Machine Learning Research},
  year={2024},
  issn={2835-8856},
  url={https://openreview.net/forum?id=dLaazW9zuF},
  note={Expert Certification}
}
```

## Related software

- Original code accompanying the paper: <https://github.com/nikita-0209/mf-al-gfn>
- Intermediate active-learning implementation referenced by this repository: <https://github.com/alexhernandezgarcia/activelearning>
- GFlowNet dependency used by the sampler integration: <https://github.com/alexhernandezgarcia/gflownet>
- BoTorch, used by the current validated surrogate and acquisition baseline: <https://github.com/pytorch/botorch>

## Benchmark context

The built-in Branin and Hartmann benchmark oracles provide the current paper-aligned reference problems in this repository. For the clearest runnable path, start with [Branin Benchmark](../examples/branin_toy.md).
