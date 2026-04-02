# References and Citation

This page collects the primary references for this repository.

## Primary reference

[*Multi-Fidelity Active Learning with GFlowNets*](http://arxiv.org/abs/2306.11715)
Alex Hernandez-Garcia, Nikita Saxena, Moksh Jain, Cheng-Hao Liu, and Yoshua Bengio.
Transactions on Machine Learning Research, 2024.

This work introduced the budget-constrained multi-fidelity active-learning workflow
that this repository builds on and extends.

## Talks and Slides

- **Talk (video)**: [Multi-Fidelity Active Learning with GFlowNets — TMLR presentation](https://www.dailymotion.com/video/k1k8KKYS67DgFCB516w) by Alex Hernandez-Garcia. Covers the motivation, methodology, and experimental results.

- **Slides**: [MF-AL GFlowNets — Presentation slides](https://alexhernandezgarcia.com/slides/mfgfn-tmlr) (Alex Hernandez-Garcia). The slide deck accompanying the TMLR presentation.

## Suggested citation

If you use this codebase in academic work, please cite:

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

- Original MF-AL GFlowNets code: <https://github.com/nikita-0209/mf-al-gfn>
- Earlier active-learning implementation: <https://github.com/alexhernandezgarcia/activelearning>
- GFlowNet library (planned sampler integration): <https://github.com/alexhernandezgarcia/gflownet>
- BoTorch, used by the surrogate and acquisition baseline: <https://github.com/pytorch/botorch>

## Benchmark context

The built-in Branin and Hartmann benchmark oracles provide standard
multi-fidelity reference problems. For a guided walkthrough, start with the
[Branin Experiment Tutorial](../tutorials/branin_experiment.md).
