"""End-to-end tests for multi-file config loading and the active learning loop.

These tests verify that:
- ``load_config`` correctly merges multiple YAML files left to right.
- The merged config, when validated and built, produces a functional active
  learning loop that runs to completion.
- The CLI entry point correctly separates config file paths from key=value
  overrides when both are passed as positional arguments.
"""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from activelearning.config import ActiveLearningConfig
from activelearning.utils.config_loader import load_and_parse, load_config


def test_format_activelearning_command_quotes_shell_sensitive_arguments() -> None:
    """Reproduction commands must preserve paths and overrides with spaces."""
    from activelearning.main import _format_activelearning_command

    command = _format_activelearning_command(
        [Path("config with spaces.yaml")],
        ["run.name=experiment name"],
    )

    assert (
        command == "activelearning 'config with spaces.yaml' 'run.name=experiment name'"
    )


# ---------------------------------------------------------------------------
# Fixtures: minimal YAML fragments written to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config(tmp_path):
    """Base config with runtime, dataset, surrogate, and a tiny budget."""
    content = textwrap.dedent("""
        runtime:
          device: cpu
          precision: 64

        dataset:
          type: ListDataset

        surrogate:
          type: BoTorchGPSurrogate

        sampler:
          type: HypercubeSampler
          bounds:
            - [0.0, 1.0]
            - [0.0, 1.0]
          num_samples: 20
          fidelities: [1, 2]

        selector:
          type: CostAwareSelector

        oracle:
          type: BraninOracle
          fidelity_costs:
            1: 0.01
            2: 0.1

        budget:
          available_budget: 0.05
          schedule:
            type: constant
            value: 0.05
    """)
    path = tmp_path / "base.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def acquisition_config(tmp_path):
    """Acquisition block defined in a separate config file."""
    content = textwrap.dedent("""
        acquisition:
          type: DummyAcquisition
    """)
    path = tmp_path / "acquisition.yaml"
    path.write_text(content)
    return path


@pytest.fixture
def budget_override_config(tmp_path):
    """Override config that tightens the budget further."""
    content = textwrap.dedent("""
        budget:
          available_budget: 0.02
          schedule:
            type: constant
            value: 0.02
    """)
    path = tmp_path / "budget_override.yaml"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Unit tests: load_config merge behaviour
# ---------------------------------------------------------------------------


def test_load_config_two_file_merge_produces_complete_config(
    base_config, acquisition_config
):
    """Merging two YAML files must produce a config containing keys from both."""
    cfg = load_config([base_config, acquisition_config])
    assert "dataset" in cfg
    assert "surrogate" in cfg
    assert "budget" in cfg
    assert "acquisition" in cfg


def test_load_config_later_file_overrides_earlier(
    base_config, acquisition_config, budget_override_config
):
    """A later file must override shared keys from earlier files."""
    cfg = load_config([base_config, acquisition_config, budget_override_config])
    assert cfg.budget.available_budget == pytest.approx(0.02)


def test_load_config_earlier_file_not_overridden_by_missing_key(
    base_config, acquisition_config
):
    """Keys present only in an earlier file must survive the merge."""
    cfg = load_config([base_config, acquisition_config])
    # surrogate is only in base_config; acquisition_config does not touch it
    assert cfg.surrogate.type == "BoTorchGPSurrogate"


def test_load_config_dotlist_overrides_applied_last(base_config, acquisition_config):
    """Dotlist overrides passed to ``load_config`` must take precedence over all YAML files."""
    cfg = load_config(
        [base_config, acquisition_config],
        overrides=["budget.available_budget=0.001"],
    )
    assert cfg.budget.available_budget == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# End-to-end test: multi-file config → active learning loop
# ---------------------------------------------------------------------------


def test_active_learning_loop_runs_from_merged_configs(base_config, acquisition_config):
    """Active learning loop must run to completion when built from merged configs."""
    from activelearning.active_learning import active_learning
    from activelearning.runtime import bind_runtime_context

    cfg = load_and_parse([base_config, acquisition_config], ActiveLearningConfig)

    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    sampler = cfg.sampler.build()
    selector = cfg.selector.build()
    oracle = cfg.oracle.build()
    budget = cfg.budget.build()
    runtime_context = cfg.runtime.build(logger=None)

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle],
        runtime_context,
    )

    dataset_out, total_cost, num_rounds = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=runtime_context,
    )

    assert num_rounds >= 1, "Loop must complete at least one round."
    assert total_cost > 0.0, "At least one oracle query must have been executed."
    observations = list(dataset_out.get_observations_iterable())
    assert len(observations) > 0, "Dataset must contain observations after the loop."


def test_active_learning_loop_budget_override_reduces_rounds(
    base_config, acquisition_config, budget_override_config
):
    """Merging a tighter budget override must reduce total cost vs. the base config."""
    from activelearning.active_learning import active_learning
    from activelearning.runtime import bind_runtime_context

    def _run(paths):
        cfg = load_and_parse(paths, ActiveLearningConfig)
        dataset = cfg.dataset.build()
        surrogate = cfg.surrogate.build()
        acquisition = cfg.acquisition.build()
        sampler = cfg.sampler.build()
        selector = cfg.selector.build()
        oracle = cfg.oracle.build()
        budget = cfg.budget.build()
        runtime_context = cfg.runtime.build(logger=None)
        bind_runtime_context(
            [dataset, surrogate, acquisition, sampler, selector, oracle],
            runtime_context,
        )
        _, total_cost, _ = active_learning(
            dataset=dataset,
            surrogate=surrogate,
            acquisition=acquisition,
            sampler=sampler,
            selector=selector,
            oracle=oracle,
            budget=budget,
            runtime_context=runtime_context,
        )
        return total_cost

    cost_base = _run([base_config, acquisition_config])
    cost_override = _run([base_config, acquisition_config, budget_override_config])

    assert cost_override <= cost_base, (
        "Tighter budget override must not increase total cost."
    )


# ---------------------------------------------------------------------------
# CLI arg parsing: configs vs. key=value overrides
# ---------------------------------------------------------------------------


def test_cli_separates_overrides_from_config_paths(base_config, acquisition_config):
    """key=value tokens must be treated as overrides, not as config file paths.

    Regression test: with nargs='+' on a single positional, all tokens are
    consumed greedily. The splitting logic (args with '=' → overrides, rest →
    config paths) must ensure that dotlist overrides are applied correctly and
    that no override token is passed to OmegaConf.load() as a file path.
    """
    from activelearning.main import main

    override = "budget.available_budget=0.02"

    with patch.object(
        sys,
        "argv",
        ["activelearning", str(base_config), str(acquisition_config), override],
    ):
        # main() must complete without raising (e.g. FileNotFoundError on the
        # override token being treated as a config path, or a Pydantic
        # validation error from a missing acquisition block).
        main()


def test_cli_override_takes_effect_over_config_value(base_config, acquisition_config):
    """A key=value token in sys.argv must override the value set in the YAML file.

    Patches ``active_learning`` to capture the ``budget`` argument passed at
    runtime, then asserts its ``available_budget`` reflects the CLI override
    rather than the value defined in the YAML file.
    """
    from activelearning.main import main

    captured = {}

    def _capture_budget(*args, **kwargs):
        captured["budget"] = kwargs.get("budget") or args[5]
        captured["surrogate"] = kwargs.get("surrogate") or args[1]
        # Return the shape expected by main(): (dataset, total_cost, num_rounds)
        return kwargs.get("dataset") or args[0], 0.0, 0

    with patch.object(
        sys,
        "argv",
        [
            "activelearning",
            str(base_config),
            str(acquisition_config),
            "budget.available_budget=0.001",
        ],
    ):
        with patch(
            "activelearning.active_learning.active_learning",
            side_effect=_capture_budget,
        ):
            main()

    assert "budget" in captured, "active_learning was not called."
    assert captured["budget"].available_budget == pytest.approx(0.001), (
        "CLI override must take precedence over the value defined in the YAML config."
    )


def test_cli_raises_when_no_config_path_provided():
    """Passing only key=value tokens with no config path must raise ValueError."""
    from activelearning.main import main

    with patch.object(sys, "argv", ["activelearning", "budget.available_budget=1.0"]):
        with pytest.raises(ValueError, match="At least one config file path"):
            main()


def test_cli_unknown_flags_emit_warning_and_do_not_crash(
    base_config, acquisition_config
):
    """Unknown --flag arguments must emit a UserWarning and not raise SystemExit.

    Regression: parse_args() exits on unrecognised flags; parse_known_args()
    collects them so the process can continue and warn the user.
    """
    from activelearning.main import main

    with patch.object(
        sys,
        "argv",
        [
            "activelearning",
            str(base_config),
            str(acquisition_config),
            "--seed",
            "42",
        ],
    ):
        with pytest.warns(UserWarning, match="Unrecognised arguments ignored"):
            main()
