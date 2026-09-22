from collections import OrderedDict
from functools import partial

import pytest
import torch
from gflownet.envs.grid import Grid
from gflownet.utils.common import copy as gflownet_copy

import activelearning.sampler.gflownet.gflownet_sampler  # noqa: F401
import gflownet.utils.batch as gflownet_batch
from activelearning.sampler.gflownet.fast_copy import copy_state
from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    build_multi_fidelity_env_wrapper,
)


def _assert_same_and_independent(original, copied):
    """Check equal structure and values, with no shared mutable objects."""
    assert type(copied) is type(original)
    if torch.is_tensor(original):
        assert torch.equal(copied, original)
        assert copied.data_ptr() != original.data_ptr()
    elif isinstance(original, dict):
        assert copied is not original
        assert list(copied) == list(original)
        for key in original:
            _assert_same_and_independent(original[key], copied[key])
    elif isinstance(original, (list, tuple)):
        if isinstance(original, list):
            assert copied is not original
        assert len(copied) == len(original)
        for a, b in zip(original, copied):
            _assert_same_and_independent(a, b)
    else:
        assert copied == original


@pytest.mark.parametrize("fidelity_action", ["any", "first", "last"])
def test_copy_state_matches_gflownet_copy_on_wrapper_states(fidelity_action):
    env = build_multi_fidelity_env_wrapper(
        fidelity_action=fidelity_action,
        env_base_maker=partial(Grid, n_dim=2, length=3),
        n_fidelities=3,
    )
    torch.manual_seed(0)
    states = [env.source]
    for _ in range(3):
        env.reset()
        while not env.done:
            env.step_random()
            states.append(env.state)

    for state in states:
        reference = gflownet_copy(state)
        copied = copy_state(state)
        _assert_same_and_independent(state, copied)
        _assert_same_and_independent(reference, copied)


def test_copy_state_falls_back_to_deepcopy_for_subclasses():
    state = OrderedDict(a=[1, 2], b=torch.arange(3))
    copied = copy_state(state)
    assert type(copied) is OrderedDict
    _assert_same_and_independent(state, copied)


def test_importing_sampler_installs_fast_copy_in_batch():
    assert gflownet_batch.copy is copy_state
