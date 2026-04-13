import pytest
from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choice import Choice
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.stack import Stack
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid

from activelearning.sampler.gflownet.multi_fidelity_env_wrapper import (
    MultiFidelityGFlowNetEnvWrapper,
)


@pytest.fixture
def env_grid2d():
    return Grid(n_dim=2, length=3)


@pytest.fixture
def env_cube3d():
    return ContinuousCube(n_dim=3)


@pytest.fixture
def env_choice():
    return Choice(n_options=5)


@pytest.fixture
def env_stack_cube_setgrids():
    """
    Stack of cube and set of grids, to test compositionality of meta environments.
    0: Cube
    1: SetFix
        0: Grid
        1: Grid
    """
    return Stack(
        subenvs=(
            ContinuousCube(n_dim=2),
            SetFix(
                subenvs=(
                    Grid(n_dim=2, length=3),
                    Grid(n_dim=2, length=3),
                )
            ),
        )
    )


@pytest.fixture
def env_stack_cube_setstacks():
    """
    Stack of set of stacks, to test compositionality of meta environments.
    0: Cube
    1: SetFix
        Stack
            0: Cube
            1: Grid
        Stack
            0: Grid
            1: Cube
    """
    return Stack(
        subenvs=(
            ContinuousCube(n_dim=2),
            SetFix(
                subenvs=(
                    Stack(
                        subenvs=(
                            ContinuousCube(n_dim=2),
                            Grid(n_dim=2, length=3),
                        )
                    ),
                    Stack(
                        subenvs=(
                            Grid(n_dim=2, length=3),
                            ContinuousCube(n_dim=2),
                        )
                    ),
                )
            ),
        )
    )


@pytest.fixture
def env_set_cubes():
    """Set of three 2D cubes."""
    return SetFix(
        subenvs=(
            ContinuousCube(n_dim=2),
            ContinuousCube(n_dim=2),
            ContinuousCube(n_dim=2),
        ),
    )


@pytest.mark.parametrize(
    "env",
    [
        "env_grid2d",
        "env_cube3d",
        "env_choice",
        "env_stack_cube_setgrids",
        "env_stack_cube_setstacks",
        "env_set_cubes",
    ],
)
def test__base_envs_initialize_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env_base",
    [
        "env_grid2d",
        "env_cube3d",
        "env_choice",
        "env_stack_cube_setgrids",
        "env_stack_cube_setstacks",
        "env_set_cubes",
    ],
)
@pytest.mark.parametrize(
    "n_fidelities",
    [
        1,
        2,
        3,
        5,
        10,
    ],
)
def test__env_wrapper_set_initializes_properly(env_base, n_fidelities, request):
    env_base = request.getfixturevalue(env_base)
    env = MultiFidelityGFlowNetEnvWrapper(env_base=env_base, n_fidelities=n_fidelities)
    assert isinstance(env, GFlowNetEnv)


@pytest.mark.parametrize(
    "env_base",
    [
        "env_grid2d",
        "env_cube3d",
        "env_choice",
        "env_stack_cube_setgrids",
        "env_stack_cube_setstacks",
        "env_set_cubes",
    ],
)
@pytest.mark.parametrize(
    "n_fidelities",
    [
        1,
        3,
    ],
)
def test__get_states_base_and_fidelities_returns_expected(
    env_base, n_fidelities, request
):
    env_base = request.getfixturevalue(env_base)
    env = MultiFidelityGFlowNetEnvWrapper(env_base=env_base, n_fidelities=n_fidelities)

    # Sample a batch of random states
    n_states = 10
    states = []
    states_base_expected = []
    fidelities_expected = []
    for _ in range(n_states):
        env.get_random_states(n_states=1)
        states.append(env.state)
        states_base_expected.append(env.env_base.state)
        fidelity = env.env_fidelity.state[0]
        assert fidelity in range(n_fidelities + 1)
        fidelities_expected.append(fidelity)

    # Retrieve base states and fidelities using env wrapper method
    states_base, fidelities = env.get_states_base_and_fidelities(states)

    # Check equality
    for state, state_exp in zip(states_base, states_base_expected):
        assert env.env_base.equal(state, state_exp)
    assert fidelities == fidelities_expected


@pytest.mark.parametrize(
    "env_base",
    [
        "env_grid2d",
        "env_cube3d",
        "env_choice",
        "env_stack_cube_setgrids",
        "env_stack_cube_setstacks",
        "env_set_cubes",
    ],
)
@pytest.mark.parametrize(
    "n_fidelities",
    [
        1,
        3,
    ],
)
def test__get_state_base_and_fidelity_returns_expected(env_base, n_fidelities, request):
    env_base = request.getfixturevalue(env_base)
    env = MultiFidelityGFlowNetEnvWrapper(env_base=env_base, n_fidelities=n_fidelities)

    n_states = 10
    for _ in range(n_states):
        env.get_random_states(n_states=1)
        state_base_expected = env.env_base.state
        fidelity_expected = env.env_fidelity.state[0]
        assert fidelity_expected in range(n_fidelities + 1)
        state_base, fidelity = env.get_state_base_and_fidelity(env.state)
        assert env.env_base.equal(state_base, state_base_expected)
        assert fidelity == fidelity_expected
