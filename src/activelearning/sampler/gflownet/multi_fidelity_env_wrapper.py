from typing import Any, List, Sequence, Tuple

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choice import Choice
from gflownet.envs.composite.base import CompositeBase
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.stack import Stack


class MultiFidelityGFlowNetEnvWrapperBase(CompositeBase):
    """Common base environment for all the multi-fidelity environment wrappers"""

    def get_states_base_and_fidelities(
        self, states: Sequence
    ) -> Tuple[List[Any], List[int]]:
        """Retrieves the states of the base env and the fidelities of a batch of
        states.

        Parameters
        ----------
        states : Sequence
            A batch of states in environment format.

        Returns
        -------
        states_base : List[Any]
            The states of the base environment in the batch.
        fidelities : List[int]
            The fidelity index in each state of the batch.
        """
        states_base = []
        fidelities = []
        for state in states:
            states_base.append(self._get_substate(state, self.idx_base_env))
            fidelities.append(self._get_substate(state, self.idx_fidelity)[0])
        return states_base, fidelities

    def get_state_base_and_fidelity(self, state: Any = None) -> Tuple[Any, int]:
        """Retrieves the state of the base env and the fidelity of a state.

        This method constructs a dummy batch with the input state and calls
        ``get_states_base_and_fidelities()``; then returns the first element of the
        output.

        Parameters
        ----------
        states : Any or ``None``
            A state in environment format. If ``None``, ``self.state`` is used.

        Returns
        -------
        states_base : Any
            The state of the base environment.
        fidelities : int
            The fidelity index in the input state.
        """
        state = self._get_state(state)
        states_base, fidelities = self.get_states_base_and_fidelities([state])
        return states_base[0], fidelities[0]


class MultiFidelityGFlowNetEnvWrapper(SetFix, MultiFidelityGFlowNetEnvWrapperBase):
    """Turns any GFlowNet environment into a multi-fidelity environment.

    The wrapper is a SetFix GFlowNet environment whose sub-environments are the
    base environment and a Choice environment to model the discrete fidelity index.
    """

    def __init__(
        self,
        env_base: GFlowNetEnv,
        n_fidelities: int,
        **kwargs,
    ) -> None:
        self.env_base = env_base
        self.env_fidelity = Choice(n_options=n_fidelities)
        self.idx_base_env = 0
        self.idx_fidelity = 1
        super().__init__(subenvs=tuple([self.env_base, self.env_fidelity]), **kwargs)


class MultiFidelityGFlowNetEnvWrapperFidFirst(
    Stack, MultiFidelityGFlowNetEnvWrapperBase
):
    """Turns any GFlowNet environment into a multi-fidelity environment.

    The wrapper is a Stack GFlowNet environment whose sub-environments are:
        1. A Choice environment to model the discrete fidelity index.
        2. The base environment.

    Therefore, the fidelity is sampled first, followed by the actions of the base
    environment.
    """

    def __init__(
        self,
        env_base: GFlowNetEnv,
        n_fidelities: int,
        **kwargs,
    ) -> None:
        self.env_base = env_base
        self.env_fidelity = Choice(n_options=n_fidelities)
        self.idx_fidelity = 0
        self.idx_base_env = 1
        super().__init__(subenvs=tuple([self.env_fidelity, self.env_base]), **kwargs)


class MultiFidelityGFlowNetEnvWrapperFidLast(
    Stack, MultiFidelityGFlowNetEnvWrapperBase
):
    """Turns any GFlowNet environment into a multi-fidelity environment.

    The wrapper is a Stack GFlowNet environment whose sub-environments are:
        1. The base environment.
        2. A Choice environment to model the discrete fidelity index.

    Therefore, the actions of the base environment are sampled first, and the fidelity
    is sampled at the end of the trajectory.
    """

    def __init__(
        self,
        env_base: GFlowNetEnv,
        n_fidelities: int,
        **kwargs,
    ) -> None:
        self.env_base = env_base
        self.env_fidelity = Choice(n_options=n_fidelities)
        self.idx_base_env = 0
        self.idx_fidelity = 1
        super().__init__(subenvs=tuple([self.env_base, self.env_fidelity]), **kwargs)
