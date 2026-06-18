from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Sequence, Tuple

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choice import Choice
from gflownet.envs.composite.base import CompositeBase
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.stack import Stack

#: Mapping from ``fidelity_action`` string to the corresponding wrapper class.
#: Populated at the bottom of this module after all classes are defined.


class MultiFidelityGFlowNetEnvWrapperBase(CompositeBase, ABC):
    """Common base environment for all the multi-fidelity environment wrappers.

    Subclasses **must** define the following class-level attributes:

    Attributes
    ----------
    idx_base_env : int
        The sub-environment index corresponding to the base environment.
    idx_fidelity : int
        The sub-environment index corresponding to the fidelity environment.
    """

    @property
    @abstractmethod
    def idx_base_env(self) -> int:
        """Index of the base environment among the composite sub-environments."""
        ...

    @property
    @abstractmethod
    def idx_fidelity(self) -> int:
        """Index of the fidelity environment among the composite sub-environments."""
        ...

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
        state : Any or ``None``
            A state in environment format. If ``None``, ``self.state`` is used.

        Returns
        -------
        state_base : Any
            The state of the base environment.
        fidelity : int
            The fidelity index in the input state.
        """
        state = self._get_state(state)
        states_base, fidelities = self.get_states_base_and_fidelities([state])
        return states_base[0], fidelities[0]


class MultiFidelityGFlowNetEnvWrapper(SetFix, MultiFidelityGFlowNetEnvWrapperBase):
    """Turns any GFlowNet environment into a multi-fidelity environment.

    The wrapper is a SetFix GFlowNet environment whose sub-environments are the
    base environment and a Choice environment to model the discrete fidelity index.

    Attributes
    ----------
    env_base : GFlowNetEnv
        An instance of the base environment.
    env_fidelity : Choice
        An instance of a Choice environment to model the choice of fidelity index.
    idx_base_env : int
        The sub-environment index corresponding to the base environment. It is set to
        0, arbitrarily.
    idx_fidelity : int
        The sub-environment index corresponding to the fidelity environment. It is set
        to 1, arbitrarily.
    """

    idx_base_env = 0
    idx_fidelity = 1

    def __init__(
        self,
        env_base_maker: Callable[..., GFlowNetEnv],
        n_fidelities: int,
        **kwargs,
    ) -> None:
        """Initializes a MultiFidelityGFlowNetEnvWrapper instance.

        Parameters
        ----------
        env_base_maker : Callable[..., GFlowNetEnv]
            An environment maker (partial) to instantiate the base environment.
        n_fidelities : int
            The number of possible fidelity indices.
        """
        self.env_base = env_base_maker()
        self.env_fidelity = Choice(
            n_options=n_fidelities,
            float_precision=kwargs.get("float_precision", 32),
            device=kwargs.get("device", "cpu"),
        )
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

    Attributes
    ----------
    env_base : GFlowNetEnv
        An instance of the base environment.
    env_fidelity : Choice
        An instance of a Choice environment to model the choice of fidelity index.
    idx_fidelity : int
        The sub-environment index corresponding to the fidelity environment. It is set
        to 0 because the fidelity is the first sub-environment.
    idx_base_env : int
        The sub-environment index corresponding to the base environment. It is set to
        1 because the base environment is sampled only after the fidelity.
    """

    idx_fidelity = 0
    idx_base_env = 1

    def __init__(
        self,
        env_base_maker: Callable[..., GFlowNetEnv],
        n_fidelities: int,
        **kwargs,
    ) -> None:
        """Initializes a MultiFidelityGFlowNetEnvWrapperFidFirst instance.

        Parameters
        ----------
        env_base_maker : Callable[..., GFlowNetEnv]
            An environment maker (partial) to instantiate the base environment.
        n_fidelities : int
            The number of possible fidelity indices.
        """
        self.env_base = env_base_maker()
        self.env_fidelity = Choice(
            n_options=n_fidelities,
            float_precision=kwargs.get("float_precision", 32),
            device=kwargs.get("device", "cpu"),
        )
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

    Attributes
    ----------
    env_base : GFlowNetEnv
        An instance of the base environment.
    env_fidelity : Choice
        An instance of a Choice environment to model the choice of fidelity index.
    idx_base_env : int
        The sub-environment index corresponding to the base environment. It is set to
        0 because the base environment is sampled first, before the fidelity.
    idx_fidelity : int
        The sub-environment index corresponding to the fidelity environment. It is set
        to 1 because the fidelity is sampled only after the base environment.
    """

    idx_base_env = 0
    idx_fidelity = 1

    def __init__(
        self,
        env_base_maker: Callable[..., GFlowNetEnv],
        n_fidelities: int,
        **kwargs,
    ) -> None:
        """Initializes a MultiFidelityGFlowNetEnvWrapperFidLast instance.

        Parameters
        ----------
        env_base_maker : Callable[..., GFlowNetEnv]
            An environment maker (partial) to instantiate the base environment.
        n_fidelities : int
            The number of possible fidelity indices.
        """
        self.env_base = env_base_maker()
        self.env_fidelity = Choice(
            n_options=n_fidelities,
            float_precision=kwargs.get("float_precision", 32),
            device=kwargs.get("device", "cpu"),
        )
        super().__init__(subenvs=tuple([self.env_base, self.env_fidelity]), **kwargs)


# ---------------------------------------------------------------------------
# Populate the action → class mapping
# ---------------------------------------------------------------------------

_FIDELITY_ACTION_TO_WRAPPER: dict[str, type] = {
    "any": MultiFidelityGFlowNetEnvWrapper,
    "first": MultiFidelityGFlowNetEnvWrapperFidFirst,
    "last": MultiFidelityGFlowNetEnvWrapperFidLast,
}


def build_multi_fidelity_env_wrapper(
    fidelity_action: Literal["any", "first", "last"],
    env_base_maker: Callable[..., GFlowNetEnv],
    n_fidelities: int,
    **kwargs: Any,
) -> MultiFidelityGFlowNetEnvWrapperBase:
    """Instantiate the appropriate multi-fidelity wrapper for *fidelity_action*.

    Parameters
    ----------
    fidelity_action : {"any", "first", "last"}
        Controls *when* the fidelity choice is made during a trajectory:
        - ``"any"`` — :class:`MultiFidelityGFlowNetEnvWrapper` (SetFix): fidelity
          may be chosen at any point interleaved with base-env actions.
        - ``"first"`` — :class:`MultiFidelityGFlowNetEnvWrapperFidFirst` (Stack):
          fidelity is chosen before any base-env action.
        - ``"last"`` — :class:`MultiFidelityGFlowNetEnvWrapperFidLast` (Stack):
          fidelity is chosen after all base-env actions are complete.
    env_base_maker : Callable[..., GFlowNetEnv]
        Callable (partial) that constructs a fresh base environment instance.
    n_fidelities : int
        Number of possible fidelity choices.
    **kwargs
        Forwarded to the wrapper's ``__init__``.

    Returns
    -------
    MultiFidelityGFlowNetEnvWrapperBase
        A freshly constructed wrapper of the appropriate type.

    Raises
    ------
    ValueError
        If *fidelity_action* is not one of ``"any"``, ``"first"``, ``"last"``.
    """
    if fidelity_action not in _FIDELITY_ACTION_TO_WRAPPER:
        raise ValueError(
            f"Unknown fidelity_action {fidelity_action!r}. "
            f"Expected one of {sorted(_FIDELITY_ACTION_TO_WRAPPER)}."
        )
    wrapper_cls = _FIDELITY_ACTION_TO_WRAPPER[fidelity_action]
    return wrapper_cls(
        env_base_maker=env_base_maker, n_fidelities=n_fidelities, **kwargs
    )
