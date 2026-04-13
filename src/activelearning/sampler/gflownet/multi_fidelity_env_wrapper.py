from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choice import Choice
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.stack import Stack


class MultiFidelityGFlowNetEnvWrapper(SetFix):
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
        self.fidelity_env = Choice(n_options=n_fidelities)
        super().__init__(subenvs=tuple([self.env_base, self.fidelity_env]), **kwargs)


class MultiFidelityGFlowNetEnvWrapperFidFirst(Stack):
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
        self.fidelity_env = Choice(n_options=n_fidelities)
        super().__init__(subenvs=tuple([self.fidelity_env, self.env_base]), **kwargs)


class MultiFidelityGFlowNetEnvWrapperFidLast(Stack):
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
        self.fidelity_env = Choice(n_options=n_fidelities)
        super().__init__(subenvs=tuple([self.env_base, self.fidelity_env]), **kwargs)
