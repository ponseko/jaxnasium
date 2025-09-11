try:
    # Weird import for proper copy from the CLI
    from jaxnasium.algorithms._algorithm import RLAlgorithm as RLAlgorithm  # noqa: I001

    from .networks import (
        ActorNetwork as ActorNetwork,
        QValueNetwork as QValueNetwork,
        ValueNetwork as ValueNetwork,
    )

    from ._dqn import DQN as DQN
    from ._ppo import PPO as PPO
    from ._pqn import PQN as PQN
    from ._sac import SAC as SAC

except ImportError:
    raise ImportError(
        """Trying to import jaxnasium.algorithms without jaxnasium[algs] installed,
        please install it with pip install jaxnasium[algs]"""
    )
