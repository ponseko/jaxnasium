try:
    from jaxnasium.algorithms import core as core

    from ._algorithm import RLAgent as RLAgent, RLAlgorithm as RLAlgorithm
    from .agent_networks import (
        ActorNetwork as ActorNetwork,
        AdvantageNetwork as AdvantageNetwork,
        QValueNetwork as QValueNetwork,
        ValueNetwork as ValueNetwork,
    )
    from .dqn import DQN as DQN
    from .ppo import PPO as PPO
    from .pqn import PQN as PQN
    from .sac import SAC as SAC

except ImportError:
    raise ImportError(
        """Trying to import jaxnasium.algorithms without jaxnasium[algs] installed,
        please install it with pip install jaxnasium[algs]"""
    )
