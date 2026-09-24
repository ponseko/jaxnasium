# Trainer & Agent API

Every algorithm consists of an *algorithm* (or *trainer*) and an *agent*. The trainer holds the hyperparameters
and the training logic; the agent holds the trainable state along with the trainer that produced it.

## RLAlgorithm

::: jaxnasium.algorithms.RLAlgorithm
    options:
        members:
            - train
            - init_agent
            - evaluate
            - load

## RLAgent

::: jaxnasium.algorithms.RLAgent
    options:
        members:
            - trainer
            - get_action
            - train
            - evaluate
            - with_hyperparams
            - replace
            - save
            - load

## Multi-agent method markers

These decorators declare how a method of an `RLAgent` behaves once the agent is
in multi-agent mode. See [Multi-Agent](Multi-Agent.md).

::: jaxnasium.algorithms._algorithm.per_agent

::: jaxnasium.algorithms._algorithm.collective
