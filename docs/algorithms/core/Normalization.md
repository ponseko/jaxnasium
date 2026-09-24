# Normalization

Algorithm-side observation and reward normalization, enabled with the
`normalize_observations` / `normalize_rewards` hyperparameters. Because the running
statistics live inside the agent, they are checkpointed and restored along with it. This is then typically preferred
over the environment wrappers ([`NormalizeVecObsWrapper`][jaxnasium.NormalizeVecObsWrapper] and
[`NormalizeVecRewardWrapper`][jaxnasium.NormalizeVecRewardWrapper]).

::: jaxnasium.algorithms.core.Normalizer
    options:
        members:
            - normalize_obs
            - normalize_reward
            - update
            - update_obs
            - update_reward

