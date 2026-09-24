from __future__ import annotations

import logging
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium.algorithms import RLAgent, RLAlgorithm
from jaxnasium.algorithms.core import (
    EpsilonGreedy,
    Normalizer,
    Schedule,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .agent_networks import QValueNetwork

logger = logging.getLogger(__name__)


class DQN(RLAlgorithm):
    """Deep Q-Network (DQN) algorithm implementation.

    This implementation uses target networks with soft updates (polyak averaging),
    a replay buffer and epsilon-greedy exploration with optional annealing.
    """

    learning_rate_start: float = 2.5e-4
    learning_rate_end: float | None = eqx.field(static=True, default=None)
    epsilon_start: float = 1.0
    epsilon_end: float | None = eqx.field(static=True, default=0.05)
    exploration_fraction: float = eqx.field(static=True, default=0.1)
    """ Fraction of `total_timesteps` over which epsilon anneals from start to end. """
    gamma: float = 0.99
    max_grad_norm: float | None = 10.0
    update_every: int = eqx.field(static=True, default=256)
    num_updates: int = eqx.field(static=True, default=8)
    replay_buffer_size: int = eqx.field(static=True, default=50_000)
    batch_size: int = eqx.field(static=True, default=256)
    warmup_steps: int = eqx.field(static=True, default=5_000)
    tau: float = 0.05

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=8)

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=False)

    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    def learning_rate_schedule(self) -> Schedule:
        return Schedule(
            self.learning_rate_start, self.learning_rate_end, self.num_training_updates
        )

    @property
    def epsilon_schedule(self) -> Schedule:
        """Annealed over `exploration_fraction` of the gradient-step budget."""
        return Schedule(
            self.epsilon_start,
            self.epsilon_end,
            max(1, int(self.exploration_fraction * self.num_training_updates)),
        )

    @property
    def optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm)
            if self.max_grad_norm is not None
            else optax.identity(),
            optax.adam(learning_rate=self.learning_rate_schedule, eps=1e-4),
        )

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.rollout_length // self.num_envs)

    @property
    def rollout_length(self):
        if self.update_every < self.num_envs:
            raise ValueError(
                f"`update_every` ({self.update_every}) must be >= `num_envs` ({self.num_envs})"
            )
        return int(self.update_every // self.num_envs)

    @property
    def num_training_updates(self):
        return self.num_iterations * self.num_updates

    @eqx.filter_jit
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> DQNAgent:
        return DQNAgent(key=key, env=env, trainer=self)

    @eqx.filter_jit
    def train(
        self, key: PRNGKeyArray, env: Environment, agent: DQNAgent | None = None
    ) -> tuple[DQNAgent, PyTree[Float[Array, " num_iterations"]]]:
        env = self.__check_env__(env, vectorized=True)

        if agent is None:
            agent = self.init_agent(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

        # Note that the randomness of the warmup data is dependent on the initial epsilon.
        warmup_length = max(2, max(self.warmup_steps, self.batch_size) // self.num_envs)
        warmup_state, dummy_trajectory = self._collect_rollout(
            agent, (env_state, obsv, key), env, length=warmup_length
        )
        buffer = TransitionBuffer(
            max_size=self.replay_buffer_size,
            sample_batch_size=self.batch_size,
            data_sample=dummy_trajectory,
        )
        buffer = buffer.insert(dummy_trajectory)  # Add minimum data to the buffer

        # Update the normalizer with the warmup data.
        agent = agent.update_normalizer(dummy_trajectory)

        train_iteration_fn = partial(self.train_iteration, env=env)
        train_iteration_fn = scan_callback(
            func=train_iteration_fn,
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
            reduce_ys_fn=self.reduce_metrics_fn,
        )

        runner_state = (agent, buffer, *warmup_state)
        runner_state, metrics = jax.lax.scan(
            train_iteration_fn, runner_state, jnp.arange(self.num_iterations)
        )
        updated_agent = runner_state[0]
        return updated_agent, metrics

    def train_iteration(self, runner_state, train_iter, *, env: Environment):
        """
        Performs a single training iteration (A single `Collect data + Update` run).

        Typically, the method is wrapped in a partial `train_fn = partial(train_iteration, env=env)`,
        and scanned over until the total number of timesteps is reached.
        """
        # Do rollout of single trajactory
        agent: DQNAgent = runner_state[0]
        buffer: TransitionBuffer = runner_state[1]
        rollout_state = runner_state[2:]
        (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
            agent, rollout_state, env
        )
        metric = trajectory_batch.info or {}

        # Add new data to buffer & Sample update batch from the buffer
        buffer = buffer.insert(trajectory_batch)
        agent = agent.update_normalizer(trajectory_batch)

        rng, update_key = jax.random.split(rng)
        agent = self._update_agent(update_key, agent, buffer)

        runner_state = (agent, buffer, env_state, last_obs, rng)
        return runner_state, metric

    def _update_agent(
        self, key: PRNGKeyArray, agent: DQNAgent, buffer: TransitionBuffer
    ) -> DQNAgent:
        """`num_updates` gradient steps, each on its own freshly sampled batch."""

        def scan_fn(carry, _):
            agent, rng = carry
            rng, sample_key = jax.random.split(rng)
            minibatch = buffer.sample(sample_key, with_next_obs=True)
            minibatch = minibatch.normalize(agent.normalizer)
            return (agent.update_params(minibatch), rng), None

        (updated_agent, _), _ = jax.lax.scan(
            scan_fn, (agent, key), None, length=self.num_updates
        )
        return updated_agent

    def _collect_rollout(
        self, agent: DQNAgent, rollout_state, env: Environment, length=None
    ):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            update_count = jym.tree.get_first(agent.optimizer_state, "count")
            get_action = partial(
                agent.get_action, epsilon=self.epsilon_schedule(update_count)
            )
            action = jax.vmap(get_action, in_axes=(0, 0))(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # Build a single transition. jax.lax.scan builds a batch of transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        if length is None:
            length = self.rollout_length

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, length
        )

        return rollout_state, trajectory_batch


class DQNAgent(RLAgent):
    trainer: DQN

    critic: QValueNetwork
    critic_target: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: DQN):
        self.trainer = trainer
        self.critic = QValueNetwork(
            env.observation_space,
            env.action_space,
            key=key,
            **trainer.critic_kwargs,
        )
        self.critic_target = jax.tree.map(lambda x: x, self.critic)

        self.optimizer_state = trainer.optimizer.init(
            eqx.filter(self.critic, eqx.is_inexact_array)
        )

        self.normalizer = Normalizer(
            obs_space=env.observation_space,
            normalize_obs=trainer.normalize_observations,
            normalize_rew=trainer.normalize_rewards,
            gamma=trainer.gamma,
            rew_shape=(trainer.num_envs,),
        )

    def get_action(
        self,
        key: PRNGKeyArray,
        observation: PyTree,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ):
        if deterministic:
            assert epsilon == 0.0, "Non-zero epsilon for deterministic action"
        observation = self.normalizer.normalize_obs(observation)
        q_values = self.critic(observation)
        action_mask = getattr(observation, "action_mask", None)
        action_dist = EpsilonGreedy(q_values, epsilon=epsilon, action_mask=action_mask)
        return action_dist.sample(seed=key)

    def get_value(self, observation: PyTree):
        observation = self.normalizer.normalize_obs(observation)
        return self.critic(observation)

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def update_normalizer(self, batch: Transition):
        return self.replace(normalizer=self.normalizer.update(batch))

    def update_params(self, batch: Transition):
        @eqx.filter_grad
        def __dqn_loss(params: QValueNetwork, train_batch: Transition):
            q_out_1 = jax.vmap(params)(train_batch.observation)
            q_taken = jym.tree.gather_actions(q_out_1, train_batch.action)
            q_taken = jym.tree.batch_mean(q_taken)
            q_loss = optax.losses.squared_error(q_taken, target)
            return jym.tree.mean(q_loss)

        trainer = self.trainer

        # Compute target
        q_target_output = jax.vmap(self.critic_target)(batch.next_observation)
        q_target_output = jym.tree.batch_mean(
            jax.tree.map(lambda q: jnp.max(q, axis=-1), q_target_output)
        )
        target = batch.reward + (1 - batch.terminated) * trainer.gamma * q_target_output

        grads = __dqn_loss(self.critic, batch)
        updates, optimizer_state = trainer.optimizer.update(
            grads, self.optimizer_state, eqx.filter(self.critic, eqx.is_inexact_array)
        )
        new_critic = eqx.apply_updates(self.critic, updates)

        # update target policy
        new_critic_target = jax.tree.map(
            lambda x, y: (1 - trainer.tau) * x + trainer.tau * y,
            self.critic_target,
            new_critic,
        )

        return self.replace(
            critic=new_critic,
            critic_target=new_critic_target,
            optimizer_state=optimizer_state,
        )
