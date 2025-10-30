import logging
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAlgorithm
from jaxnasium.algorithms.utils import (
    DistraxContainer,
    Normalizer,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .networks import QValueNetwork

logger = logging.getLogger(__name__)


class DQNState(eqx.Module):
    critic: QValueNetwork
    critic_target: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer


class DQN(RLAlgorithm):
    """Deep Q-Network (DQN) algorithm implementation.

    This implementation uses target networks with soft updates (polyak averaging),
    a replay buffer and epsilon-greedy exploration with optional annealing.
    """

    state: DQNState = eqx.field(default=None)
    "State of the DQN algorithm, containing the networks, optimizer state and optional normalization running statistics."
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)
    "Optimizer used for training the networks."

    learning_rate: float = 2.5e-4
    anneal_learning_rate: bool | float = eqx.field(static=True, default=False)
    "Whether to anneal the learning rate over time. Set to a float to specify the end value. True means 0.0."
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    update_every: int = eqx.field(static=True, default=int(2e2))
    replay_buffer_size: int = int(1e4)
    batch_size: int = 64
    epsilon: float = 0.2
    anneal_epsilon: bool | float = eqx.field(static=True, default=True)
    "Whether to anneal the exploration rate over time. Set to a float to specify the end value. True means 0.0."
    tau: float = 0.95
    "Soft update coefficient for target network."

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=4)
    "Number of parallel environments."

    normalize_observations: bool = eqx.field(static=True, default=False)
    "Whether to normalize observations via running statistics."
    normalize_rewards: bool = eqx.field(static=True, default=False)
    "Whether to normalize rewards via running statistics."

    @property
    def _learning_rate_schedule(self):
        if self.anneal_learning_rate:
            end_value = (
                0.0 if self.anneal_learning_rate is True else self.anneal_learning_rate
            )
            return optax.linear_schedule(
                init_value=self.learning_rate,
                end_value=end_value,
                transition_steps=self.num_training_updates,
            )
        return optax.constant_schedule(self.learning_rate)

    @property
    def _epsilon_schedule(self) -> Callable[..., float]:
        if self.anneal_epsilon:
            end_value = 0.0 if self.anneal_epsilon is True else self.anneal_epsilon
            return optax.linear_schedule(
                init_value=self.epsilon,
                end_value=end_value,
                transition_steps=self.num_training_updates,
            )  # type: ignore
        return optax.constant_schedule(self.epsilon)  # type: ignore

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.update_every)

    @property
    def num_steps(self):  # rollout length
        return int(self.update_every // self.num_envs)

    @property
    def num_training_updates(self):
        return self.num_iterations  # * num_epochs

    @staticmethod
    def get_action(
        key: PRNGKeyArray,
        state: DQNState,
        observation: PyTree,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ):
        if deterministic:
            assert epsilon == 0.0, (
                "Epsilon set to non-zero value for deterministic action"
            )
        observation = state.normalizer.normalize_obs(observation)
        q_values = state.critic(observation)
        action_dist = DistraxContainer(
            jax.tree.map(lambda x: distrax.EpsilonGreedy(x, epsilon=epsilon), q_values)
        )
        return action_dist.sample(seed=key)

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "DQN":
        if getattr(env, "multi_agent", False) and self.auto_upgrade_multi_agent:
            self = self.__make_multi_agent__()

        if self.optimizer is None:
            self = replace(
                self,
                optimizer=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adabelief(learning_rate=self._learning_rate_schedule),
                ),
            )

        agent_states = self._make_agent_state(
            key=key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            critic_kwargs=self.critic_kwargs,
        )

        return replace(self, state=agent_states)

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "DQN":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            """
            Performs a single training iteration (A single `Collect data + Update` run).
            This is repeated until the total number of timesteps is reached.
            """

            # Do rollout of single trajactory
            self: DQN = runner_state[0]
            buffer: TransitionBuffer = runner_state[1]
            rollout_state = runner_state[2:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )
            metric = trajectory_batch.info or {}

            # Post-process the trajectory batch: normalization update (possibly per-agent)
            updated_state = self._postprocess_rollout(trajectory_batch, self.state)

            # Add new data to buffer & Sample update batch from the buffer
            buffer = buffer.insert(trajectory_batch)
            train_data = buffer.sample(rng)

            # Update
            updated_state = self._update_agent_state(
                rng,
                updated_state,  # <-- use updated_state w/ updated norm
                train_data,
            )
            self = replace(self, state=updated_state)

            runner_state = (self, buffer, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_state(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

        # Set up the buffer
        _, dummy_trajectory = self._collect_rollout(
            (env_state, obsv, key), env, length=self.batch_size // self.num_envs
        )
        buffer = TransitionBuffer(
            max_size=self.replay_buffer_size,
            sample_batch_size=self.batch_size,
            data_sample=dummy_trajectory,
        )
        buffer = buffer.insert(dummy_trajectory)  # Add minimum data to the buffer

        runner_state = (self, buffer, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self

    def _collect_rollout(self, rollout_state, env: Environment, length=None):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            update_count = jym.tree.get_first(self.state, "count")
            current_epsilon = self._epsilon_schedule(update_count)
            get_action = partial(self.get_action, epsilon=current_epsilon)
            action = jax.vmap(get_action, in_axes=(0, None, 0))(
                sample_key, self.state, last_obs
            )

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
                next_observation=info[ORIGINAL_OBSERVATION_KEY],
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        if length is None:
            length = self.num_steps

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, length
        )

        return rollout_state, trajectory_batch

    def _postprocess_rollout(
        self, trajectory_batch: Transition, current_state: DQNState
    ) -> DQNState:
        """
        1) Returns updated normalization based on the new trajectory batch.
        """
        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return updated_state

    def _update_agent_state(
        self, key: PRNGKeyArray, current_state: DQNState, batch: Transition
    ) -> DQNState:
        @eqx.filter_grad
        def __dqn_loss(params: QValueNetwork, train_batch: Transition):
            q_out_1 = jax.vmap(params)(train_batch.observation)
            q_taken = jym.tree.gather_actions(q_out_1, train_batch.action)
            q_taken = jym.tree.batch_sum(q_taken)
            q_loss = optax.huber_loss(q_taken, target)
            return jym.tree.mean(q_loss)

        assert batch.next_observation is not None

        normalizer = current_state.normalizer
        batch = replace(
            batch,
            observation=normalizer.normalize_obs(batch.observation),
            next_observation=normalizer.normalize_obs(batch.next_observation),
            reward=normalizer.normalize_reward(batch.reward),
        )

        # Compute target
        q_target_output = jax.vmap(current_state.critic_target)(batch.next_observation)
        q_target_output = jym.tree.batch_sum(
            jax.tree.map(lambda q: jnp.max(q, axis=-1), q_target_output)
        )
        target = batch.reward + ~batch.terminated * self.gamma * q_target_output

        grads = __dqn_loss(current_state.critic, batch)
        updates, optimizer_state = self.optimizer.update(
            grads, current_state.optimizer_state
        )
        new_critic = eqx.apply_updates(current_state.critic, updates)

        # update target policy
        new_critic_target = jax.tree.map(
            lambda x, y: self.tau * x + (1 - self.tau) * y,
            current_state.critic_target,
            new_critic,
        )

        updated_state = DQNState(
            critic=new_critic,
            critic_target=new_critic_target,
            optimizer_state=optimizer_state,
            normalizer=current_state.normalizer,
        )
        return updated_state

    def _make_agent_state(
        self,
        key: PRNGKeyArray,
        obs_space: jym.Space,
        output_space: jym.Space,
        critic_kwargs: dict[str, Any],
    ):
        critic = QValueNetwork(
            key=key,
            obs_space=obs_space,
            output_space=output_space,
            **critic_kwargs,
        )
        critic_target = jax.tree.map(lambda x: x, critic)

        optimizer_state = self.optimizer.init(eqx.filter(critic, eqx.is_inexact_array))

        dummy_obs = jax.tree.map(
            lambda space: space.sample(jax.random.PRNGKey(0)),
            obs_space,
        )
        normalization_state = Normalizer(
            dummy_obs,
            normalize_obs=self.normalize_observations,
            normalize_rew=self.normalize_rewards,
            gamma=self.gamma,
            rew_shape=(self.num_steps, self.num_envs),
        )

        return DQNState(
            critic=critic,
            critic_target=critic_target,
            optimizer_state=optimizer_state,
            normalizer=normalization_state,
        )
