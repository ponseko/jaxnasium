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
    scan_callback,
)

from .networks import QValueNetwork

logger = logging.getLogger(__name__)


class PQNState(eqx.Module):
    critic: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer


class PQN(RLAlgorithm):
    """Parallel Q-Network (PQN) algorithm implementation."""

    state: PQNState = eqx.field(default=None)
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float = 2.5e-4
    anneal_learning_rate: bool | float = eqx.field(static=True, default=True)
    gamma: float = 0.99
    max_grad_norm: float = 10.0
    epsilon: float = 0.25
    anneal_epsilon: bool | float = eqx.field(static=True, default=True)
    q_lambda: float = 0.65

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=12)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=False)

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
    def minibatch_size(self):
        return self.num_envs * self.num_steps // self.num_minibatches

    @property
    def num_iterations(self):
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    def batch_size(self):
        return self.minibatch_size * self.num_minibatches

    @property
    def num_training_updates(self):
        return self.num_iterations * self.num_epochs * self.num_minibatches

    @staticmethod
    def get_action(
        key: PRNGKeyArray,
        state: PQNState,
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

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "PQN":
        if getattr(env, "multi_agent", False) and self.auto_upgrade_multi_agent:
            self = self.__make_multi_agent__()

        if self.optimizer is None:
            self = replace(
                self,
                optimizer=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adabelief(learning_rate=self.learning_rate),
                ),
            )

        agent_states = self._make_agent_state(
            key=key,
            obs_space=env.observation_space,
            output_space=env.action_space,
            critic_kwargs=self.critic_kwargs,
        )

        return replace(self, state=agent_states)

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PQN":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, train_iter):
            """
            Performs a single training iteration (A single `Collect data + Update` run).
            This is repeated until the total number of timesteps is reached.
            """

            # Do rollout of single trajactory
            self: PQN = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env, train_iter
            )
            metric = trajectory_batch.info or {}

            # Post-process the trajectory batch (GAE, returns, normalization)
            trajectory_batch, updated_state = self._postprocess_rollout(
                trajectory_batch, self.state
            )

            # Update agent
            updated_state = self._update_agent_state(
                rng,
                updated_state,  # <-- Use updated_state w/ updated normalizer
                trajectory_batch,
            )
            self = replace(self, state=updated_state)

            runner_state = (self, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_state(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        runner_state = (self, env_state, obsv, key)
        runner_state, metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        updated_self = runner_state[0]
        return updated_self

    def _collect_rollout(self, rollout_state, env: Environment, train_iter: int = 0):
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
                next_observation=info[ORIGINAL_OBSERVATION_KEY],
                info=info,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, self.num_steps
        )

        return rollout_state, trajectory_batch

    def _postprocess_rollout(
        self, trajectory_batch: Transition, current_state: PQNState
    ) -> tuple[Transition, PQNState]:
        """Returns updated normalization based on the new trajectory batch
        and returns the Qlambda targets.
        """

        def compute_q_lambda_scan(next_return, batch: Transition):
            next_obs = batch.next_observation
            norm_next_obs = current_state.normalizer.normalize_obs(next_obs)
            norm_reward = current_state.normalizer.normalize_reward(batch.reward)
            next_q_values = jax.vmap(current_state.critic)(norm_next_obs)
            next_q_values = jax.tree.map(lambda q: jnp.max(q, axis=-1), next_q_values)
            next_q_values = jym.tree.batch_sum(next_q_values)

            done = batch.terminated
            if done.ndim < norm_reward.ndim:
                # correct for multi-agent envs that do not return done flags per agent
                done = jnp.expand_dims(done, axis=-1)

            return_this_step = norm_reward + (1 - done) * self.gamma * (
                self.q_lambda * next_return + (1 - self.q_lambda) * next_q_values
            )
            return return_this_step, return_this_step

        assert trajectory_batch.next_observation is not None
        last_obs = current_state.normalizer.normalize_obs(
            jax.tree.map(lambda x: x[-1], trajectory_batch.next_observation)
        )
        last_q_values = jax.vmap(current_state.critic)(last_obs)
        last_q_values = jax.tree.map(lambda q: jnp.max(q, axis=-1), last_q_values)
        last_q_values = jym.tree.batch_sum(last_q_values)
        _, returns = jax.lax.scan(
            compute_q_lambda_scan,
            last_q_values.astype(jnp.float32),
            trajectory_batch,
            reverse=True,
            unroll=16,
        )

        trajectory_batch = replace(trajectory_batch, return_=returns)

        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return trajectory_batch, updated_state

    def _update_agent_state(
        self, key, current_state: PQNState, train_data: Transition
    ) -> PQNState:
        def scan_minibatch_update(current_state: PQNState, minibatch: Transition):
            @eqx.filter_grad
            def __dqn_loss(params: QValueNetwork, train_batch: Transition):
                q_out_1 = jax.vmap(params)(train_batch.observation)
                q_taken = jym.tree.gather_actions(q_out_1, train_batch.action)
                q_taken = jym.tree.batch_sum(q_taken)
                q_loss = optax.huber_loss(q_taken, minibatch.return_)
                return jym.tree.mean(q_loss)

            grads = __dqn_loss(current_state.critic, minibatch)
            updates, optimizer_state = self.optimizer.update(
                grads, current_state.optimizer_state
            )
            new_critic = eqx.apply_updates(current_state.critic, updates)

            updated_state = PQNState(
                critic=new_critic,
                optimizer_state=optimizer_state,
                normalizer=current_state.normalizer,  # already updated
            )
            return updated_state, None

        def scan_epoch_update(current_state, key):
            minibatches = train_data.make_minibatches(
                key, self.num_minibatches, n_batch_axis=2
            )
            updated_state, _ = jax.lax.scan(
                scan_minibatch_update, current_state, minibatches
            )
            return updated_state, None

        train_data = replace(
            train_data,
            observation=current_state.normalizer.normalize_obs(train_data.observation),
        )
        update_keys = jax.random.split(key, self.num_epochs)
        updated_state, _ = jax.lax.scan(
            scan_epoch_update, current_state, update_keys, unroll=16
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

        return PQNState(
            critic=critic,
            optimizer_state=optimizer_state,
            normalizer=normalization_state,
        )
