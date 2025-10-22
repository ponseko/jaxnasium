import logging
from dataclasses import replace
from functools import partial
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAlgorithm
from jaxnasium.algorithms.utils import Normalizer, Transition, scan_callback

from .networks import ActorNetwork, ValueNetwork

logger = logging.getLogger(__name__)


class PPOState(eqx.Module):
    actor: ActorNetwork
    critic: ValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm implementation."""

    state: PPOState = eqx.field(default=None)
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float = 2.5e-3
    anneal_learning_rate: bool | float = eqx.field(static=True, default=True)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    ent_coef: float = 0.01
    anneal_ent_coef: bool | float = eqx.field(static=True, default=False)
    vf_coef: float = 0.25

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs

    normalize_observations: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=True)

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
    def _ent_coef_schedule(self):
        if self.anneal_ent_coef:
            end_value = 0.0 if self.anneal_ent_coef is True else self.anneal_ent_coef
            return optax.linear_schedule(
                init_value=self.ent_coef,
                end_value=end_value,
                transition_steps=self.num_training_updates,
            )
        return optax.constant_schedule(self.ent_coef)

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
        return self.num_iterations * self.num_epochs

    @staticmethod
    def get_action(
        key: PRNGKeyArray,
        state: PPOState,
        observation: PyTree,
        deterministic: bool = False,
        get_log_prob: bool = False,
    ) -> Array | Tuple[Array, Array]:
        observation = state.normalizer.normalize_obs(observation)
        action_dist = state.actor(observation)
        if deterministic:
            assert not get_log_prob, "Cannot get log prob in deterministic mode"
            return action_dist.mode()
        if get_log_prob:
            return action_dist.sample_and_log_prob(seed=key)  # type: ignore
        return action_dist.sample(seed=key)

    @staticmethod
    def get_value(state: PPOState, observation: PyTree):
        observation = state.normalizer.normalize_obs(observation)
        return state.critic(observation)

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "PPO":
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
            actor_kwargs=self.actor_kwargs,
            critic_kwargs=self.critic_kwargs,
        )

        return replace(self, state=agent_states)

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "PPO":
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
            self: PPO = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )
            metric = trajectory_batch.info or {}

            # Post-process the trajectory batch (GAE, returns, normalization)
            trajectory_batch, updated_state = self._postprocess_rollout(
                trajectory_batch, self.state
            )

            # Update agent
            updated_state = self._update_agent_state(
                rng,
                updated_state,  # <-- Use updated state with updated normalizer
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

    def _collect_rollout(self, rollout_state, env: Environment):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            get_action_and_log_prob = partial(self.get_action, get_log_prob=True)
            action, log_prob = jax.vmap(get_action_and_log_prob, in_axes=(0, None, 0))(
                sample_key, self.state, last_obs
            )

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )
            value = jax.vmap(self.get_value, in_axes=(None, 0))(self.state, last_obs)
            next_value = jax.vmap(self.get_value, in_axes=(None, 0))(
                self.state, info[ORIGINAL_OBSERVATION_KEY]
            )

            # TODO ?
            # gamma = info.get("ENV_GAMMA", self.gamma)

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                info=info,
                value=value,
                next_value=next_value,
            )

            rollout_state = (env_state, obsv, rng)
            return rollout_state, transition

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, self.num_steps
        )

        return rollout_state, trajectory_batch

    def _postprocess_rollout(
        self, trajectory_batch: Transition, current_state: PPOState
    ) -> Tuple[Transition, PPOState]:
        """
        1) Computes GAE and Returns and adds them to the trajectory batch.
        2) Returns updated normalization based on the new trajectory batch.
        """

        def compute_gae_scan(gae, batch: Transition):
            """
            Computes the Generalized Advantage Estimation (GAE) for the given batch of transitions.
            """

            assert batch.value is not None
            assert batch.next_value is not None

            reward = current_state.normalizer.normalize_reward(batch.reward)
            done = batch.terminated
            if done.ndim < reward.ndim:
                # correct for multi-agent envs that do not return done flags per agent
                done = jnp.expand_dims(done, axis=-1)

            delta = reward + self.gamma * batch.next_value * (1 - done) - batch.value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            return gae, (gae, gae + batch.value)

        assert trajectory_batch.value is not None
        _, (advantages, returns) = jax.lax.scan(
            compute_gae_scan,
            optax.tree.zeros_like(trajectory_batch.value[-1]),
            trajectory_batch,
            reverse=True,
            unroll=16,
        )

        trajectory_batch = replace(
            trajectory_batch,
            advantage=advantages,
            return_=returns,
        )

        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return trajectory_batch, updated_state

    def _update_agent_state(
        self, key, current_state: PPOState, train_data: Transition
    ) -> PPOState:
        @eqx.filter_grad
        def __ppo_los_fn(
            params: Tuple[ActorNetwork, ValueNetwork],
            train_batch: Transition,
        ):
            assert train_batch.advantage is not None
            assert train_batch.return_ is not None
            assert train_batch.log_prob is not None

            actor, critic = params
            norm_obs = current_state.normalizer.normalize_obs(train_batch.observation)
            action_dist = jax.vmap(actor)(norm_obs)
            log_prob = action_dist.log_prob(train_batch.action)
            entropy = action_dist.entropy()
            value = jax.vmap(critic)(norm_obs)
            init_log_prob = train_batch.log_prob

            log_prob = jym.tree.batch_sum(log_prob)
            init_log_prob = jym.tree.batch_sum(init_log_prob)
            entropy = jym.tree.batch_sum(entropy)

            ratio = jnp.exp(log_prob - init_log_prob)
            _advantages = (train_batch.advantage - train_batch.advantage.mean()) / (
                train_batch.advantage.std() + 1e-8
            )
            actor_loss1 = _advantages * ratio

            actor_loss2 = (
                jnp.clip(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                * _advantages
            )
            actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

            # critic loss
            value_pred_clipped = train_batch.value + (
                jnp.clip(
                    value - train_batch.value,
                    -self.clip_coef_vf,
                    self.clip_coef_vf,
                )
            )
            value_losses = jnp.square(value - train_batch.return_)
            value_losses_clipped = jnp.square(value_pred_clipped - train_batch.return_)
            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

            update_count = jym.tree.get_first(current_state.optimizer_state, "count")
            ent_coef = self._ent_coef_schedule(update_count)

            # Total loss
            total_loss = (
                actor_loss + self.vf_coef * value_loss - ent_coef * entropy.mean()
            )
            return total_loss  # , (actor_loss, value_loss, entropy)

        def scan_minibatch_update(current_state, minibatch):
            actor, critic = current_state.actor, current_state.critic
            grads = __ppo_los_fn((actor, critic), minibatch)
            updates, optimizer_state = self.optimizer.update(
                grads, current_state.optimizer_state
            )
            new_actor, new_critic = eqx.apply_updates((actor, critic), updates)

            updated_state = PPOState(
                actor=new_actor,
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
        actor_kwargs: dict[str, Any],
        critic_kwargs: dict[str, Any],
    ):
        actor_key, critic_key = jax.random.split(key)
        actor = ActorNetwork(
            key=actor_key,
            obs_space=obs_space,
            output_space=output_space,
            **actor_kwargs,
        )
        critic = ValueNetwork(
            key=critic_key,
            obs_space=obs_space,
            **critic_kwargs,
        )
        optimizer_state = self.optimizer.init(
            eqx.filter((actor, critic), eqx.is_inexact_array)
        )

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

        return PPOState(
            actor=actor,
            critic=critic,
            optimizer_state=optimizer_state,
            normalizer=normalization_state,
        )
