import logging
from dataclasses import replace
from typing import Any

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAlgorithm
from jaxnasium.algorithms.utils import (
    Normalizer,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .networks import ActorNetwork, QValueNetwork

logger = logging.getLogger(__name__)


class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init=jnp.log(0.2)):
        self.ent_coef = jnp.array(ent_coef_init)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.ent_coef)


class SACState(eqx.Module):
    actor: ActorNetwork
    critic1: QValueNetwork
    critic2: QValueNetwork
    critic1_target: QValueNetwork
    critic2_target: QValueNetwork
    alpha: Alpha
    optimizer_state: optax.OptState
    normalizer: Normalizer


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC) algorithm implementation.

    This implementation uses soft target updates, a replay buffer, and a target entropy scale with optional annealing.
    """

    state: SACState = eqx.field(default=None)
    optimizer: optax.GradientTransformation = eqx.field(static=True, default=None)

    learning_rate: float = 3e-3
    anneal_learning_rate: bool | float = eqx.field(static=True, default=True)
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    update_every: int = eqx.field(static=True, default=128)
    replay_buffer_size: int = 5000
    batch_size: int = 128
    init_alpha: float = 0.2
    learn_alpha: bool = eqx.field(static=True, default=True)
    target_entropy_scale: float = 0.5
    anneal_entropy_scale: bool | float = eqx.field(static=True, default=False)
    tau: float = 0.95

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=8)

    normalize_observations: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=False)

    actor_kwargs: dict[str, Any] = eqx.field(
        static=True, default_factory=lambda: {"continuous_output_dist": "tanhNormal"}
    )

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
    def _target_entropy_scale_schedule(self):
        if self.anneal_entropy_scale:
            end_value = (
                0.0 if self.anneal_entropy_scale is True else self.anneal_entropy_scale
            )
            return optax.linear_schedule(
                init_value=self.target_entropy_scale,
                end_value=end_value,
                transition_steps=self.num_training_updates,
            )
        return optax.constant_schedule(self.target_entropy_scale)

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
        state: SACState,
        observation: PyTree,
        deterministic: bool = False,
    ) -> Array:
        observation = state.normalizer.normalize_obs(observation)
        action_dist = state.actor(observation)
        if deterministic:
            return action_dist.mode()
        return action_dist.sample(seed=key)

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "SAC":
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

    def train(self, key: PRNGKeyArray, env: Environment, **hyperparams) -> "SAC":
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
            self: SAC = runner_state[0]
            buffer: TransitionBuffer = runner_state[1]
            rollout_state = runner_state[2:]
            (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
                rollout_state, env
            )

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

            metric = trajectory_batch.info or {}
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
            action = jax.vmap(self.get_action, in_axes=(0, None, 0))(
                sample_key, self.state, last_obs
            )

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # Build a single transition. Jax.lax.scan will build the batch
            # returning num_steps transitions.
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
        self, trajectory_batch: Transition, current_state: SACState
    ) -> SACState:
        """
        1) Returns updated normalization based on the new trajectory batch.
        """
        # Update normalization params
        updated_state = replace(
            current_state, normalizer=current_state.normalizer.update(trajectory_batch)
        )

        return updated_state

    def _update_agent_state(
        self, key: PRNGKeyArray, current_state: SACState, batch: Transition
    ) -> SACState:
        def _compute_soft_target(action_dist, action_log_probs, q_1, q_2):
            def discrete_soft_target(action_probs, q_1, q_2):
                action_log_prob = jnp.log(action_probs + 1e-8)
                min_q = jnp.minimum(q_1, q_2)
                target = min_q - current_state.alpha() * action_log_prob
                weighted_target = (action_probs * target).sum(axis=-1)
                return weighted_target

            def continuous_soft_target(action_log_probs, q_1, q_2):
                min_q = jnp.minimum(q_1, q_2)
                return min_q - current_state.alpha() * action_log_probs

            if isinstance(action_dist, distrax.Categorical):
                return discrete_soft_target(action_dist.probs, q_1, q_2)
            return continuous_soft_target(action_log_probs, q_1, q_2)

        @eqx.filter_grad
        def __sac_qnet_loss(params, batch: Transition):
            def get_q_from_actions(q, actions):
                """Get the Q value from the actions that were taken."""
                if q.squeeze().shape == actions.squeeze().shape:
                    # Q is already given for the taken action (Continuous case)
                    return q
                # Discrete case: we need to index the Q values with the actions
                return jnp.take_along_axis(q, actions[..., None], axis=-1).squeeze()

            q_out = jax.vmap(params)(batch.observation, batch.action)
            q_out = jax.tree.map(get_q_from_actions, q_out, batch.action)
            # q_loss = jax.tree.map(
            #     lambda q, t: optax.losses.huber_loss(q, t), q_out, q_target
            # )
            q_loss = jax.tree.map(lambda q, t: jnp.mean((q - t) ** 2), q_out, q_target)
            return jym.tree.mean(q_loss)

        @eqx.filter_grad
        def __sac_actor_loss(params, batch: Transition):
            action_dist = jax.vmap(params)(batch.observation)
            action, log_prob = action_dist.sample_and_log_prob(seed=actor_loss_key)
            q_1 = jax.vmap(current_state.critic1)(batch.observation, action)
            q_2 = jax.vmap(current_state.critic2)(batch.observation, action)
            target = jym.tree.map_distribution(
                _compute_soft_target, action_dist, log_prob, q_1, q_2
            )
            return -jym.tree.mean(target)

        @eqx.filter_grad
        def __sac_alpha_loss(params: Alpha):
            def alpha_loss_per_action_dist(action_dist):
                if isinstance(action_dist, distrax.Categorical):
                    log_probs = jnp.log(action_dist.probs + 1e-8)
                    action_dim = jnp.prod(jnp.array(log_probs.shape[1:]))
                    action_dim = jnp.log(1 / action_dim)
                else:  # Continuous action space
                    _, log_probs = action_dist.sample_and_log_prob(seed=actor_loss_key)
                    action_dim = jnp.prod(jnp.array(batch.action.shape[1:]))

                update_count = jym.tree.get_first(
                    current_state.optimizer_state, "count"
                )
                target_entropy_scale = self._target_entropy_scale_schedule(update_count)
                target_entropy = -(target_entropy_scale * action_dim)
                return -jnp.mean(params() * (log_probs + target_entropy))

            action_dist = jax.vmap(current_state.actor)(batch.observation)
            loss = jym.tree.map_distribution(alpha_loss_per_action_dist, action_dist)
            loss = jym.tree.mean(loss)
            return loss

        rng, target_key, actor_loss_key = jax.random.split(key, 3)

        action_dist = jax.vmap(current_state.actor)(batch.next_observation)
        action, action_log_prob = action_dist.sample_and_log_prob(seed=target_key)
        q_1 = jax.vmap(current_state.critic1_target)(batch.next_observation, action)
        q_2 = jax.vmap(current_state.critic2_target)(batch.next_observation, action)
        target = jym.tree.map_distribution(
            _compute_soft_target, action_dist, action_log_prob, q_1, q_2
        )
        reward = current_state.normalizer.normalize_reward(batch.reward)

        def broadcast_to_match(x, target_ndim):
            """Add dimensions to x to match target_ndim."""
            return jnp.reshape(x, x.shape + (1,) * (target_ndim - x.ndim))

        q_target = jax.tree.map(
            lambda targ: broadcast_to_match(reward, targ.ndim)
            + (1.0 - broadcast_to_match(batch.terminated, targ.ndim))
            * self.gamma
            * targ,
            target,
        )

        critic1_grads = __sac_qnet_loss(current_state.critic1, batch)
        critic2_grads = __sac_qnet_loss(current_state.critic2, batch)
        actor_grads = __sac_actor_loss(current_state.actor, batch)
        alpha_grads = __sac_alpha_loss(current_state.alpha)

        updates, optimizer_state = self.optimizer.update(
            (actor_grads, critic1_grads, critic2_grads, alpha_grads),
            current_state.optimizer_state,
        )
        new_actor, new_critic1, new_critic2, new_alpha = eqx.apply_updates(
            (
                current_state.actor,
                current_state.critic1,
                current_state.critic2,
                current_state.alpha,
            ),
            updates,
        )

        # Update target networks
        new_critic1_target, new_critic2_target = jax.tree.map(
            lambda x, y: self.tau * x + (1 - self.tau) * y,
            (current_state.critic1_target, current_state.critic2_target),
            (new_critic1, new_critic2),
        )

        updated_state = SACState(
            actor=new_actor,
            critic1=new_critic1,
            critic2=new_critic2,
            critic1_target=new_critic1_target,
            critic2_target=new_critic2_target,
            optimizer_state=optimizer_state,
            alpha=new_alpha if self.learn_alpha else current_state.alpha,
            normalizer=current_state.normalizer,
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
        actor_key, critic1_key, critic2_key = jax.random.split(key, 3)
        actor = ActorNetwork(
            key=actor_key,
            obs_space=obs_space,
            output_space=output_space,
            **actor_kwargs,
        )
        critic1 = QValueNetwork(
            key=critic1_key,
            obs_space=obs_space,
            output_space=output_space,
            **critic_kwargs,
        )
        critic2 = QValueNetwork(
            key=critic2_key,
            obs_space=obs_space,
            output_space=output_space,
            **critic_kwargs,
        )
        critic1_target = jax.tree.map(lambda x: x, critic1)
        critic2_target = jax.tree.map(lambda x: x, critic2)
        alpha = Alpha(jnp.log(self.init_alpha))

        optimizer_state = self.optimizer.init(
            eqx.filter((actor, critic1, critic2, alpha), eqx.is_inexact_array)
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

        return SACState(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
            alpha=alpha,
            optimizer_state=optimizer_state,
            normalizer=normalization_state,
        )
