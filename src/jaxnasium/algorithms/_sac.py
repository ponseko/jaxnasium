import logging
from dataclasses import replace
from typing import Any

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

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
    optimizer_state_actor: optax.OptState
    optimizer_state_critics: optax.OptState
    normalizer: Normalizer


def _create_schedule(anneal_param, init_value, num_training_updates):
    """Create a linear or constant schedule based on annealing parameter."""
    if anneal_param:
        end_value = 0.0 if anneal_param is True else anneal_param
        return optax.linear_schedule(
            init_value=init_value,
            end_value=end_value,
            transition_steps=num_training_updates,
        )
    return optax.constant_schedule(init_value)


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC) algorithm implementation.

    This implementation uses soft target updates, a replay buffer, and a target entropy scale with optional annealing.
    """

    state: SACState = eqx.field(default=None)
    optimizer_actor: optax.GradientTransformation = eqx.field(static=True, default=None)
    optimizer_critics: optax.GradientTransformation = eqx.field(
        static=True, default=None
    )

    learning_rate_actor: float = 3e-3
    learning_rate_critics: float = 3e-4
    anneal_learning_rate_actor: bool | float = eqx.field(static=True, default=True)
    anneal_learning_rate_critics: bool | float = eqx.field(static=True, default=True)

    gamma: float = 0.99
    max_grad_norm: float = 0.5
    # update_every: int = eqx.field(static=True, default=128)
    num_steps: int = eqx.field(static=True, default=16)
    replay_buffer_size: int = 5000
    batch_size: int = 128
    init_alpha: float = 0.2
    learn_alpha: bool = eqx.field(static=True, default=True)
    target_entropy_scale: float = 0.5
    anneal_entropy_scale: bool | float = eqx.field(static=True, default=False)
    tau: float = 0.95

    actor_num_epochs: int = eqx.field(static=True, default=1)
    actor_num_minibatches: int = eqx.field(static=True, default=1)
    critics_num_epochs: int = eqx.field(static=True, default=8)
    critics_num_minibatches: int = eqx.field(static=True, default=1)

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=8)

    normalize_observations: bool = eqx.field(static=True, default=False)
    normalize_rewards: bool = eqx.field(static=True, default=False)

    actor_kwargs: dict[str, Any] = eqx.field(
        static=True, default_factory=lambda: {"continuous_output_dist": "tanhNormal"}
    )

    @property
    def _target_entropy_scale_schedule(self):
        return _create_schedule(
            self.anneal_entropy_scale,
            self.target_entropy_scale,
            self.num_training_updates_actor,
        )

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.num_steps // self.num_envs)

    @property
    def update_every(self):
        return int(self.num_steps * self.num_envs)

    @property
    def num_training_updates_actor(self):
        return self.num_iterations * self.actor_num_epochs * self.actor_num_minibatches

    @property
    def num_training_updates_critics(self):
        return (
            self.num_iterations * self.critics_num_epochs * self.critics_num_minibatches
        )

    @staticmethod
    def get_action(
        key: PRNGKeyArray, state: SACState, observation, deterministic: bool = False
    ) -> Array:
        observation = state.normalizer.normalize_obs(observation)
        action_dist = state.actor(observation)
        if deterministic:
            return action_dist.mode()
        return action_dist.sample(seed=key)

    def init_state(self, key: PRNGKeyArray, env: Environment) -> "SAC":
        if getattr(env, "multi_agent", False) and self.auto_upgrade_multi_agent:
            self = self.__make_multi_agent__()

        if self.optimizer_actor is None:
            self = replace(
                self,
                optimizer_actor=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adabelief(
                        learning_rate=_create_schedule(
                            self.anneal_learning_rate_actor,
                            self.learning_rate_actor,
                            self.num_training_updates_actor,
                        )
                    ),
                ),
            )

        if self.optimizer_critics is None:
            self = replace(
                self,
                optimizer_critics=optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.adabelief(
                        learning_rate=_create_schedule(
                            self.anneal_learning_rate_critics,
                            self.learning_rate_critics,
                            self.num_training_updates_critics,
                        )
                    ),
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
            train_batch = buffer.sample(rng)

            # Update
            updated_state = self._update_agent_state(
                rng,
                updated_state,  # <-- use updated_state w/ updated norm
                train_batch,
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
        self, key: PRNGKeyArray, current_state: SACState, train_batch: Transition
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

        def update_critics(current_state: SACState, batch: Transition):
            @eqx.filter_grad
            def __sac_qnet_loss(params, batch: Transition):
                q_out = jax.vmap(params)(batch.observation, batch.action)
                q_taken = jym.tree.gather_actions(q_out, batch.action)
                q_taken = jym.tree.batch_sum(q_taken)
                q_loss = optax.losses.huber_loss(q_taken, q_target)
                return jym.tree.mean(q_loss)

            action_dist = jax.vmap(current_state.actor)(batch.next_observation)
            action, action_log_prob = action_dist.sample_and_log_prob(seed=keys[3])
            q_1_target = jax.vmap(current_state.critic1_target)(
                batch.next_observation, action
            )
            q_2_target = jax.vmap(current_state.critic2_target)(
                batch.next_observation, action
            )
            target = jym.tree.map_distribution(
                _compute_soft_target,
                action_dist,
                action_log_prob,
                q_1_target,
                q_2_target,
            )
            target = jym.tree.batch_sum(target)
            q_target = batch.reward + (1.0 - batch.terminated) * self.gamma * target
            critic_1_grads = __sac_qnet_loss(current_state.critic1, batch)
            critic_2_grads = __sac_qnet_loss(current_state.critic2, batch)
            updates, optimizer_state = self.optimizer_critics.update(
                (critic_1_grads, critic_2_grads),
                current_state.optimizer_state_critics,
            )
            new_critic1, new_critic2 = eqx.apply_updates(
                (current_state.critic1, current_state.critic2),
                updates,
            )
            return replace(
                current_state,
                critic1=new_critic1,
                critic2=new_critic2,
                optimizer_state_critics=optimizer_state,
            ), None

        def update_actor_and_alpha(current_state: SACState, batch: Transition):
            @eqx.filter_grad
            def __sac_actor_loss(params, batch: Transition):
                action_dist = jax.vmap(params)(batch.observation)
                action, log_prob = action_dist.sample_and_log_prob(seed=keys[4])
                q_1 = jax.vmap(current_state.critic1)(batch.observation, action)
                q_2 = jax.vmap(current_state.critic2)(batch.observation, action)
                target = jym.tree.map_distribution(
                    _compute_soft_target, action_dist, log_prob, q_1, q_2
                )
                return -jym.tree.mean(target)

            @eqx.filter_grad
            def __sac_alpha_loss(params: Alpha, batch: Transition):
                def alpha_loss_per_action_dist(action_dist):
                    if isinstance(action_dist, distrax.Categorical):
                        log_probs = jnp.log(action_dist.probs + 1e-8)
                        action_dim = jnp.prod(jnp.array(log_probs.shape[1:]))
                        action_dim = jnp.log(1 / action_dim)
                    else:  # Continuous action space
                        _, log_probs = action_dist.sample_and_log_prob(seed=keys[5])
                        action_dim = jnp.prod(jnp.array(batch.action.shape[1:]))

                    update_count = jym.tree.get_first(
                        current_state.optimizer_state_actor, "count"
                    )
                    target_entropy_scale = self._target_entropy_scale_schedule(
                        update_count
                    )
                    target_entropy = -(target_entropy_scale * action_dim)
                    return -jnp.mean(params() * (log_probs + target_entropy))

                action_dist = jax.vmap(current_state.actor)(batch.observation)
                loss = jym.tree.map_distribution(
                    alpha_loss_per_action_dist, action_dist
                )
                loss = jym.tree.mean(loss)
                return loss

            actor_grads = __sac_actor_loss(current_state.actor, batch)
            alpha_grads = __sac_alpha_loss(current_state.alpha, batch)

            updates, optimizer_state = self.optimizer_actor.update(
                (actor_grads, alpha_grads),
                current_state.optimizer_state_actor,
            )
            new_actor, new_alpha = eqx.apply_updates(
                (current_state.actor, current_state.alpha),
                updates,
            )
            return replace(
                current_state,
                actor=new_actor,
                alpha=new_alpha if self.learn_alpha else current_state.alpha,
                optimizer_state_actor=optimizer_state,
            ), None

        # Normalize all used inputs, if normalization is disabled, these are no-ops
        normalizer = current_state.normalizer
        train_batch = replace(
            train_batch,
            observation=normalizer.normalize_obs(train_batch.observation),
            next_observation=normalizer.normalize_obs(train_batch.next_observation),
            reward=normalizer.normalize_reward(train_batch.reward),
        )

        keys = jax.random.split(key, 5)

        # Update the critics for each minibatch and epoch
        critic_train_batches = train_batch.make_minibatches(
            keys[1],
            self.critics_num_minibatches,
            self.critics_num_epochs,
            n_batch_axis=1,
        )
        updated_state, _ = jax.lax.scan(
            update_critics, current_state, critic_train_batches
        )

        # Update the actor and Alpha for each minibatch and epoch
        actor_train_batches = train_batch.make_minibatches(
            keys[2],
            self.actor_num_minibatches,
            self.actor_num_epochs,
            n_batch_axis=1,
        )
        updated_state, _ = jax.lax.scan(
            update_actor_and_alpha, updated_state, actor_train_batches
        )

        # Update target networks
        new_critic1_target, new_critic2_target = jax.tree.map(
            lambda x, y: self.tau * x + (1 - self.tau) * y,
            (current_state.critic1_target, current_state.critic2_target),
            (updated_state.critic1, updated_state.critic2),
        )

        return replace(
            updated_state,
            critic1_target=new_critic1_target,
            critic2_target=new_critic2_target,
        )

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

        # Alpha shares the same optimizer
        optimizer_state_actor = self.optimizer_actor.init(
            eqx.filter((actor, alpha), eqx.is_inexact_array)
        )
        optimizer_state_critics = self.optimizer_critics.init(
            eqx.filter((critic1, critic2), eqx.is_inexact_array)
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
            optimizer_state_actor=optimizer_state_actor,
            optimizer_state_critics=optimizer_state_critics,
            normalizer=normalization_state,
        )
