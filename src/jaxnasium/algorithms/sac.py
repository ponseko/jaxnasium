from __future__ import annotations

import logging
from functools import partial
from typing import Any

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium.algorithms import RLAgent, RLAlgorithm
from jaxnasium.algorithms.core import (
    Normalizer,
    Schedule,
    TanhNormalLayer,
    Transition,
    TransitionBuffer,
    scan_callback,
)

from .agent_networks import ActorNetwork, QValueNetwork

logger = logging.getLogger(__name__)


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
def ensambled_vmap(model, *x):
    """Vmap the ensamble of critics."""
    return jax.vmap(model)(*x)


def _is_categorical_distribution(action_dist: distrax.Distribution) -> bool:
    if isinstance(action_dist, distrax.Categorical):
        return True
    elif isinstance(action_dist, distrax.Independent):
        return _is_categorical_distribution(action_dist.distribution)
    return False


def _unwrap_joint(action_dist):
    if isinstance(action_dist, distrax.Joint):
        return action_dist.distributions
    return action_dist


def _is_dist(x):
    return isinstance(x, distrax.Distribution)


class Alpha(eqx.Module):
    ent_coef: jnp.ndarray

    def __init__(self, ent_coef_init=jnp.log(0.2)):
        self.ent_coef = jnp.array(ent_coef_init)

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.ent_coef)


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC) algorithm implementation.

    This implementation uses soft target updates, a replay buffer, and a target entropy scale with optional annealing.
    """

    learning_rate_actor_start: float = 3e-3
    learning_rate_actor_end: float | None = eqx.field(static=True, default=None)
    learning_rate_critics_start: float = 3e-4
    learning_rate_critics_end: float | None = eqx.field(static=True, default=None)
    learning_rate_alpha_start: float = 3e-3
    learning_rate_alpha_end: float | None = eqx.field(static=True, default=None)

    target_entropy: float | None = eqx.field(static=True, default=None)
    target_entropy_scale_start: float = 1.0
    target_entropy_scale_end: float | None = eqx.field(static=True, default=0.1)
    init_alpha: float = 0.2
    learn_alpha: bool = eqx.field(static=True, default=True)

    gamma: float = 0.99
    max_grad_norm: float | None = 10.0
    update_every: int = eqx.field(static=True, default=128)
    replay_buffer_size: int = eqx.field(static=True, default=50_000)
    warmup_steps: int = eqx.field(static=True, default=10_000)
    """ Warmup for the normalizer and the replay buffer. """
    tau: float = 0.005
    total_timesteps: int = eqx.field(static=True, default=int(5e5))
    num_envs: int = eqx.field(static=True, default=8)

    critics_num_updates: int = eqx.field(static=True, default=16)
    actor_num_updates: int = eqx.field(static=True, default=1)
    alpha_num_updates: int = eqx.field(static=True, default=1)

    batch_size: int = eqx.field(static=True, default=512)
    critics_batch_size: int | None = eqx.field(static=True, default=None)
    actor_batch_size: int | None = eqx.field(static=True, default=None)
    alpha_batch_size: int | None = eqx.field(static=True, default=None)

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=False)
    actor_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    def target_entropy_scale_schedule(self):
        return Schedule(
            self.target_entropy_scale_start,
            self.target_entropy_scale_end,
            self.num_training_updates_alpha,
        )

    @property
    def optimizer(self):
        def _create_optimizer(lr_schedule: Schedule):
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm)
                if self.max_grad_norm is not None
                else optax.identity(),
                optax.adam(learning_rate=lr_schedule, eps=1e-4),
            )

        actor_schedule = Schedule(
            self.learning_rate_actor_start,
            self.learning_rate_actor_end,
            self.num_training_updates_actor,
        )
        critics_schedule = Schedule(
            self.learning_rate_critics_start,
            self.learning_rate_critics_end,
            self.num_training_updates_critics,
        )
        alpha_schedule = Schedule(
            self.learning_rate_alpha_start,
            self.learning_rate_alpha_end,
            self.num_training_updates_alpha,
        )
        return {
            "actor": _create_optimizer(actor_schedule),
            "critics": _create_optimizer(critics_schedule),
            "alpha": _create_optimizer(alpha_schedule),
        }

    @property
    def num_iterations(self):
        return int(self.total_timesteps // self.rollout_length // self.num_envs)

    @property
    def rollout_length(self):
        if self.update_every < self.num_envs:
            raise ValueError(
                f"`update_every` ({self.update_every}) must be >= `num_envs` "
            )
        return int(self.update_every // self.num_envs)

    @property
    def num_training_updates_actor(self):
        return self.num_iterations * self.actor_num_updates

    @property
    def num_training_updates_critics(self):
        return self.num_iterations * self.critics_num_updates

    @property
    def num_training_updates_alpha(self):
        return self.num_iterations * self.alpha_num_updates

    @eqx.filter_jit
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> SACAgent:
        return SACAgent(key=key, env=env, trainer=self)

    @eqx.filter_jit
    def train(
        self, key: PRNGKeyArray, env: Environment, agent: SACAgent | None = None
    ) -> tuple[SACAgent, PyTree[Float[Array, " num_iterations"]]]:
        env = self.__check_env__(env, vectorized=True)

        if agent is None:
            agent = self.init_agent(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))

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
        agent: SACAgent = runner_state[0]
        buffer: TransitionBuffer = runner_state[1]
        rollout_state = runner_state[2:]
        (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
            agent, rollout_state, env
        )

        buffer = buffer.insert(trajectory_batch)
        agent = agent.update_normalizer(trajectory_batch)

        # Update
        agent = self._update_agent(rng, agent, buffer)
        metric = trajectory_batch.info or {}
        runner_state = (agent, buffer, env_state, last_obs, rng)
        return runner_state, metric

    def _collect_rollout(
        self, agent: SACAgent, rollout_state, env: Environment, length=None
    ):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            action = jax.vmap(agent.get_action, in_axes=(0, 0))(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            # Build a single transition. Jax.lax.scan will build the batch
            # returning rollout_length transitions.
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

    def _update_agent(
        self, key: PRNGKeyArray, current_agent: SACAgent, buffer: TransitionBuffer
    ) -> SACAgent:
        def _scan_update(update_fn, update_key, agent, num_updates, batch_size):
            def scan_fn(carry, _):
                agent, rng = carry
                rng, sample_key, update_step_key = jax.random.split(rng, 3)
                minibatch = buffer.sample(
                    sample_key, batch_size=batch_size, with_next_obs=True
                )
                minibatch = minibatch.normalize(agent.normalizer)
                agent = update_fn(agent, update_step_key, minibatch)
                return (agent, rng), None

            (updated_agent, _), _ = jax.lax.scan(
                scan_fn, (agent, update_key), None, length=num_updates
            )
            return updated_agent

        critic_key, actor_key, alpha_key = jax.random.split(key, 3)

        updated_agent = _scan_update(
            lambda agent, *args: agent.update_critics_params(*args),
            critic_key,
            current_agent,
            self.critics_num_updates,
            self.critics_batch_size or self.batch_size,
        )

        updated_agent = _scan_update(
            lambda agent, *args: agent.update_actor_params(*args),
            actor_key,
            updated_agent,
            self.actor_num_updates,
            self.actor_batch_size or self.batch_size,
        )
        if self.learn_alpha:
            updated_agent = _scan_update(
                lambda agent, *args: agent.update_alpha_params(*args),
                alpha_key,
                updated_agent,
                self.alpha_num_updates,
                self.alpha_batch_size or self.batch_size,
            )

        return updated_agent


class SACAgent(RLAgent):
    trainer: SAC

    actor: ActorNetwork
    critics: QValueNetwork
    critics_target: QValueNetwork
    alpha: Alpha
    optimizer_state: dict[str, optax.OptState]
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: SAC):
        self.trainer = trainer
        actor_key, critics_key = jax.random.split(key, 2)

        # Set default continuous distribution to tanhnormal if not specified
        actor_kwargs = dict(trainer.actor_kwargs)
        actor_kwargs.setdefault("continuous_output_layer", TanhNormalLayer)
        self.actor = ActorNetwork(
            env.observation_space,
            env.action_space,
            key=actor_key,
            **actor_kwargs,
        )
        ensamble_critics_keys = jax.random.split(critics_key, 2)  # 2 critics
        self.critics = jax.vmap(
            lambda key: QValueNetwork(
                env.observation_space,
                env.action_space,
                key=key,
                **trainer.critic_kwargs,
            )
        )(ensamble_critics_keys)

        self.critics_target = jax.tree.map(lambda x: x, self.critics)
        self.alpha = Alpha(jnp.log(trainer.init_alpha))

        self.optimizer_state = jym.tree.map_one_level(
            lambda opt, params: opt.init(eqx.filter(params, eqx.is_inexact_array)),
            trainer.optimizer,
            {
                "actor": self.actor,
                "critics": self.critics,
                "alpha": self.alpha,
            },
        )

        self.normalizer = Normalizer(
            obs_space=env.observation_space,
            normalize_obs=trainer.normalize_observations,
            normalize_rew=trainer.normalize_rewards,
            gamma=trainer.gamma,
            rew_shape=(trainer.num_envs,),
        )

    def get_action(
        self, key: PRNGKeyArray, observation, deterministic: bool = False
    ) -> Array:
        observation = self.normalizer.normalize_obs(observation)
        action_dist = self.actor(observation)
        if deterministic:
            return action_dist.mode()
        return action_dist.sample(seed=key)

    def update_normalizer(self, batch: Transition):
        return self.replace(normalizer=self.normalizer.update(batch))

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def _compute_soft_target(self, action_dist, q, action_log_prob=None):
        min_q = q.min(axis=0)
        if _is_categorical_distribution(action_dist):
            # for wrapped in Independent:
            action_dist = getattr(action_dist, "distribution", action_dist)
            action_log_prob = jax.nn.log_softmax(action_dist.logits)
            target = min_q - self.alpha() * action_log_prob
            weighted_target = (action_dist.probs * target).sum(axis=-1)
            return weighted_target
        assert action_log_prob is not None
        min_q = jym.tree.batch_mean(min_q)
        target = min_q - self.alpha() * action_log_prob
        return target

    def update_actor_params(self, key, batch: Transition):
        @eqx.filter_grad
        def __sac_actor_loss(params, train_batch: Transition):
            action_dist = jax.vmap(params)(train_batch.observation)
            action_dist = _unwrap_joint(action_dist)
            keys = jym.tree.split_key_like(key, action_dist, is_leaf=_is_dist)
            action, action_log_prob = jym.tree.map_distribution(
                lambda d, k: d.sample_and_log_prob(seed=k), action_dist, keys
            )
            q = ensambled_vmap(self.critics, train_batch.observation, action)
            target = jym.tree.map_distribution(
                self._compute_soft_target, action_dist, q, action_log_prob
            )
            target = jym.tree.batch_mean(target)
            return -jym.tree.mean(target)

        trainer = self.trainer

        actor_grads = __sac_actor_loss(self.actor, batch)

        updates, optimizer_state = trainer.optimizer["actor"].update(
            actor_grads,
            self.optimizer_state["actor"],
            eqx.filter(self.actor, eqx.is_inexact_array),
        )
        new_actor = eqx.apply_updates(self.actor, updates)
        optimizer_state = {**self.optimizer_state, "actor": optimizer_state}
        return self.replace(actor=new_actor, optimizer_state=optimizer_state)

    def update_critics_params(self, key, batch: Transition):
        @eqx.filter_grad
        def __sac_qnet_loss(params, train_batch: Transition):
            q_out = jax.vmap(params)(train_batch.observation, train_batch.action)
            q_taken = jym.tree.gather_actions(q_out, train_batch.action)
            q_taken = jym.tree.batch_mean(q_taken)
            q_loss = optax.losses.squared_error(q_taken, q_target)
            return jym.tree.mean(q_loss)

        trainer = self.trainer

        action_dist = jax.vmap(self.actor)(batch.next_observation)
        action_dist = _unwrap_joint(action_dist)
        keys = jym.tree.split_key_like(key, action_dist, is_leaf=_is_dist)
        action, action_log_prob = jym.tree.map_distribution(
            lambda d, k: d.sample_and_log_prob(seed=k), action_dist, keys
        )
        q = ensambled_vmap(self.critics_target, batch.next_observation, action)
        target = jym.tree.map_distribution(
            self._compute_soft_target, action_dist, q, action_log_prob
        )
        target = jym.tree.batch_mean(target)
        q_target = batch.reward + (1.0 - batch.terminated) * trainer.gamma * target
        grads = jax.vmap(__sac_qnet_loss, in_axes=(0, None))(self.critics, batch)
        updates, optimizer_state = trainer.optimizer["critics"].update(
            grads,
            self.optimizer_state["critics"],
            eqx.filter(self.critics, eqx.is_inexact_array),
        )
        new_critics = eqx.apply_updates(self.critics, updates)

        new_critics_target = jax.tree.map(
            lambda x, y: (1 - trainer.tau) * x + trainer.tau * y,
            self.critics_target,
            new_critics,
        )
        optimizer_state = {**self.optimizer_state, "critics": optimizer_state}
        return self.replace(
            critics=new_critics,
            critics_target=new_critics_target,
            optimizer_state=optimizer_state,
        )

    def update_alpha_params(self, key, batch: Transition):
        @eqx.filter_grad
        def __sac_alpha_loss(params: Alpha, train_batch: Transition):
            def _compute_alpha_signal(action_dist: distrax.Distribution):
                base_target = trainer.target_entropy
                # NOTE: setting a fixed target entropy in mixed action spaces likely does not set a proper target entropy
                if _is_categorical_distribution(action_dist):
                    # for wrapped in Independent:
                    action_dist = getattr(action_dist, "distribution", action_dist)
                    action_probs = action_dist.probs  # type: ignore[Attribute]
                    log_probs = jax.nn.log_softmax(action_dist.logits)  # type: ignore[Attribute]
                    if base_target is None:
                        base_target = 0.89 * jnp.log(log_probs.shape[-1])
                    # Target is positive and scaling it toward 0 drives the policy deterministic.
                    target_entropy = target_entropy_scale * base_target
                    return (action_probs * (log_probs + target_entropy)).sum(axis=-1)
                else:  # Continuous action space
                    _, log_probs = action_dist.sample_and_log_prob(seed=key)
                    if base_target is None:
                        base_target = -max(1, int(np.prod(action_dist.event_shape)))
                    # Target is negative and driving it to zero would increase stochasticity, hence we scale it here instead.
                    target_entropy = base_target * (2.0 - target_entropy_scale)
                    return log_probs + target_entropy

            signals = jym.tree.map_distribution(_compute_alpha_signal, action_dist)
            signals = jym.tree.batch_sum(signals)
            return -jnp.mean(params() * signals)

        trainer = self.trainer

        count = jym.tree.get_first(self.optimizer_state["alpha"], "count")
        target_entropy_scale = trainer.target_entropy_scale_schedule(count)
        action_dist = jax.vmap(self.actor)(batch.observation)
        alpha_grads = __sac_alpha_loss(self.alpha, batch)

        updates, optimizer_state = trainer.optimizer["alpha"].update(
            alpha_grads,
            self.optimizer_state["alpha"],
            eqx.filter(self.alpha, eqx.is_inexact_array),
        )
        new_alpha = eqx.apply_updates(self.alpha, updates)
        optimizer_state = {**self.optimizer_state, "alpha": optimizer_state}
        return self.replace(alpha=new_alpha, optimizer_state=optimizer_state)
