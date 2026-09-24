from __future__ import annotations

import logging
from dataclasses import replace
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import jaxnasium as jym
from jaxnasium import Environment
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import RLAgent, RLAlgorithm
from jaxnasium.algorithms.core import (
    Normalizer,
    Schedule,
    Transition,
    scan_callback,
    scan_minibatch_epoch,
)

from .agent_networks import ActorNetwork, ValueNetwork

logger = logging.getLogger(__name__)


class PPO(RLAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm implementation."""

    learning_rate_start: float = 2.5e-4
    learning_rate_end: float | None = eqx.field(static=True, default=None)
    ent_coef_start: float = 0.1
    ent_coef_end: float | None = eqx.field(static=True, default=0.01)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float | None = 10.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0
    vf_coef: float = 0.25

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=6)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs
    warmup_steps: int = eqx.field(static=True, default=5_000)
    """Normalizer statistics warmup steps."""

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=True)

    actor_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    def learning_rate_schedule(self):
        return Schedule(
            start=self.learning_rate_start,
            end=self.learning_rate_end,
            transition_steps=self.num_training_updates,
        )

    @property
    def ent_coef_schedule(self):
        return Schedule(
            start=self.ent_coef_start,
            end=self.ent_coef_end,
            transition_steps=self.num_training_updates,
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

    @eqx.filter_jit
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> PPOAgent:
        return PPOAgent(key=key, env=env, trainer=self)

    @eqx.filter_jit
    def train(
        self, key: PRNGKeyArray, env: Environment, agent: PPOAgent | None = None
    ) -> tuple[PPOAgent, PyTree[Float[Array, " num_iterations"]]]:
        env = self.__check_env__(env, vectorized=True)

        if agent is None:
            agent = self.init_agent(key, env)

        train_iteration_fn = partial(self.train_iteration, env=env)
        train_iteration_fn = scan_callback(
            func=train_iteration_fn,
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
            reduce_ys_fn=self.reduce_metrics_fn,
        )

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        warmup_length = max(1, self.warmup_steps // self.num_envs)
        warmup_state, warmup_trajectory = self._collect_rollout(
            agent, (env_state, obsv, key), env, length=warmup_length
        )
        agent = agent.update_normalizer(warmup_trajectory)

        runner_state = (agent, *warmup_state)
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
        agent: PPOAgent = runner_state[0]
        rollout_state = runner_state[1:]
        (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
            agent, rollout_state, env
        )
        metric = trajectory_batch.info or {}
        trajectory_batch = replace(trajectory_batch, info=None)

        # Normalize the train_batch before updating the normalizer so stats are the same as during rollout
        train_batch = trajectory_batch.normalize(agent.normalizer)
        agent = agent.update_normalizer(trajectory_batch)

        # Calculate GAE and returns, add to trajectory batch
        _, (advantages, returns) = (
            train_batch.scan(  # We can use a normal scan, but this custom scan automatically handles multi-agent scenarios
                lambda gae, transition: self._compute_gae_scan(gae, transition),
                jnp.zeros(self.num_envs),
                reverse=True,
                unroll=8,
            )
        )
        train_batch = replace(train_batch, advantage=advantages, return_=returns)

        # (num_steps * num_envs, ...) > (batch_size, ...)
        train_batch = jax.tree.map(
            lambda x: x.reshape((self.batch_size,) + x.shape[2:]),
            train_batch,
        )

        # Update agent over multiple epochs x minibatches
        key, rng = jax.random.split(rng)
        agent = self._update_agent(key, agent, train_batch)

        runner_state = (agent, env_state, last_obs, rng)
        return runner_state, metric

    def _update_agent(
        self, key: PRNGKeyArray, agent: PPOAgent, train_batch: Transition
    ) -> PPOAgent:
        """`num_epochs` x `num_minibatches` gradient steps over `train_batch`."""
        agent, _ = scan_minibatch_epoch(
            lambda agent, minibatch: (agent.update_params(minibatch), None),
            agent,
            train_batch,
            minibatch_rng=key,
            num_epochs=self.num_epochs,
            num_minibatches=self.num_minibatches,
        )
        return agent

    def _collect_rollout(
        self, agent: PPOAgent, rollout_state, env: Environment, length=None
    ):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            get_action_and_log_prob = partial(agent.get_action, get_log_prob=True)

            action, log_prob = jax.vmap(get_action_and_log_prob)(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )
            value = jax.vmap(agent.get_value)(last_obs)
            next_value = jax.vmap(agent.get_value)(info[ORIGINAL_OBSERVATION_KEY])

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

        if length is None:
            length = self.num_steps

        # Do rollout
        rollout_state, trajectory_batch = jax.lax.scan(
            env_step, rollout_state, None, length
        )

        return rollout_state, trajectory_batch

    def _compute_gae_scan(self, gae, transition: Transition):
        assert transition.value is not None
        assert transition.next_value is not None

        # No bootstrap on terminated
        delta = (
            transition.reward
            + self.gamma * transition.next_value * (1 - transition.terminated)
            - transition.value
        )

        # But cut off on any done (terminated or truncated)
        done = jnp.logical_or(transition.terminated, transition.truncated)
        gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
        return gae, (gae, gae + transition.value)


class PPOAgent(RLAgent):
    trainer: PPO

    actor: ActorNetwork
    critic: ValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: PPO):
        self.trainer = trainer
        actor_key, critic_key = jax.random.split(key)
        self.actor = ActorNetwork(
            env.observation_space,
            env.action_space,
            key=actor_key,
            **trainer.actor_kwargs,
        )
        self.critic = ValueNetwork(
            env.observation_space,
            key=critic_key,
            **trainer.critic_kwargs,
        )
        self.optimizer_state = trainer.optimizer.init(
            eqx.filter((self.actor, self.critic), eqx.is_inexact_array)
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
        get_log_prob: bool = False,
    ) -> Array | tuple[Array, Array]:
        observation = self.normalizer.normalize_obs(observation)
        action_dist = self.actor(observation)
        if deterministic:
            assert not get_log_prob, "Cannot get log prob in deterministic mode"
            return action_dist.mode()
        if get_log_prob:
            return action_dist.sample_and_log_prob(seed=key)  # type: ignore
        return action_dist.sample(seed=key)

    def get_value(self, observation: PyTree):
        observation = self.normalizer.normalize_obs(observation)
        return self.critic(observation)

    def update_normalizer(self, batch: Transition):
        return self.replace(normalizer=self.normalizer.update(batch))

    def normalize_observation(self, observations: PyTree):
        return self.normalizer.normalize_obs(observations)

    def normalize_reward(self, rewards: PyTree):
        return self.normalizer.normalize_reward(rewards)

    def update_params(self, batch: Transition):
        @eqx.filter_grad
        def __ppo_loss_fn(
            params: tuple[ActorNetwork, ValueNetwork],
            train_batch: Transition,
        ):
            assert train_batch.advantage is not None
            assert train_batch.return_ is not None
            assert train_batch.log_prob is not None
            assert train_batch.value is not None

            actor, critic = params
            action_dist = jax.vmap(actor)(train_batch.observation)
            log_prob = action_dist.log_prob(train_batch.action)
            entropy = action_dist.entropy()
            value = jax.vmap(critic)(train_batch.observation)
            init_log_prob = train_batch.log_prob

            ratio = jnp.exp(log_prob - init_log_prob)
            _advantages = (train_batch.advantage - train_batch.advantage.mean()) / (
                train_batch.advantage.std() + 1e-8
            )
            actor_loss1 = _advantages * ratio

            actor_loss2 = (
                jnp.clip(ratio, 1.0 - trainer.clip_coef, 1.0 + trainer.clip_coef)
                * _advantages
            )
            actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

            # critic loss
            value_pred_clipped = train_batch.value + (
                jnp.clip(
                    value - train_batch.value,
                    -trainer.clip_coef_vf,
                    trainer.clip_coef_vf,
                )
            )
            value_losses = jnp.square(value - train_batch.return_)
            value_losses_clipped = jnp.square(value_pred_clipped - train_batch.return_)
            value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

            update_count = jym.tree.get_first(self.optimizer_state, "count")
            ent_coef = trainer.ent_coef_schedule(update_count)

            # Total loss
            total_loss = (
                actor_loss + trainer.vf_coef * value_loss - ent_coef * entropy.mean()
            )
            return total_loss  # , (actor_loss, value_loss, entropy)

        trainer = self.trainer

        actor, critic = self.actor, self.critic
        grads = __ppo_loss_fn((actor, critic), batch)
        updates, optimizer_state = trainer.optimizer.update(
            grads,
            self.optimizer_state,
            eqx.filter((actor, critic), eqx.is_inexact_array),
        )
        new_actor, new_critic = eqx.apply_updates((actor, critic), updates)
        return self.replace(
            actor=new_actor, critic=new_critic, optimizer_state=optimizer_state
        )
