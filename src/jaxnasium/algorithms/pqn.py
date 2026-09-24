from __future__ import annotations

import logging
from collections import namedtuple
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
    EpsilonGreedy,
    Normalizer,
    Schedule,
    Transition,
    scan_callback,
    scan_minibatch_epoch,
)

from .agent_networks import QValueNetwork

logger = logging.getLogger(__name__)

# differentiates tuple carry from multi agent tuple agents
QLambdaCarry = namedtuple("QLambdaCarry", ["next_return", "q_lambda"])


class PQN(RLAlgorithm):
    """Parallel Q-Network (PQN) algorithm implementation."""

    learning_rate_start: float = 2.5e-4
    learning_rate_end: float | None = eqx.field(static=True, default=None)
    epsilon_start: float = 1.0
    epsilon_end: float | None = eqx.field(static=True, default=0.05)
    exploration_fraction: float = eqx.field(static=True, default=0.5)
    """ Fraction of `total_timesteps` over which epsilon anneals from start to end. """
    gamma: float = 0.99
    max_grad_norm: float | None = 10.0
    q_lambda: float = 0.65

    total_timesteps: int = eqx.field(static=True, default=int(1e6))
    num_envs: int = eqx.field(static=True, default=12)
    num_steps: int = eqx.field(static=True, default=128)  # steps per environment
    num_minibatches: int = eqx.field(static=True, default=4)  # Number of mini-batches
    num_epochs: int = eqx.field(static=True, default=4)  # K epochs
    warmup_steps: int = eqx.field(static=True, default=5_000)
    """Normalizer statistics warmup steps."""

    normalize_observations: bool = eqx.field(static=True, default=True)
    normalize_rewards: bool = eqx.field(static=True, default=True)

    critic_kwargs: dict[str, Any] = eqx.field(static=True, default_factory=dict)

    @property
    def learning_rate_schedule(self):
        return Schedule(
            start=self.learning_rate_start,
            end=self.learning_rate_end,
            transition_steps=self.num_training_updates,
        )

    @property
    def epsilon_schedule(self):
        """Annealed over `exploration_fraction` of the gradient-step budget."""
        return Schedule(
            start=self.epsilon_start,
            end=self.epsilon_end,
            transition_steps=max(
                1, int(self.exploration_fraction * self.num_training_updates)
            ),
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
    def init_agent(self, key: PRNGKeyArray, env: Environment) -> PQNAgent:
        return PQNAgent(key=key, env=env, trainer=self)

    @eqx.filter_jit
    def train(
        self, key: PRNGKeyArray, env: Environment, agent: PQNAgent | None = None
    ) -> tuple[PQNAgent, PyTree[Float[Array, " num_iterations"]]]:
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
        agent: PQNAgent = runner_state[0]
        rollout_state = runner_state[1:]
        (env_state, last_obs, rng), trajectory_batch = self._collect_rollout(
            agent, rollout_state, env
        )
        metric = trajectory_batch.info or {}
        trajectory_batch = replace(trajectory_batch, info=None)

        # Normalize the train_batch before updating the normalizer so stats are the same as during rollout
        train_batch = trajectory_batch.normalize(agent.normalizer)
        agent = agent.update_normalizer(trajectory_batch)

        # Calculate Qlambda returns, add to trajectory batch
        # we pass a q_lambda of 0.0 on the first iteration
        carry = QLambdaCarry(jnp.zeros(self.num_envs), 0.0)
        _, returns = (
            train_batch.scan(  # We can use a normal scan, but this custom scan automatically handles multi-agent scenarios
                lambda re, transition: self._compute_q_lambda_scan(re, transition),
                carry,
                reverse=True,
                unroll=8,
            )
        )
        train_batch = replace(train_batch, return_=returns)

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
        self, key: PRNGKeyArray, agent: PQNAgent, train_batch: Transition
    ) -> PQNAgent:
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
        self, agent: PQNAgent, rollout_state, env: Environment, length=None
    ):
        def env_step(rollout_state, _):
            env_state, last_obs, rng = rollout_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            # select an action
            sample_key = jax.random.split(sample_key, self.num_envs)
            update_count = jym.tree.get_first(agent.optimizer_state, "count")
            current_epsilon = self.epsilon_schedule(update_count)
            get_action = partial(agent.get_action, epsilon=current_epsilon)
            action = jax.vmap(get_action)(sample_key, last_obs)

            # take a step in the environment
            step_key = jax.random.split(step_key, self.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = env.step(
                step_key, env_state, action
            )

            next_q_value = jax.vmap(agent.get_value)(info[ORIGINAL_OBSERVATION_KEY])

            # Build a single transition. jax.lax.scan builds a batch of transitions.
            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                next_value=next_q_value,
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

    def _compute_q_lambda_scan(self, next_return_and_lambda, transition: Transition):
        # q_lambda is 0.0 on the first iteration, afterwards it is self.q_lambda
        next_return, q_lambda = next_return_and_lambda

        next_q_values = jym.tree.batch_mean(
            jax.tree.map(lambda q: jnp.max(q, axis=-1), transition.next_value)
        )

        # bootstrap only on truncated or non-terminal
        bootstrap = (1 - transition.terminated) * next_q_values
        blended = q_lambda * next_return + (1 - q_lambda) * bootstrap

        done = jnp.logical_or(transition.terminated, transition.truncated)
        return_this_step = transition.reward + (
            self.gamma * (((1 - done) * blended) + (done * bootstrap))
        )

        return QLambdaCarry(return_this_step, self.q_lambda), return_this_step


class PQNAgent(RLAgent):
    trainer: PQN

    critic: QValueNetwork
    optimizer_state: optax.OptState
    normalizer: Normalizer

    def __init__(self, key, env: Environment, trainer: PQN):
        self.trainer = trainer
        self.critic = QValueNetwork(
            env.observation_space,
            env.action_space,
            key=key,
            **trainer.critic_kwargs,
        )

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
            q_loss = optax.losses.squared_error(q_taken, train_batch.return_)
            return jym.tree.mean(q_loss)

        trainer = self.trainer

        grads = __dqn_loss(self.critic, batch)
        updates, optimizer_state = trainer.optimizer.update(
            grads, self.optimizer_state, eqx.filter(self.critic, eqx.is_inexact_array)
        )
        new_critic = eqx.apply_updates(self.critic, updates)

        return self.replace(critic=new_critic, optimizer_state=optimizer_state)
