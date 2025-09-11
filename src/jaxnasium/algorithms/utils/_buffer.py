import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

from ._transition import Transition


class TransitionBuffer(eqx.Module):
    """
    A buffer for storing transitions. Samples uniformly from the buffer.
    The buffer is implemented as a circular buffer, where the oldest transitions are overwritten
    when the buffer is full.

    **Arguments**:
        `max_size`: The maximum size of the buffer.
        `sample_batch_size`: The number of transitions to sample from the buffer.
        `data_sample`: A sample `Transition` to initialize the buffer structure.
        `num_batch_axes`: The number of batch axes in the transition data. Defaults to 2, which expects the first two axes
            to be batch dimensions (e.g, (`rollout_length`, `num_envs`, ...))
    """

    data: Transition
    insert_position: int
    size: int
    max_size: int = eqx.field(static=True)
    sample_batch_size: int = eqx.field(static=True)
    num_batch_axes: int = eqx.field(static=True, default=2)

    def __check_init__(self):
        assert self.sample_batch_size > 0, "sample_batch_size must be greater than 0"
        assert self.max_size > 0, "max_size must be greater than 0"
        assert self.sample_batch_size <= self.max_size, (
            "sample_batch_size must be less than or equal to max_size"
        )

    def __init__(
        self,
        max_size: int,
        sample_batch_size: int,
        data_sample: Transition,
        num_batch_axes: int = 2,
    ):
        self.insert_position = 0
        self.max_size = max_size
        self.sample_batch_size = sample_batch_size
        self.size = 0
        self.num_batch_axes = num_batch_axes

        self.data = jax.tree.map(
            lambda x: jnp.zeros(
                (self.max_size,) + x.shape[self.num_batch_axes :], dtype=x.dtype
            ),
            data_sample,
        )

    def insert(self, transition: Transition) -> "TransitionBuffer":
        """
        Insert a transition into the buffer.
        """
        data = self.data
        data_len = np.prod(jax.tree.leaves(transition)[0].shape[: self.num_batch_axes])
        insert_position = self.insert_position

        assert data_len <= self.max_size, (
            "Transition length exceeds buffer size. "
            f"Transition length: {data_len}, Buffer size: {self.max_size}"
        )

        idx = (jnp.arange(data_len) + insert_position) % self.max_size
        data = jax.tree.map(
            lambda x, y: x.at[idx].set(
                y.reshape(-1, *y.shape[self.num_batch_axes :]),
                unique_indices=True,
            ),
            data,
            transition,
        )

        insert_position = (insert_position + data_len) % self.max_size
        size = jnp.minimum(self.size + data_len, self.max_size)
        buffer = self
        buffer = eqx.tree_at(lambda x: x.data, buffer, data)
        buffer = eqx.tree_at(lambda x: x.insert_position, buffer, insert_position)
        buffer = eqx.tree_at(lambda x: x.size, buffer, size)
        return buffer

    def sample(self, key: PRNGKeyArray, with_replacement: bool = False) -> Transition:
        """
        Sample a batch of transitions from the buffer.
        """
        if with_replacement:
            idx = jax.random.randint(
                key, (self.sample_batch_size,), minval=0, maxval=self.size
            )
        else:
            valid_samples = jnp.arange(self.max_size) < self.size
            id_probs = jax.random.uniform(key, (self.max_size,))
            id_probs = jnp.where(valid_samples, id_probs, -jnp.inf)
            idx = jax.lax.top_k(id_probs, self.sample_batch_size)[1]

        batch: Transition = jax.tree.map(lambda x: jnp.take(x, idx, axis=0), self.data)
        return batch
