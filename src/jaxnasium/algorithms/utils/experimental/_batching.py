import itertools
import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


def create_batched_grid_search(
    fn: Callable,
    *,
    vmap_args: dict[str, list | tuple | np.ndarray | jnp.ndarray] | None = None,
    static_args: dict[str, list | tuple | np.ndarray | jnp.ndarray] | None = None,
    max_vmap_chunk_size: int | None = 50,
):
    """Takes a function along with sequences of its arguments and returns a
    function that can be used to process all combinations of the arguments. For instance for
    sweeping over a range of hyperparameters or environment configurations.

    It supports two types of parameters:
    - `vmap_args`: Parameters that are vectorized over (batched together)
    - `static_args`: Parameters that create separate job instances (Cartesian product)

    The function can additionally chunk large vmap ranges to avoid memory issues.

    **Arguments**:
        `fn`: The function to be batched. Must accept keyword arguments.
        `vmap_args`: Dictionary of parameters to be vectorized. Values can be lists, tuples,
            or numpy/jax arrays. These will be processed in parallel using JAX vmap.
        `static_args`: Dictionary of parameters for which all combinations will be created.
            Values can be lists, tuples, or numpy/jax arrays. Creates a Cartesian product
            of all static parameter combinations.
        `max_vmap_chunk_size`: Maximum size of chunks for vmap operations. If None, no chunking
            is applied. Defaults to 50. Lower values decrease memory usage, but increases the
            number of jobs.

    **Returns**:
        `vmapped_fn`: A function that takes a tuple of (vmap_args_dict, static_args_dict)
            and returns the vmapped result.
        `pos_and_kw_fn_args`: List of all argument combinations that can be indexed and
            passed to `vmapped_fn`.

    **Example**:
    ```python
    >>> def train_model(learning_rate, batch_size, epochs, data):
    ...     # Your training logic here
    ...     return trained_model

    >>> vmapped_fn, argument_batches = create_batched_grid_search(
    ...     train_model,
    ...     vmap_args={"learning_rate": np.linspace(0.001, 0.003, 10)},
    ...     static_args={"batch_size": [32, 64], "epochs": [10, 20, 50]}
    ... )
    ... len(argument_batches) # 6 (2 batch sizes * 3 epochs --> Learning rates are vectorized)

    >>> # Process all combinations
    >>> for i, arg_combo in enumerate(args):
    ...     results: list = vmapped_fn(arg_combo)
    ...     print(f"Job {i}: Arguments used: result[i][1], fn output: {result[i][0]}")

    >>> # Or in a slurm job array
    ... #SBATCH --array=0-5
    ... python process_single_job.py --batch_idx $SLURM_ARRAY_TASK_ID
    ```
    """
    assert vmap_args is not None or static_args is not None, (
        "Either vmap_args or static_args must be provided"
    )

    # Handle empty or None cases
    if vmap_args is None:
        vmap_args = {}
    if static_args is None:
        static_args = {}

    # Convert vmap args to np.arrays and static args to lists
    if vmap_args:
        try:
            vmap_args = {k: np.atleast_1d(v) for k, v in vmap_args.items()}
        except AttributeError:
            raise ValueError(
                f"Vmap args must be a list, tuple or np/jnp array, got {vmap_args}"
            )

    if static_args:
        try:
            static_args = {
                k: np.atleast_1d(v).tolist() if not isinstance(v, list) else v
                for k, v in static_args.items()
            }
        except AttributeError:
            raise ValueError(
                f"Static args must be a list, tuple or np/jnp array, got {static_args}"
            )

    # Create vmap_grid with parameter names preserved
    vmap_param_names = list(vmap_args.keys())

    if vmap_args:
        vmap_grid = np.meshgrid(*vmap_args.values(), indexing="ij")
        vmap_args = {k: v.flatten() for k, v in zip(vmap_param_names, vmap_grid)}

        # vmap args are now flattened and each param is of size (X,), for example (500,).
        # We want to chunk these into chunks of size max_vmap_chunk_size.
        skip = 0
        chunks = []
        vmap_full_size = len(list(vmap_args.values())[0])
        while max_vmap_chunk_size is not None and skip < vmap_full_size:
            chunk_dict = {}
            for k, v in vmap_args.items():
                chunk_dict[k] = v[skip : skip + max_vmap_chunk_size]
            chunks.append(chunk_dict)
            skip += max_vmap_chunk_size

        if len(chunks) == 0:
            vmap_arg_chunks = [vmap_args]
        else:
            vmap_arg_chunks = chunks
    else:
        # No vmap args, create a single empty chunk
        vmap_arg_chunks = [{}]

    # Create static combinations with parameter names preserved
    static_param_names = list(static_args.keys())
    if static_args:
        static_combinations = [
            {name: value for name, value in zip(static_param_names, combination)}
            for combination in itertools.product(*static_args.values())
        ]
    else:
        # No static args, create a single empty combination
        static_combinations = [{}]

    def vmapped_fn(pos_args_and_kw_args: tuple):
        def fn_positional(*args, **kwargs):
            positional_args = {k: v for k, v in zip(vmap_param_names, args)}
            return fn(**positional_args, **kwargs)

        pos_args, kw_args = pos_args_and_kw_args

        if vmap_param_names:
            # We have vmap parameters, use jax.vmap
            res = jax.vmap(lambda *args: fn_positional(*args, **kw_args))(
                *pos_args.values()
            )
            return [
                (
                    res[i],
                    {**{k: pos_args[k][i] for k, v in pos_args.items()}, **kw_args},
                )
                for i in range(res.shape[0])
            ]
        else:
            # No vmap parameters, just call the function directly
            res = fn(**kw_args)
            return [(res, kw_args)]

    # Create a list of all combinations of arguments that can subsequently be indexed
    pos_and_kw_fn_args = list(itertools.product(vmap_arg_chunks, static_combinations))

    logger.info(f"Created {len(pos_and_kw_fn_args)} argument combinations")
    return vmapped_fn, pos_and_kw_fn_args
    # Usage:
    # vmapped_fn(pos_and_kw_fn_args[0])


def create_batched_random_search(
    fn: Callable,
    *,
    seed: PRNGKeyArray,
    num_samples: int,
    args: dict[str, list | tuple],
    max_vmap_chunk_size: int | None = None,
):
    assert args is not None and len(args) > 0, "args must be a non-empty dictionary"
    assert num_samples > 0, "num_samples must be greater than 0"
    assert max_vmap_chunk_size is None or max_vmap_chunk_size >= 0, (
        "max_vmap_chunk_size must be None or non-negative"
    )

    sampled_args = {}
    for k, v in args.items():
        seed, _ = jax.random.split(seed)
        if (
            isinstance(v, tuple)
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
        ):
            low, high = v
            if isinstance(low, int) and isinstance(high, int):
                sampled_args[k] = jax.random.randint(
                    seed, shape=(num_samples,), minval=low, maxval=high
                ).tolist()
            else:
                sampled_args[k] = jax.random.uniform(
                    seed, shape=(num_samples,), minval=low, maxval=high
                ).tolist()
        elif isinstance(v, list):
            indices = jax.random.randint(
                seed, shape=(num_samples,), minval=0, maxval=len(v)
            )
            sampled_args[k] = [v[i] for i in indices]
        else:
            raise ValueError(
                f"arg values must be a list of possible values or a tuple of (low, high), got {v}"
            )

    if max_vmap_chunk_size is not None and max_vmap_chunk_size > 0:
        try:

            def chunk_dict_of_arrays(args_dict, max_chunk_size):
                """Split a dict of arrays into a list of dicts with chunked arrays."""
                total_size = len(next(iter(args_dict.values())))
                chunks = []

                for start_idx in range(0, total_size, max_chunk_size):
                    end_idx = min(start_idx + max_chunk_size, total_size)
                    chunk = {k: v[start_idx:end_idx] for k, v in args_dict.items()}
                    chunks.append(chunk)

                return chunks

            sampled_args: dict = jax.tree.map(
                lambda x: jnp.asarray(x),
                sampled_args,
                is_leaf=lambda x: x is not sampled_args,
            )
            sampled_arg_chunks = chunk_dict_of_arrays(sampled_args, max_vmap_chunk_size)

            vmapped_fn = jax.vmap(
                lambda arg: fn(
                    **{k: v for k, v in zip(sampled_args.keys(), arg.values())}
                )
            )

            return (vmapped_fn, sampled_arg_chunks)

        except Exception as e:
            raise ValueError(
                f"max_vmap_chunk_size is set > 0, but all sampled args must be convertible to jnp arrays for chunking, got {sampled_args}"
            ) from e

    else:
        # No chunking, just return the (original) function and sampled arguments as a list of dicts
        list_sampled_args = [
            {
                **{k: sampled_args[k][i] for k in sampled_args.keys()},
            }
            for i in range(num_samples)
        ]

    return (fn, list_sampled_args)
