import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxnasium.eval import (
    AlgorithmEvaluation,
    GridSearch,
    OneAtATimeSearch,
    RandomSearch,
    SobolSearch,
    Sweep,
)


def test_grid_search_gives_one_job_per_config():
    def fn(env_name, task):
        return 0.0

    sweep = GridSearch({"env_name": ["a", "b"], "task": ["x", "y", "z"]}).sweep(fn)

    assert len(sweep) == 6
    result = sweep[0]()
    assert result.arguments == {"env_name": "a", "task": "x"}
    assert result.duration >= 0.0


def test_with_cost_size_estimate(capsys):
    def fn(lr, env_name):
        return lr

    _sweep = GridSearch({"env_name": ["a", "b"], "lr": [0.1, 0.2]}).sweep(
        fn, print_cost_estimate=True
    )

    # `_log` writes the estimate to stderr.
    err = capsys.readouterr().err
    assert "Estimating the cost of the first job." in err
    assert "CostEstimate('fn':" in err


def test_log_cost_estimate_returns_dataclass():
    from jaxnasium.eval._probe import CostEstimate, log_cost_estimate

    def fn(lr):
        return (jnp.ones(8) * lr).sum()

    estimate = log_cost_estimate(fn, lr=0.1)
    assert isinstance(estimate, CostEstimate)
    assert estimate.error is None
    assert estimate.device_memory_bytes is not None


def test_sweep_len_iter_and_slice():
    def fn(lr):
        return lr

    sweep = GridSearch({"lr": [0.1, 0.2, 0.3]}).sweep(fn)
    assert len(sweep) == 3

    # Iteration yields one callable per job; calling it runs that job.
    results = [run() for run in sweep]
    assert [r.result for r in results] == pytest.approx([0.1, 0.2, 0.3])
    assert [r.arguments["lr"] for r in results] == pytest.approx([0.1, 0.2, 0.3])

    # Integer index: same callable shape as iteration items.
    first = sweep[0]
    assert callable(first)
    assert first().result == pytest.approx(0.1)
    assert sweep[-1]().result == pytest.approx(0.3)

    # Slice: a new Sweep over a subset of jobs; still iterable / indexable.
    sliced = sweep[1:]
    assert isinstance(sliced, Sweep)
    assert len(sliced) == 2
    assert [run().result for run in sliced] == pytest.approx([0.2, 0.3])
    assert sliced[0]().result == pytest.approx(0.2)

    mid = sweep[1:2]
    assert len(mid) == 1
    assert mid.jobs[0] is sweep.jobs[1]


def test_no_stages_raises():
    with pytest.raises(ValueError, match="At least one sweep stage"):
        Sweep(lambda: None)


def test_result_to_json_handles_arrays_and_pytrees():
    def fn(lr):
        return {"loss": jnp.ones(3) * lr, "steps": jnp.asarray(2)}

    sweep = GridSearch({"lr": [0.5, 1.5]}).sweep(fn)

    payload = json.loads(sweep[0]().to_json())
    assert payload["arguments"] == {"lr": 0.5}
    assert payload["result"]["loss"] == pytest.approx([0.5, 0.5, 0.5])
    assert payload["result"]["steps"] == 2
    assert "result" not in json.loads(sweep[0]().to_json_no_result())


# --- sampling searches ------------------------------------------------------


def test_random_search_samples_each_config_once():
    def fn(scale):
        return scale

    sweep = RandomSearch(
        {"scale": (0.5, 2.0)}, num_samples=6, seed=jax.random.PRNGKey(0)
    ).sweep(fn)

    assert len(sweep) == 6
    sampled = [job.args["scale"] for job in sweep.jobs]
    assert len(set(sampled)) == 6
    assert all(0.5 <= scale <= 2.0 for scale in sampled)
    assert sweep[0]().result == pytest.approx(sampled[0])


def test_sampling_search_requires_its_own_seed():
    with pytest.raises(TypeError, match="seed"):
        RandomSearch({"scale": (0.5, 2.0)}, num_samples=2)  # type: ignore[call-arg]


def test_search_seed_accepts_a_plain_int():
    assert (
        RandomSearch({"scale": (0.5, 2.0)}, num_samples=4, seed=0).configs()
        == RandomSearch(
            {"scale": (0.5, 2.0)}, num_samples=4, seed=jax.random.PRNGKey(0)
        ).configs()
    )


def test_linear_and_log_ranges_sample_differently_and_correctly():
    low, high, mid = 1e-4, 1e-2, 1e-3

    key = jax.random.PRNGKey(0)
    linear = RandomSearch({"lr": (low, high)}, num_samples=20_000, seed=key).configs()
    log = RandomSearch(
        {"lr": (low, high, "log")}, num_samples=20_000, seed=key
    ).configs()

    linear_above = sum(c["lr"] >= mid for c in linear) / len(linear)
    log_above = sum(c["lr"] >= mid for c in log) / len(log)

    assert linear_above == pytest.approx(0.909, abs=0.02)
    assert log_above == pytest.approx(0.5, abs=0.02)
    assert linear_above > log_above + 0.3


def test_grid_then_random_shares_its_draws():
    """A search draws once for the whole sweep, so every grid cell evaluates the
    same values — which is what makes the cells comparable."""

    sweep = Sweep(
        lambda env_name, lr: lr,
        GridSearch({"env_name": ["a", "b"]}),
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=4, seed=jax.random.PRNGKey(7)),
    )

    assert len(sweep) == 8
    assert [job.args["env_name"] for job in sweep.jobs] == ["a"] * 4 + ["b"] * 4

    results = [sweep.fn(job) for job in sweep.jobs]
    lrs_a = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "a"]
    lrs_b = [r.arguments["lr"] for r in results if r.arguments["env_name"] == "b"]
    assert lrs_a == lrs_b
    assert len(set(lrs_a)) == 4


def test_two_sampling_searches_are_independent_but_both_shared():
    def fn(env_name, lr, dropout):
        return lr * dropout

    sweep = Sweep(
        fn,
        GridSearch({"env_name": ["a", "b"]}),
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=2, seed=jax.random.PRNGKey(1)),
        RandomSearch(
            {"dropout": (0.0, 0.5)}, num_samples=2, seed=jax.random.PRNGKey(2)
        ),
    )

    # 2 envs * 2 lrs * 2 dropouts = 8 configurations.
    assert len(sweep) == 8
    by_env = {"a": [], "b": []}
    for job in sweep.jobs:
        by_env[job.args["env_name"]].append((job.args["lr"], job.args["dropout"]))

    # Both searches own their seed, so both envs see the same pairs.
    assert by_env["a"] == by_env["b"]
    assert len({lr for lr, _ in by_env["a"]}) == 2
    assert len({d for _, d in by_env["a"]}) == 2


def test_random_then_random_pairs_every_combination():
    def fn(lr, gamma):
        return lr * gamma

    sweep = Sweep(
        fn,
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=3, seed=jax.random.PRNGKey(0)),
        RandomSearch({"gamma": (0.9, 1.0)}, num_samples=2, seed=jax.random.PRNGKey(5)),
    )

    # Each of the 3 learning rates is paired with both of the 2 gammas.
    assert len(sweep) == 6
    lrs = [job.args["lr"] for job in sweep.jobs]
    gammas = [job.args["gamma"] for job in sweep.jobs]
    assert len(set(lrs)) == 3
    assert lrs[0] == lrs[1] and lrs[2] == lrs[3] and lrs[4] == lrs[5]
    assert len(set(gammas)) == 2
    assert gammas[0:2] == gammas[2:4] == gammas[4:6]


def test_random_then_grid_chains():
    def fn(lr, env_name):
        return lr

    sweep = Sweep(
        fn,
        RandomSearch({"lr": (0.0, 1.0)}, num_samples=3, seed=jax.random.PRNGKey(0)),
        GridSearch({"env_name": ["a", "b"]}),
    )

    # The grid comes last, so it is repeated within every sampled lr.
    assert len(sweep) == 6
    assert [job.args["env_name"] for job in sweep.jobs] == list("ababab")


def test_param_swept_by_two_searches_raises():
    def fn(lr):
        return lr

    with pytest.raises(ValueError, match="more than one stage"):
        Sweep(
            fn,
            GridSearch({"lr": [0.1, 0.2]}),
            RandomSearch({"lr": (0.0, 1.0)}, num_samples=2, seed=jax.random.PRNGKey(0)),
        )


def test_sobol_search():
    def fn(lr, layers):
        return lr

    def sample(seed):
        sweep = SobolSearch(
            {"lr": (0.0, 1.0), "layers": [16, 32, 64, 128]}, num_samples=4, seed=seed
        ).sweep(fn)
        return [job.args for job in sweep.jobs]

    assert sample(jax.random.PRNGKey(0)) == sample(jax.random.PRNGKey(0))
    assert sample(jax.random.PRNGKey(0)) != sample(jax.random.PRNGKey(1))
    assert {c["layers"] for c in sample(jax.random.PRNGKey(0))} <= {16, 32, 64, 128}


# --- branches ---------------------------------------------------------------


def test_branch_records_the_label_and_calls_the_sub_space():
    def fn(algorithm, lr=None, tau=None):
        return algorithm

    sweep = GridSearch(
        {
            "algorithm": {
                "PPO": [{"algorithm": "<ppo>"}, GridSearch({"lr": [0.1, 0.2]})],
                "SAC": [{"algorithm": "<sac>"}, GridSearch({"tau": [0.01]})],
            }
        }
    ).sweep(fn)

    assert len(sweep) == 3
    # The label is what lands in the results; the sub-space is what gets called.
    assert [job.args["algorithm"] for job in sweep.jobs] == ["<ppo>", "<ppo>", "<sac>"]
    assert [job.arguments["algorithm"] for job in sweep.jobs] == ["PPO", "PPO", "SAC"]

    results = [run() for run in sweep]
    assert [r.result for r in results] == ["<ppo>", "<ppo>", "<sac>"]
    assert [r.arguments for r in results] == [
        {"algorithm": "PPO", "lr": 0.1},
        {"algorithm": "PPO", "lr": 0.2},
        {"algorithm": "SAC", "tau": 0.01},
    ]


def test_branch_of_bare_values():
    """A branch value that is not a sub-space is the value for the parameter."""

    def fn(algorithm):
        return algorithm

    sweep = GridSearch({"algorithm": {"PPO": "<ppo>", "SAC": "<sac>"}}).sweep(fn)

    assert [job.args for job in sweep.jobs] == [
        {"algorithm": "<ppo>"},
        {"algorithm": "<sac>"},
    ]
    assert [run().arguments["algorithm"] for run in sweep] == ["PPO", "SAC"]


def test_branch_adds_arguments_under_other_names():
    """One tag in the results, several keyword arguments in the call."""

    sweep = GridSearch(
        {
            "network": {
                "mlp": {"actor_kwargs": {"body": "MLP"}, "critic_kwargs": {}},
                "simba": {"actor_kwargs": {"body": "SimBa"}, "critic_kwargs": {}},
            }
        }
    ).sweep(lambda actor_kwargs, critic_kwargs: actor_kwargs["body"])

    results = [run() for run in sweep]
    assert [r.result for r in results] == ["MLP", "SimBa"]
    assert [r.arguments["network"] for r in results] == ["mlp", "simba"]


def test_sibling_branches_may_reuse_a_param_name():
    sweep = GridSearch(
        {
            "algorithm": {
                "PPO": GridSearch({"lr": [0.1]}),
                "SAC": GridSearch({"lr": [0.2, 0.3]}),
            }
        }
    ).sweep(lambda algorithm, lr: lr)

    assert [(job.arguments["algorithm"], job.args["lr"]) for job in sweep.jobs] == [
        ("PPO", 0.1),
        ("SAC", 0.2),
        ("SAC", 0.3),
    ]


def test_a_branch_colliding_with_an_outer_stage_raises():
    with pytest.raises(ValueError, match="more than one stage"):
        Sweep(
            lambda lr, algorithm: lr,
            GridSearch({"lr": [0.1]}),
            GridSearch({"algorithm": {"PPO": {"lr": 0.9}}}),
        )


def test_branches_nest():
    sweep = GridSearch(
        {
            "algorithm": {
                "SAC": [
                    {"algorithm": "<sac>"},
                    OneAtATimeSearch(
                        {
                            "network": {
                                "mlp": {"body": "MLP"},
                                "simba": {"body": "SimBa"},
                            },
                            "n_step": [1, 3],
                        }
                    ),
                ]
            }
        }
    ).sweep(lambda algorithm, body, n_step: (body, n_step))

    # OFAT baseline, then one config per non-baseline value.
    assert [
        (job.arguments["network"], job.args["body"], job.args["n_step"])
        for job in sweep.jobs
    ] == [("mlp", "MLP", 1), ("simba", "SimBa", 1), ("mlp", "MLP", 3)]
    assert all(job.arguments["algorithm"] == "SAC" for job in sweep.jobs)


def test_branch_labels_can_be_sampled():
    sweep = SobolSearch(
        {"algorithm": {"PPO": {"lr": 0.1}, "SAC": {"lr": 0.2}}},
        num_samples=8,
        seed=jax.random.PRNGKey(0),
    ).sweep(lambda lr: lr)

    labels = [job.labels["algorithm"] for job in sweep.jobs]
    assert set(labels) == {"PPO", "SAC"}
    assert all(
        job.args["lr"] == (0.1 if job.labels["algorithm"] == "PPO" else 0.2)
        for job in sweep.jobs
    )


# --- one-at-a-time ----------------------------------------------------------


def test_one_at_a_time_search_configs():
    search = OneAtATimeSearch(
        {
            "gamma": [0.99, 0.95, 0.999],
            "lr": [0.1, 0.01],
        }
    )
    assert search.configs() == [
        {"gamma": 0.99, "lr": 0.1},
        {"gamma": 0.95, "lr": 0.1},
        {"gamma": 0.999, "lr": 0.1},
        {"gamma": 0.99, "lr": 0.01},
    ]


def test_one_at_a_time_search_fewer_configs_than_grid():
    params = {"a": [1, 2, 3], "b": ["x", "y"]}
    oat = OneAtATimeSearch(params).configs()
    grid = GridSearch(params).configs()
    assert len(oat) == 1 + 2 + 1
    assert len(grid) == 3 * 2
    assert all(c in grid for c in oat)


def test_one_at_a_time_search_sweep():
    def fn(gamma, lr):
        return gamma * lr

    sweep = OneAtATimeSearch({"gamma": [0.99, 0.95], "lr": [0.1, 0.01]}).sweep(fn)

    assert len(sweep) == 3
    results = [run() for run in sweep]
    assert [r.arguments for r in results] == [
        {"gamma": 0.99, "lr": 0.1},
        {"gamma": 0.95, "lr": 0.1},
        {"gamma": 0.99, "lr": 0.01},
    ]
    assert [r.result for r in results] == pytest.approx([0.099, 0.095, 0.0099])


def test_one_at_a_time_search_single_valued_params():
    assert OneAtATimeSearch({"a": [1], "b": [2]}).configs() == [{"a": 1, "b": 2}]


def test_grid_then_one_at_a_time_chains():
    def fn(env_name, gamma, lr):
        return lr

    sweep = Sweep(
        fn,
        GridSearch({"env_name": ["a", "b"]}),
        OneAtATimeSearch({"gamma": [0.99, 0.95], "lr": [0.1, 0.01]}),
    )

    # The same 3 one-at-a-time configurations are repeated within each env.
    results = [run() for run in sweep]
    assert len(results) == 6
    per_env = {
        env: sorted(
            (r.arguments["gamma"], r.arguments["lr"])
            for r in results
            if r.arguments["env_name"] == env
        )
        for env in ("a", "b")
    }
    assert per_env["a"] == per_env["b"]
    assert per_env["a"] == [(0.95, 0.1), (0.99, 0.01), (0.99, 0.1)]


# --- seeds, which belong to the trial function ------------------------------


@pytest.mark.parametrize(
    ("seed", "shape"),
    [
        (jax.random.key(0), ()),
        (jax.random.PRNGKey(0), ()),
        (jax.random.fold_in(jax.random.key(0), 5), ()),
        (jax.random.fold_in(jax.random.PRNGKey(0), 5), ()),
        (jax.random.split(jax.random.key(0), 3), (3,)),
        (jax.random.split(jax.random.PRNGKey(0), 3), (3,)),
    ],
)
def test_both_key_styles_normalise_to_the_same_shape(seed, shape):
    """A legacy PRNGKey is raw uint32 data, so it is wrapped up front — after
    which `ndim == 0` means one run for either style."""
    config = AlgorithmEvaluation(seed)._create_config({})
    assert jnp.issubdtype(config.seed.dtype, jax.dtypes.prng_key)
    assert config.seed.shape == shape


def test_int_seeds_are_rejected():
    with pytest.raises(AttributeError):
        AlgorithmEvaluation(0)._create_config({})  # type: ignore[arg-type]


def test_algorithm_evaluation_seed_from_kwargs():
    seed = jax.random.key(0)
    config = AlgorithmEvaluation()._create_config({"seed": seed})
    assert jnp.array_equal(config.seed, seed)


def test_algorithm_evaluation_seed_required_at_run():
    with pytest.raises(ValueError, match="No seed was provided to"):
        AlgorithmEvaluation()._create_config({})


def test_algorithm_evaluation_seed_cannot_be_given_twice():
    with pytest.raises(ValueError, match="but its run was also given a seed"):
        AlgorithmEvaluation(jax.random.key(0))._create_config(
            {"seed": jax.random.key(1)}
        )


def _tiny_algorithm():
    from jaxnasium.algorithms import PPO
    from jaxnasium.algorithms.architectures import MLP

    body = {"body": MLP.with_params(hidden_sizes=(8,))}
    return PPO(
        total_timesteps=512,
        num_envs=1,
        num_steps=128,
        num_minibatches=1,
        num_epochs=1,
        warmup_steps=16,
        log_function=None,
        normalize_observations=False,
        normalize_rewards=False,
        actor_kwargs=body,
        critic_kwargs=body,
    )


def test_algorithm_evaluation_single_seed_has_no_leading_axis():
    result = AlgorithmEvaluation(
        jax.random.key(0), algorithm=_tiny_algorithm(), num_evaluations=2
    )()
    assert result.shape == (2,)


@pytest.mark.parametrize("batch_size", [None, 2, 5])
def test_algorithm_evaluation_stacks_several_seeds(batch_size):
    """`batch_size` is the lax.map batch size, so it changes how the seeds run,
    never what comes out — including when it exceeds the number of seeds."""
    result = AlgorithmEvaluation(
        jax.random.split(jax.random.key(0), 3),
        algorithm=_tiny_algorithm(),
        batch_size=batch_size,
        num_evaluations=2,
    )()
    assert result.shape == (3, 2)


def test_algorithm_evaluation_with_train_metrics():
    result = AlgorithmEvaluation(
        jax.random.split(jax.random.key(0), 2),
        algorithm=_tiny_algorithm(),
        batch_size=2,
        num_evaluations=2,
        return_train_metrics=True,
    )()
    assert set(result) == {"evaluation", "train_metrics"}
    assert result["evaluation"].shape[0] == 2
    assert result["train_metrics"].shape[0] == 2


def test_algorithm_evaluation_context_records_the_seeds():
    """A key array has no `tolist`, so the raw key data gets recorded — which is
    what reproduces the runs, and reads the same for either key style."""
    keys = jax.random.split(jax.random.key(0), 3)
    context = AlgorithmEvaluation(keys, algorithm=_tiny_algorithm()).context()
    assert context["seed"] == jax.random.key_data(keys).tolist()
    assert context["algorithm"] == "PPO"
    assert context["algorithm_parameters"]["num_envs"] == 1


def test_save_path_writes_the_four_files(tmp_path):
    result = AlgorithmEvaluation(
        jax.random.split(jax.random.key(0), 2),
        algorithm=_tiny_algorithm(),
        batch_size=2,
        num_evaluations=2,
        return_train_metrics=True,
        save_path=tmp_path,
    )()

    (run_dir,) = list(tmp_path.iterdir())
    # A string env records its registry id; an instance records its innermost class.
    assert run_dir.name.startswith("PPO_CartPole-v1_")
    assert sorted(f.name for f in run_dir.iterdir()) == [
        "full_parameters.json",
        "input_parameters.json",
        "results.npy",
        "train_curve.npy",
    ]

    # Arrays are seeds-first, matching what was returned.
    assert np.load(run_dir / "results.npy").shape == result["evaluation"].shape
    assert np.load(run_dir / "train_curve.npy").shape == result["train_metrics"].shape

    inputs = json.loads((run_dir / "input_parameters.json").read_text())
    full = json.loads((run_dir / "full_parameters.json").read_text())
    assert "algorithm_parameters" not in inputs  # the small one, for scanning
    assert full["algorithm_parameters"]["num_envs"] == 1
    assert full["duration"] > 0


def test_save_as_zip_writes_a_zip_instead_of_a_folder(tmp_path):
    AlgorithmEvaluation(
        jax.random.split(jax.random.key(0), 2),
        algorithm=_tiny_algorithm(),
        batch_size=2,
        num_evaluations=2,
        return_train_metrics=True,
        save_path=tmp_path,
        save_as_zip=True,
    )()

    (zip_path,) = list(tmp_path.iterdir())
    assert zip_path.is_file()
    assert zip_path.suffix == ".zip"
    assert zip_path.name.startswith("PPO_CartPole-v1_")
    assert not zip_path.with_suffix("").exists()


def test_save_path_gives_each_configuration_its_own_folder(tmp_path):
    """The folder is named after a digest of the configuration, so sweeping writes
    one folder per point, and re-running a point overwrites it."""
    sweep = Sweep(
        AlgorithmEvaluation(
            jax.random.key(0),
            algorithm=_tiny_algorithm(),
            num_evaluations=2,
            save_path=tmp_path,
        ),
        GridSearch({"gamma": [0.95, 0.99]}),
    )
    for run in sweep:
        run()
    assert len(list(tmp_path.iterdir())) == 2

    for run in sweep:  # re-running lands in the same folders
        run()
    assert len(list(tmp_path.iterdir())) == 2


def test_no_save_path_writes_nothing(tmp_path):
    AlgorithmEvaluation(
        jax.random.key(0), algorithm=_tiny_algorithm(), num_evaluations=2
    )()
    assert list(tmp_path.iterdir()) == []


# --- end to end -------------------------------------------------------------


def test_real_ppo_env_and_hparams_sweep():
    """A typical use case: sweep env and hparams, several seeds inside each job."""
    sweep = Sweep(
        AlgorithmEvaluation(
            jax.random.split(jax.random.key(0), 2),
            algorithm=_tiny_algorithm(),
            batch_size=2,
            num_evaluations=2,
        ),
        GridSearch({"env": ["CartPole-v1", "Acrobot-v1"]}),
        RandomSearch(
            {"learning_rate_start": (1e-4, 1e-2, "log"), "gamma": [0.98, 0.99]},
            num_samples=2,
            seed=jax.random.PRNGKey(1),
        ),
    )

    # 2 envs * 2 samples, each job running both seeds itself.
    assert len(sweep) == 4
    assert [job.args["env"] for job in sweep.jobs] == ["CartPole-v1"] * 2 + [
        "Acrobot-v1"
    ] * 2

    # The search owns its seed → both envs share the same (lr, gamma) draws.
    cartpole = [job for job in sweep.jobs if job.args["env"] == "CartPole-v1"]
    acrobot = [job for job in sweep.jobs if job.args["env"] == "Acrobot-v1"]
    for key in ("learning_rate_start", "gamma"):
        assert [job.args[key] for job in cartpole] == [job.args[key] for job in acrobot]

    results = [sweep.fn(job) for job in sweep.jobs]
    assert all(r.result.shape == (2, 2) for r in results)  # (seeds, evaluations)
    assert all(jnp.isfinite(r.result).all() for r in results)
    assert {r.arguments["env"] for r in results} == {"CartPole-v1", "Acrobot-v1"}
    assert all(1e-4 <= r.arguments["learning_rate_start"] <= 1e-2 for r in results)
    assert all(r.arguments["gamma"] in (0.98, 0.99) for r in results)
