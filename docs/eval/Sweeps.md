# Sweeps

`jaxnasium.eval.Sweep` allows us to build a (parameter) sweep over any function by combining different
[search stages](Searches.md) into a flat list of jobs.

The primary use case here is to create any kind of evaluation function which accepts some parameters,
and then perform a hyperparameter sweep over said function. Below is a simple example, testing various learning
rates in two environments.

```python
import jax
import jaxnasium as jym
from jaxnasium.eval import GridSearch, RandomSearch, Sweep
from jaxnasium.algorithms import PPO

SEED = jax.random.key(0)

def simple_eval_function(env, learning_rate_start):
    env = jym.make(env)
    ppo = PPO(learning_rate_start=learning_rate_start)
    ppo, _metrics = ppo.train(SEED, env)
    return ppo.evaluate(SEED, env).mean()

sweep = Sweep(
    simple_eval_function,
    GridSearch({"env": ["CartPole-v1", "Acrobot-v1"]}),
    RandomSearch(
        {"learning_rate_start": (1e-4, 1e-2, "log")},
        num_samples=32,
        seed=SEED,
    ),
)

for run in sweep:
    result = run()
    print(result.arguments, result.result)
```

A standard algorithm evaluation function is provided in [`AlgorithmEvaluation`][jaxnasium.eval.AlgorithmEvaluation]. It accepts `env`, `algorithm` and any leftover keywords as hyperparameters.

Typically, one may want to launch jobs through a slurm system:


```python
parser = argparse.ArgumentParser()
parser.add_argument(
    "index",
    type=int,
    nargs="?",
    help="sweep job to run; omit to print the number of jobs",
)
args = parser.parse_args()

sweep = Sweep(...)

if args.index is None:
    print(len(sweep))
    return

result = sweep[args.index]()
print(result.arguments)
print("mean return:", float(result.result["evaluation"].mean()))
```

```bash
python sweep.py          # prints number of jobs
#SBATCH --array=0-NUM_JOBS
python sweep.py $SLURM_ARRAY_TASK_ID
```

# Full example

```python
# sweep.py
import argparse
from pathlib import Path

import jax
from jaxnasium.algorithms import PPO, SAC
from jaxnasium.eval import AlgorithmEvaluation, GridSearch, SobolSearch, Sweep

OUTPUT_DIR = Path("results/sweep")
KEY = jax.random.key(0)
SEARCH_KEY = jax.random.PRNGKey(1)

DEFAULTS = {"total_timesteps": 1_000_000, "num_envs": 8, "log_function": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index",
        type=int,
        nargs="?",
        help="sweep job to run; omit to print the number of jobs",
    )
    args = parser.parse_args()

    evaluation = AlgorithmEvaluation(
        jax.random.split(KEY, 10),
        batch_size=5,
        num_evaluations=50,
        return_train_metrics=True,
        save_path=OUTPUT_DIR,
    )

    sweep = Sweep(
        evaluation,
        GridSearch({"env": ["CartPole-v1", "Pendulum-v1"]}),
        GridSearch(
            {
                "algorithm": {
                    "PPO": [
                        {"algorithm": PPO(**DEFAULTS)},
                        SobolSearch(
                            {
                                "clip_coef": (0.1, 0.3),
                                "ent_coef_start": (1e-3, 1e-1, "log"),
                            },
                            num_samples=32,
                            seed=SEARCH_KEY,
                        ),
                    ],
                    "SAC": [
                        {"algorithm": SAC(**DEFAULTS)},
                        SobolSearch(
                            {
                                "tau": (1e-3, 5e-2, "log"),
                                "init_alpha": (0.05, 0.5),
                            },
                            num_samples=32,
                            seed=SEARCH_KEY,
                        ),
                    ],
                }
            }
        ),
        SobolSearch(
            {"learning_rate_start": (1e-4, 3e-3, "log"), "gamma": (0.95, 0.999)},
            num_samples=16,
            seed=SEARCH_KEY,
        ),
    )
    # 2 envs × (32 PPO + 32 SAC) × 16 shared draws = 2048 jobs.

    if args.index is None:
        print(len(sweep))
        return

    result = sweep[args.index]()
    print(result.arguments)
    print("mean return:", float(result.result["evaluation"].mean()))


if __name__ == "__main__":
    main()
```

```bash
python sweep.py          # prints 2048
#SBATCH --array=0-2047
python sweep.py $SLURM_ARRAY_TASK_ID
```

::: jaxnasium.eval.Sweep

::: jaxnasium.eval.SweepJob

::: jaxnasium.eval.SweepResult
    options:
        members:
            - duration
            - to_json
            - to_json_no_result

# Algorithm Evaluation

`AlgorithmEvaluation` trains a Jaxnasium algorithm on an environment and evaluates the
resulting agent, over one seed or many. It is an intended trial function for a
[`Sweep`][jaxnasium.eval.Sweep], but may be used on its own.

```python
import jax
from jaxnasium.eval import AlgorithmEvaluation

evaluation = AlgorithmEvaluation(
    seed=jax.random.split(jax.random.key(0), 10),
    env="CartPole-v1",
    algorithm="PPO",
    batch_size=5,
    save_path="results/",
)
returns = evaluation(learning_rate_start=3e-4)  # shape (10, num_evaluations)
```

Any keyword arguments given to the call that aren't `seed`, `env` or `algorithm` are treated
as algorithm hyperparameters, which is what lets a sweep drive it directly.

::: jaxnasium.eval.AlgorithmEvaluation
    options:
        members:
            - __call__
            - context
            - with_labels
