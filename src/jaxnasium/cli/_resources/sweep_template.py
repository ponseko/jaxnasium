import argparse
from pathlib import Path

import jax

import jaxnasium as jym
from jaxnasium.algorithms import DQN, PPO
from jaxnasium.eval import AlgorithmEvaluation, GridSearch, RandomSearch, Sweep

# jym.enable_compilation_cache() # optional

OUTPUT_DIR = Path("results/sweep")
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
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Standard train + evaluation.
    evaluation = AlgorithmEvaluation(
        jax.random.split(jax.random.key(args.seed), args.num_seeds),
        batch_size=5,  # vmap 5 seeds
        num_evaluations=50,
        return_train_metrics=True,
        save_path=OUTPUT_DIR,
    )

    sweep = Sweep(
        evaluation,  # may use any other function
        GridSearch({"env": ["CartPole-v1", "Acrobot-v1"]}),
        GridSearch(
            {
                "algorithm": {
                    "PPO": [
                        {"algorithm": PPO(**DEFAULTS)},
                        RandomSearch(  # Only sampled for PPO
                            {"ent_coef_start": (1e-3, 1e-1, "log")},
                            num_samples=4,
                            seed=SEARCH_KEY,
                        ),
                    ],
                    "DQN": [
                        {"algorithm": DQN(**DEFAULTS)},
                        RandomSearch(  # Only sampled for DQN
                            {"tau": (1e-3, 5e-2, "log")},
                            num_samples=4,
                            seed=SEARCH_KEY,
                        ),
                    ],
                }
            }
        ),
        RandomSearch(  # Sampled for all algorithms
            {"learning_rate_start": (1e-4, 3e-3, "log"), "gamma": (0.95, 0.999)},
            num_samples=4,
            seed=SEARCH_KEY,
        ),
    )

    if args.index is None:
        print(f"Number of jobs: {len(sweep)}")
        print("Run one with: python sweep.py <index>")
        return

    result = sweep[args.index]()
    print(result.arguments)
    print("mean return:", float(result.result["evaluation"].mean()))


if __name__ == "__main__":
    main()
