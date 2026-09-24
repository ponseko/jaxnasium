# CLI

The `jaxnasium` command bootstraps a new project and can copy algorithm or training
templates into an existing one. It is provided by
[create-rl-app](https://github.com/ponseko/create-rl-app); Jaxnasium registers the console script and the files that `add` can copy.

The easiest way to run it without installing anything first is [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uvx jaxnasium init <projectname>
# or, equivalently:
uvx jaxnasium <projectname>
```

## `init`

Creates a new project directory: a `uv` package, a `train.py`, and (optionally) an
environment template and a local copy of an algorithm.

```bash
jaxnasium init my-project
jaxnasium init my-project -y
jaxnasium init my-project --algorithm sac --no-env-template --environment Pendulum-v1
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-y` / `--yes` | off | Accept every default without prompting |
| `--env-template` / `--no-env-template` | prompt (`yes`) | Copy a custom `Environment` template into the package |
| `--algorithm-source` / `--no-algorithm-source` | prompt (`no`) | Copy the algorithm source into the project instead of importing it |
| `--algorithm {ppo,sac,dqn,pqn}` | `ppo` | Algorithm wired into `train.py` |
| `--environment <id>` | `CartPole-v1` | Environment id for `jym.make(...)` when no env template is included |

## `add`

Copies a registered file or bundle into the current directory (or into a path you pass).
Use this inside an existing project.

```bash
jaxnasium add ppo
jaxnasium add sweep
jaxnasium add mlp architectures/mlp.py
```

| Name | What it copies |
| --- | --- |
| `starter` | `train.py` and `example_env.py` |
| `ppo` / `sac` / `dqn` / `pqn` | The algorithm file, plus `agent_networks.py` if it is not already there |
| `sweep` | `sweep.py`, a multi-environment / multi-algorithm search |
| `bronet` / `cnn` / `mlp` | The matching architecture module |

The registry lives in `src/jaxnasium/create-rl-app.toml` if you want to see or extend what `add` can copy.
