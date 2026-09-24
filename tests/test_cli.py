import os
import subprocess
from pathlib import Path

import pytest

JAXNASIUM_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.cli


@pytest.fixture(scope="session")
def jaxnasium_wheel_dir() -> Path:
    """Build a local wheel so `init` can resolve unreleased jaxnasium versions."""
    dist_dir = JAXNASIUM_ROOT / "dist"
    subprocess.run(["uv", "build", "--quiet"], cwd=JAXNASIUM_ROOT, check=True)
    return dist_dir


@pytest.fixture
def cli_env(jaxnasium_wheel_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["UV_FIND_LINKS"] = str(jaxnasium_wheel_dir)
    return env


def _run_jaxnasium_add(item: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run `jaxnasium add <item>` from the jaxnasium source tree."""
    return subprocess.run(
        [
            "uvx",
            "--no-cache",
            "--from",
            str(JAXNASIUM_ROOT),
            "jaxnasium",
            "add",
            item,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_jaxnasium_uvx(tmp_path, cli_env):
    test_dir = tmp_path / "test_project"
    result = subprocess.run(
        [
            "uvx",
            "--no-cache",
            "--from",
            str(JAXNASIUM_ROOT),
            "jaxnasium",
            "init",
            str(test_dir),
            "-y",
        ],
        env=cli_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"CLI failed with: {result.stderr}\n{result.stdout}"
    expected_files = ["pyproject.toml", "train.py", "README.md"]
    for file in expected_files:
        assert (test_dir / file).exists(), f"Expected file {file} not found"


def test_jaxnasium_add_ppo(tmp_path):
    """`jaxnasium add ppo` scaffolds algorithm files into the current directory."""
    result = _run_jaxnasium_add("ppo", tmp_path)
    assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"

    assert (tmp_path / "ppo.py").is_file()
    assert (tmp_path / "agent_networks.py").is_file()


def test_jaxnasium_add_sweep(tmp_path):
    """`jaxnasium add sweep` scaffolds the sweep template."""
    result = _run_jaxnasium_add("sweep", tmp_path)
    assert result.returncode == 0, f"CLI failed: {result.stderr}\n{result.stdout}"
    assert (tmp_path / "sweep.py").is_file()


def test_jaxnasium_add_sac(tmp_path):
    """`jaxnasium add sac` works and skips agent_networks.py if already present."""
    ppo_result = _run_jaxnasium_add("ppo", tmp_path)
    assert ppo_result.returncode == 0, ppo_result.stderr

    sac_result = _run_jaxnasium_add("sac", tmp_path)
    assert sac_result.returncode == 0, (
        f"CLI failed: {sac_result.stderr}\n{sac_result.stdout}"
    )

    assert (tmp_path / "ppo.py").is_file()
    assert (tmp_path / "sac.py").is_file()
    assert (tmp_path / "agent_networks.py").is_file()
    assert "Skipped" in sac_result.stdout
