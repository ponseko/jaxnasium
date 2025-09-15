import os
import subprocess
import sys

import _consts as TEST_CONSTS  # noqa: F401
import pytest


@pytest.mark.skipif(
    bool(os.getenv("JYMKIT_SKIP_CLI_TEST", False)), reason="Skipping CLI test"
)
def test_init_cli_with_pipx(tmp_path, monkeypatch):
    """Test that jaxnasium works when invoked through pipx."""
    test_dir = tmp_path / "test_project"

    # try with uvx first
    try:
        subprocess.run(["uvx", ".", test_dir, "-y"])
    except subprocess.SubprocessError:
        # Check if pipx is available
        print("uvx not available, trying pipx")
        try:
            subprocess.run(["pipx", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            pytest.skip("pipx not available")

        # Run through pipx
        # Run through pipx - store the result for checking
        result = subprocess.run(
            ["pipx", "run", "--no-cache", "--spec", ".", "jaxnasium", test_dir, "-y"],
            capture_output=True,
            text=True,
        )

        # Check the command executed successfully
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

    # Verify the directory was created
    assert test_dir.exists(), "Project directory wasn't created"

    # Check for expected files/folders in the structure
    expected_files = ["pyproject.toml", "train_example.py", "README.md"]
    expected_dirs = ["test_project"]

    for file in expected_files:
        assert (test_dir / file).exists(), f"Expected file {file} not found"

    for directory in expected_dirs:
        assert (test_dir / directory).is_dir(), (
            f"Expected directory {directory} not found"
        )

    # Test running a file from the created structure
    test_file = "train_example.py"
    assert (test_dir / test_file).exists(), f"Expected file {test_file} not found"
    run_result = subprocess.run(
        [sys.executable, test_dir / test_file], capture_output=True, text=True
    )
    assert run_result.returncode == 0, (
        f"Created file failed to run: {run_result.stderr}"
    )
