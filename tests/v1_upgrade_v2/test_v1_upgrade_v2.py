"""
Pytest entry point for V1-to-V2 upgrade test.

Runs the standalone shell script so that:
  python3.11 -m pytest tests/ -v -s
includes this test and executes the full upgrade flow (venv + 1.0.0b7 phase1 + 1.0.0b8 phase2).

This test lives under tests/v1_upgrade_v2/ (not under integration_tests/) so that
pytest does not load integration_tests/conftest.py, avoiding pyseekdb/httpx/idna imports
that can break in some environments.
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run_upgrade_test.sh"


@pytest.mark.skipif(
    not RUN_SCRIPT.exists(),
    reason="run_upgrade_test.sh not found",
)
def test_v1_upgrade_v2_run_script():
    """Run the V1->V2 upgrade script; assert it exits 0."""
    env = os.environ.copy()
    proc = subprocess.run(
        ["bash", str(RUN_SCRIPT)],
        cwd=SCRIPT_DIR,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"run_upgrade_test.sh exited with {proc.returncode}; "
        "run it manually for full output: ./tests/v1_upgrade_v2/run_upgrade_test.sh"
    )
