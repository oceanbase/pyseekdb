"""Tests for the CI fallback that validates pytest's final summary line."""

import subprocess
from pathlib import Path

import pytest

CHECK_SCRIPT = Path(__file__).parents[2] / ".github" / "scripts" / "check-pytest-summary.sh"


@pytest.mark.parametrize(
    ("summary", "expected_returncode"),
    [
        ("================ 92 passed, 1 xfailed in 1.23s ================", 0),
        ("================ 92 passed in 1.23s ================\n\n \t", 0),
        ("= 12 failed, 92 passed, 311 skipped, 1 xpassed in 135.47s =", 1),
        ("================ 1 error, 2 passed in 0.42s ================", 1),
        ("================ 5 skipped in 0.10s ================", 1),
        ("test session interrupted", 1),
    ],
)
def test_check_pytest_summary(tmp_path: Path, summary: str, expected_returncode: int) -> None:
    pytest_log = tmp_path / "pytest.log"
    pytest_log.write_text(f"test output\n{summary}\n", encoding="utf-8")

    result = subprocess.run(  # noqa: S603 - execute the repository's fixed CI script
        ["/bin/bash", str(CHECK_SCRIPT), str(pytest_log)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_returncode, result.stderr


def test_check_pytest_summary_rejects_missing_log(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603 - execute the repository's fixed CI script
        ["/bin/bash", str(CHECK_SCRIPT), str(tmp_path / "missing.log")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
