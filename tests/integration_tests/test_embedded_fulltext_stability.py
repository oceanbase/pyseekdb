"""Fresh-ingestion and memtable-release regressions for seekdb issue #1382.

Each case runs in a fresh process and database directory, including with legacy
process-global pylibseekdb. Do not rerun failed queries to obtain a passing result.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from embedded_fulltext_support import validate_report


@pytest.mark.parametrize("scenario", ["first-pass", "scan-freeze"])
def test_embedded_fulltext_stability(scenario):
    """Exercise the public SDK first pass and an accelerated kernel stress case."""
    repo = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SEEKDB_TEST_DATA_ROOT", repo / ".seekdb-test-data"))
    data_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"fts-{scenario}-", dir=data_root))
    evidence_root = Path(os.environ.get("SEEKDB_TEST_ARTIFACT_ROOT", data_root / "artifacts"))
    evidence = evidence_root / root.name
    evidence.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(Path(__file__).with_name("embedded_fulltext_support.py")),
        "--scenario",
        scenario,
        "--root",
        str(root),
        "--evidence",
        str(evidence),
    ]
    # A whole-case timeout includes ingestion and connection setup; the pressure
    # window itself is 75 seconds, starting only after every worker is ready.
    with (evidence / "worker.log").open("w") as log:
        # Fixed interpreter/helper and generated paths, never shell input.
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)  # noqa: S603
        started = time.monotonic()
        try:
            with (evidence / "worker.log").open() as progress:
                while process.poll() is None:
                    if time.monotonic() - started > 900:
                        raise TimeoutError(f"FTS worker exceeded 900s; evidence={evidence}")
                    # Stream bounded real output rather than hiding all ingestion
                    # and pressure activity inside pytest until the case ends.
                    text = progress.read(16384)
                    if text:
                        print(text, end="", flush=True)
                    time.sleep(1)
                print(progress.read(), end="", flush=True)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    report_path = evidence / "summary.json"
    assert report_path.is_file(), f"worker produced no terminal report; evidence={evidence}"
    report = json.loads(report_path.read_text())
    validate_report(report, scenario)
    assert process.returncode == 0, f"worker rc={process.returncode}; evidence={evidence}"
    # Remove only our generated DB after the process has exited and evidence has
    # been collected. Failed cases retain their DB for investigation.
    shutil.rmtree(root)
