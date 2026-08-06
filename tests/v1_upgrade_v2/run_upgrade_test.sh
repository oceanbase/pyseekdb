#!/usr/bin/env bash
#
# V1-to-V2 upgrade test runner.
# Uses an isolated venv so phase1 runs under installed pyseekdb==1.0.0b7 only
# (no local source), then upgrades to 1.0.0b8 for phase2.
#
# Flow: create venv -> install 1.0.0b7 -> phase1 -> install 1.0.0b8 -> phase2.
# Phase1 and phase2 use the same database via SEEKDB_PATH and SEEKDB_DATABASE.
#
# Prerequisite: python3.11 available. Run from repo root or from this script's directory.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use tests/seekdb.db (same as integration_tests default)
TESTS_DIR="${SCRIPT_DIR}/.."
SEEKDB_PATH="${SEEKDB_PATH:-${TESTS_DIR}/seekdb.db}"
SEEKDB_DATABASE="${SEEKDB_DATABASE:-test}"
# Virtual env moved to home to avoid large venv under repo
VENV_DIR="${VENV_UPGRADE_DIR:-/home/chenminsi.cms/.venv_upgrade}"

export SEEKDB_PATH
export SEEKDB_DATABASE

mkdir -p "$(dirname "${SEEKDB_PATH}")"

echo "[run_upgrade_test] SEEKDB_PATH=${SEEKDB_PATH} SEEKDB_DATABASE=${SEEKDB_DATABASE}"

echo "[run_upgrade_test] Creating isolated venv and installing pyseekdb==1.0.0b7 (V1)..."
rm -rf "${VENV_DIR}"
python3.11 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --quiet "pyseekdb==1.0.0b7"

echo "[run_upgrade_test] Running phase1 (create v1 collection and data)..."
# Clear PYTHONPATH so the venv's pyseekdb is used, not local source (e.g. from seekdb-env or project root).
run_phase1() { PYTHONPATH= "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/phase1_create_v1_data.py"; }
run_phase2() { PYTHONPATH= "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/phase2_verify_after_upgrade.py"; }

set +e
run_phase1
r1=$?
set -e
if [ "$r1" -ne 0 ]; then
  echo "[run_upgrade_test] Phase1 FAILED (exit $r1)"
  echo "[run_upgrade_test] RESULT: FAILED"
  exit 1
fi
echo "[run_upgrade_test] Phase1 OK"

echo "[run_upgrade_test] Upgrading to pyseekdb==1.0.0b8 (V2)..."
"${VENV_DIR}/bin/pip" install --quiet "pyseekdb==1.0.0b8"

echo "[run_upgrade_test] Running phase2 (verify after upgrade)..."
set +e
run_phase2
r2=$?
set -e
if [ "$r2" -ne 0 ]; then
  echo "[run_upgrade_test] Phase2 FAILED (exit $r2)"
  echo "[run_upgrade_test] RESULT: FAILED"
  exit 1
fi
echo "[run_upgrade_test] Phase2 OK"
echo "[run_upgrade_test] RESULT: PASSED"
