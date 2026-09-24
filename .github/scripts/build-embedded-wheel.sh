#!/usr/bin/env bash
# CI-local Linux wheel: not a portable manylinux release or a PyPI publication.
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${RUNNER_TEMP:?use an isolated GitHub runner temporary directory}"
: "${EMBEDDED_WHEEL_DIR:?wheel output/cache directory is required}"
[[ "$(uname -s)/$(uname -m)" == Linux/x86_64 ]]
PYTHON="$(command -v python3)"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
read -r SEEKDB_REPOSITORY SEEKDB_SHA BINDINGS_SHA FIX_COMMIT < <(
  "$PYTHON" -c 'import json, sys; p=json.load(open(sys.argv[1])); print(p["seekdb_repository"], p["seekdb_sha"], p["bindings_sha"], p["fix_commit"])' \
    "$REPO_ROOT/.github/embedded-source.json"
)
[[ "$SEEKDB_REPOSITORY" == "https://github.com/cms-cms/seekdb.git" ]]
[[ "$SEEKDB_SHA" =~ ^[0-9a-f]{40}$ && "$BINDINGS_SHA" =~ ^[0-9a-f]{40}$ && "$FIX_COMMIT" =~ ^[0-9a-f]{40}$ ]]
BUILD_ROOT="$(mktemp -d "$RUNNER_TEMP/embedded-source.XXXXXX")"
mkdir -p "$EMBEDDED_WHEEL_DIR"
[[ -z "$(ls -A "$EMBEDDED_WHEEL_DIR")" ]] || { echo 'refusing to overwrite wheel evidence' >&2; exit 1; }

checkout_exact() {
  local repository="$1" revision="$2" target="$3"
  git init -q "$target"
  git -C "$target" remote add origin "$repository"
  git -C "$target" config remote.origin.promisor true
  git -C "$target" config remote.origin.partialclonefilter blob:none
  # Full ancestry is needed to prove the kernel contains the required fix.
  # Historical source blobs are not: fetch them lazily for the exact checkout.
  git -C "$target" fetch --filter=blob:none --no-tags origin "$revision"
  git -C "$target" checkout --detach FETCH_HEAD
  [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]
}
checkout_exact "$SEEKDB_REPOSITORY" "$SEEKDB_SHA" "$BUILD_ROOT/seekdb"
git -C "$BUILD_ROOT/seekdb" merge-base --is-ancestor "$FIX_COMMIT" HEAD
checkout_exact https://github.com/oceanbase/seekdb-bindings.git "$BINDINGS_SHA" "$BUILD_ROOT/bindings"
git -C "$BUILD_ROOT/bindings" submodule update --init --depth=1 deps/mariadb-connector-c

# Match the upstream seekdb-bindings Ubuntu source-build recipe. These are
# packaging/client tools, not dependencies of the seekdb compilation target.
DEP_PROFILE="$BUILD_ROOT/seekdb/deps/init/oceanbase.el9.x86_64.deps"
sed -i -E '/^[^#]*(target=(obshell|obdeploy)|(lib)?obclient-)/d' "$DEP_PROFILE"

# Install the source's exact Rust toolchain once, before parallel compilation;
# never let two Rust targets race while auto-installing the same components.
RUST_TOOLCHAIN="$("$PYTHON" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["toolchain"]["channel"])' \
  "$BUILD_ROOT/seekdb/rust/rust-toolchain.toml")"
[[ "$RUST_TOOLCHAIN" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
export RUSTUP_HOME="$BUILD_ROOT/rustup" CARGO_HOME="$BUILD_ROOT/cargo"
rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal --no-self-update
export RUSTUP_TOOLCHAIN="$RUST_TOOLCHAIN" RUSTUP_AUTO_INSTALL=0
rustup run "$RUST_TOOLCHAIN" cargo --version
BUILD_JOBS="$(nproc)"
(( BUILD_JOBS <= 3 )) || BUILD_JOBS=3
MEMORY_JOBS="$(awk '/MemTotal:/ {n=int(($2/1024-3072)/4096); print (n>0?n:1)}' /proc/meminfo)"
(( BUILD_JOBS <= MEMORY_JOBS )) || BUILD_JOBS="$MEMORY_JOBS"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" CARGO_BUILD_JOBS="$BUILD_JOBS"
(
  cd "$BUILD_ROOT/seekdb"
  ./build.sh release --init -DBUILD_EMBED_MODE=ON --make -j"$BUILD_JOBS"
)
mapfile -t binaries < <(find "$BUILD_ROOT/seekdb/build_release" -type f -name seekdb -executable)
[[ "${#binaries[@]}" == 1 ]]
SEEKDB_BIN="${binaries[0]}"
strip --strip-debug --strip-unneeded "$SEEKDB_BIN"
"$SEEKDB_BIN" -V 2>&1 | tee "$EMBEDDED_WHEEL_DIR/seekdb-version.txt"
grep -F "$SEEKDB_SHA" "$EMBEDDED_WHEEL_DIR/seekdb-version.txt"

# Do not call the upstream wheel target (it selects an unqualified `python`).
# Build only the driver, then build a CPython 3.11 wheel with this exact venv.
"$PYTHON" -m venv "$BUILD_ROOT/venv"
BUILD_PYTHON="$BUILD_ROOT/venv/bin/python"
"$BUILD_PYTHON" -m pip install --disable-pip-version-check \
  'pip==25.3' 'build==1.3.0' 'cmake==3.31.6' 'nanobind==3.0.0' 'scikit-build-core==0.11.6' 'ninja==1.11.1.4'
export PATH="$BUILD_ROOT/venv/bin:$PATH"
cmake -S "$BUILD_ROOT/bindings" -B "$BUILD_ROOT/bindings/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DSEEKDB_BUILD_PYTHON=OFF \
  -DSEEKDB_BIN="$SEEKDB_BIN"
cmake --build "$BUILD_ROOT/bindings/build" --target seekdb --parallel "$BUILD_JOBS"
"$BUILD_PYTHON" -m pip wheel --no-deps --no-build-isolation \
  --config-settings="cmake.define.Python_EXECUTABLE=$BUILD_PYTHON" \
  --config-settings="cmake.define.Python_FIND_VIRTUALENV=ONLY" \
  --wheel-dir "$EMBEDDED_WHEEL_DIR" "$BUILD_ROOT/bindings/python"
"$PYTHON" "$REPO_ROOT/.github/scripts/embedded_wheel_manifest.py" create \
  "$REPO_ROOT/.github/embedded-source.json" "$EMBEDDED_WHEEL_DIR" "$SEEKDB_BIN"
