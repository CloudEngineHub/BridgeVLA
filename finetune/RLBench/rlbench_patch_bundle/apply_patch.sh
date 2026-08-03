#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="587a6a0e6dc8cd36612a208724eb275fe8cb4470"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-${SCRIPT_DIR}/../../bridgevla/libs/RLBench}"
PATCH_FILE="${SCRIPT_DIR}/rlbench-587a6a0-local.patch"
CHECKSUM_FILE="${SCRIPT_DIR}/SHA256SUMS"
PATCHED_FILES=(
  "rlbench/environment.py"
  "rlbench/utils.py"
  "setup.py"
)

if [ ! -d "${TARGET_DIR}" ]; then
  echo "[ERROR] RLBench target does not exist: ${TARGET_DIR}" >&2
  exit 1
fi
TARGET_DIR="$(cd "${TARGET_DIR}" && pwd)"

if ! git -C "${TARGET_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] Target is not a Git working tree: ${TARGET_DIR}" >&2
  exit 1
fi

if (cd "${TARGET_DIR}" && sha256sum -c "${CHECKSUM_FILE}" >/dev/null 2>&1); then
  echo "[OK] RLBench patch is already present and checksums match."
  exit 0
fi

ACTUAL_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
if [ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]; then
  echo "[ERROR] Unexpected RLBench commit." >&2
  echo "        expected: ${EXPECTED_COMMIT}" >&2
  echo "        actual:   ${ACTUAL_COMMIT}" >&2
  exit 1
fi

if ! git -C "${TARGET_DIR}" diff --quiet HEAD -- "${PATCHED_FILES[@]}"; then
  echo "[ERROR] One or more target files already contain local changes." >&2
  echo "        Refusing to overwrite them; inspect git diff first." >&2
  git -C "${TARGET_DIR}" status --short -- "${PATCHED_FILES[@]}" >&2
  exit 1
fi

git -C "${TARGET_DIR}" apply --check "${PATCH_FILE}"
git -C "${TARGET_DIR}" apply "${PATCH_FILE}"
(cd "${TARGET_DIR}" && sha256sum -c "${CHECKSUM_FILE}")

echo "[OK] RLBench local patch applied to ${TARGET_DIR}"
