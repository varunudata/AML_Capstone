#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
exec "${PYTHON:-python3}" -m dl.src.report --artifact-root "${AML_CAPSTONE_ARTIFACTS:-$repo}" "$@"
