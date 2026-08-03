#!/usr/bin/env bash
# Faults the pinned checkpoint back into the page cache so an arm is not
# measuring NVMe re-reads. Decode staging time doubled between two sessions at
# identical miss counts and identical byte volume, which is page-cache
# residency, not work. Arms must start from the same state.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-models/laguna-s-21}"
started=$(date +%s)
cat "$MODEL"/*.safetensors > /dev/null
echo "[warm] $MODEL faulted in $(( $(date +%s) - started )) s"
