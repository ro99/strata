#!/usr/bin/env bash
# Production operating point (experiment 0123/0126). The server pins
# CUDA_DEVICE_ORDER=PCI_BUS_ID (apps/strata_server.cpp), so devices 1,2 are
# the RTX 3090 rank pair; standalone CUDA probes see default order instead,
# where the ranks are 0,1 and the 5060 Ti is 2. Extra flags append.
# Usage: dsv4_decode15_server.sh LOGFILE [EXTRA_FLAGS...]
set -euo pipefail
LOG="${1:?logfile}"; shift
# strata_server.cpp only pins PCI_BUS_ID when the operator has not chosen an
# order themselves, and this box's login shell exports
# CUDA_DEVICE_ORDER=FASTEST_FIRST. Inheriting that made `--devices 1,2` select
# the 5060 Ti plus one 3090 -- capping both ranks at the 16 GiB card and
# leaving a 3090 idle -- which is the exact failure that file's comment warns
# about. Pin it here so the header's claim about devices 1,2 actually holds.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 8033 is the operator's llama.cpp proxy; this campaign binds 8133.
PORT="${PORT:-8133}"
exec ./build-release/strata-server --model models/dsv4f --model-type deepseek \
  --model-id strata-deepseek-v4 --devices 1,2 --context-size 16384 \
  --vram-fraction 0.95 --decode-topology rank-local-tp2 \
  --prefill-page-tokens 8192 --host 127.0.0.1 --port "$PORT" "$@" > "$LOG" 2>&1
