#!/usr/bin/env bash
# MIX-1 route census on a real DeepSeek V4 run (experiment 0162).
#
# strata-deepseek-run pins CUDA_DEVICE_ORDER=PCI_BUS_ID, so devices 1,2 are the
# RTX 3090 rank pair, matching nvidia-smi. Both cards are capped and clocked at
# the production operating point for this run.
#
# The census counts route CHOICES, not time, so the clock cap is irrelevant to
# the result. The prompt and token count are deliberately tiny: the census needs
# each route exercised at least once, not a throughput measurement, and the 156
# GB load dominates the arm regardless.
#
# Usage: dsv4_route_census.sh LOG CENSUS_JSON [EXTRA_FLAGS...]
set -euo pipefail
LOG="${1:?logfile}"; CENSUS="${2:?census json}"; shift 2
# Flags follow the recorded working invocation of experiment 0081. Note the
# remap: CUDA_VISIBLE_DEVICES selects the two 3090s, after which they are
# --devices 0,1 to this process. Without --host-routed-moe the device weight
# arena exhausts at layers.2.ffn.shared_experts.w1 -- an explicit refusal, not
# a fallback.
exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
  ./build-release/strata-deepseek-run --model models/dsv4f \
  --devices 0,1 --host-memory 216G --vram-fraction .95 \
  --max-context 256 --max-new 4 \
  --prompt "Explain tensor parallelism briefly." \
  --device-resident-runtime ${MOE_MODE:---host-routed-moe} \
  --host-attention-threads 28 \
  --route-census "$CENSUS" "$@" > "$LOG" 2>&1
