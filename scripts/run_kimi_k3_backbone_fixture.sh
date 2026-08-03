#!/usr/bin/env bash
# Emit the gate-5 and gate-6 backbone fixture from the checkpoint's own reference.
#
# Arm budget, stated before launch rather than discovered afterwards:
#
#   prompt sweep    ~205 GiB read from SATA — 93 layers of ~1.1 GiB dense
#                   spine plus up to 64 routed experts of 16.7 MiB each. At
#                   the measured 178-400 MB/s, plus GPU
#                   dequantization.                            ~12-20 min
#   decode sweep    one token routes to exactly 16 experts per layer, 3.5x
#                   less than the page, and the dense spine is warm in page
#                   cache from the prompt sweep.                 ~5-8 min
#   runtime arm     106.55 GiB dense spine load, one 4-token prefill page,
#                   one decode step (gates 5 and 6).            ~10-12 min
#                                                      total  ~27-40 min
#
# What was shortened: four prompt tokens, one prefill page, one decode step.
# Four tokens still cross causal attention positions, the chunkwise KDA path,
# and tokens that do and do not share experts; one decode step is enough to gate
# the recurrent-state handoff, which is the only thing decode adds. What must
# stay: all 93 layers and the full 7168-wide hidden state, because gate 5 is
# precisely the claim that error stays bounded across depth — a truncated
# backbone would not test it. The expert read dominates and scales with tokens,
# which is why the token count is the knob that was turned.
#
# The oracle is Python because the reference is; the runtime stays C/C++. The
# venv lives on tmpfs and every scratch path is redirected away from the NVMe.
set -euo pipefail

MODEL=${MODEL:-/data/kimi-k3}
# Reference activations are derived from model weights, so they may not land on
# the NVMe. `results/` in the working tree is on `/dev/nvme0n1p2`; this default
# is on the same SATA disk as the checkpoint. The script refuses any other.
OUT=${OUT:-/data/strata-results/kimi-k3-fixtures}
VENV=${VENV:-/dev/shm/kimi-oracle-venv}
DEVICE=${DEVICE:-cuda:1}
TOKENS=${TOKENS:-4}

if [[ ! -d "${VENV}" ]]; then
    echo "creating ${VENV}"
    python3 -m venv --system-site-packages "${VENV}"
    "${VENV}/bin/pip" install -q --no-deps fla-core==0.5.2 einops
fi

export TMPDIR=/dev/shm
export TRITON_CACHE_DIR=/dev/shm/triton-cache
export CUDA_CACHE_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/inductor-cache
export HF_HOME=/dev/shm/hf-home
export PYTHONPATH=${PYTHONPATH:-}:scripts
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${HF_HOME}"

# Field 7 of `/sys/block/<disk>/stat` is cumulative sectors written. It is field
# 10 of a `/proc/diskstats` row, which has three extra leading columns; reading
# $10 here samples `io_ticks` and reports milliseconds as if they were sectors.
# That misread hid a 20 MiB fixture write behind a plausible-looking 223 KiB.
write_sectors() { awk '{print $7}' "/sys/block/$1/stat"; }

started=$(date +%s)
before=$(write_sectors nvme0n1)
"${VENV}/bin/python" scripts/kimi_k3_reference_backbone.py \
    --model "${MODEL}" --out "${OUT}" --tokens "${TOKENS}" --device "${DEVICE}" "$@"
sync
after=$(write_sectors nvme0n1)
elapsed=$((  $(date +%s) - started ))

# An idle control taken right after the run, not an absolute constant. This
# machine's idle NVMe write rate measured 13-29 KiB/s on 2026-08-02 — journald
# and friends, nothing to do with the model. A gate fixed at the 0.1 KiB/s the
# handover recorded would fail every run here and then be ignored, which is
# worse than no gate. What must hold is that the run writes no more than the
# machine writes doing nothing.
#
# This is a whole-disk counter and therefore a coarse backstop: the first full
# run tripped it at 94 KiB/s against a 29 KiB/s control, and the excess was a
# concurrent `cmake --build` dropping object files into the working tree, which
# lives on that disk. Run this on a quiet machine, and read the oracle's own
# `process write_bytes` line as the attributable number.
control_before=$(write_sectors nvme0n1)
sleep 30
sync
control_after=$(write_sectors nvme0n1)

run_kib=$(((after - before) / 2))
control_kib=$(((control_after - control_before) / 2))
printf 'nvme0n1 during the run:  %s KiB over %s s = %s KiB/s\n' \
    "${run_kib}" "${elapsed}" "$((run_kib / (elapsed > 0 ? elapsed : 1)))"
printf 'nvme0n1 idle control:    %s KiB over 30 s = %s KiB/s\n' \
    "${control_kib}" "$((control_kib / 30))"
if (( run_kib * 30 > control_kib * elapsed * 2 )); then
    echo "NVMe write gate FAILED: the run wrote more than twice the idle rate"
    exit 1
fi
echo "NVMe write gate PASS: run rate is within twice the idle control"
