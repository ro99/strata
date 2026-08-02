#!/usr/bin/env bash
# Regenerate the Kimi-K3 layer fixtures from the checkpoint's own reference.
#
# The oracle is Python because the reference is; the runtime stays C/C++. The
# venv lives on tmpfs and every scratch path is redirected away from the NVMe,
# so running this cannot put a byte on the disk the operator is protecting.
set -euo pipefail

MODEL=${MODEL:-/data/kimi-k3}
# The fixture holds reference activations, which are derived from model weights,
# so it may not land on the NVMe. `results/` in the working tree is on
# `/dev/nvme0n1p2`; this default is on the same SATA disk as the checkpoint.
OUT=${OUT:-/data/strata-results/kimi-k3-fixtures}
VENV=${VENV:-/dev/shm/kimi-oracle-venv}
DEVICE=${DEVICE:-cuda:1}
TOKENS=${TOKENS:-64}

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
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${HF_HOME}"

# Field 7 of `/sys/block/<disk>/stat` is cumulative sectors written. It is field
# 10 of a `/proc/diskstats` row, which has three extra leading columns; reading
# $10 here samples `io_ticks` and reports milliseconds as if they were sectors.
# That misread hid a 20 MiB fixture write behind a plausible-looking 223 KiB.
write_sectors() { awk '{print $7}' "/sys/block/$1/stat"; }

before=$(write_sectors nvme0n1)
"${VENV}/bin/python" scripts/kimi_k3_reference_fixture.py \
    --model "${MODEL}" --out "${OUT}" --tokens "${TOKENS}" --device "${DEVICE}"
sync
after=$(write_sectors nvme0n1)

printf 'nvme0n1 written during the run: %s sectors (%s KiB)\n' \
    "$((after - before))" "$(((after - before) / 2))"
