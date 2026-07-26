#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
usage: scripts/run_deepseek_v4_copy_kernel_overlap_probe.sh

Answers one question, without loading a model: with a kernel actually running,
does a cold-slice H2D proceed alongside it?

Experiment 0026 rejected a dedicated copy stream after measuring that a deeper
copy queue recovers nothing. That is copy-queue depth, not copy/compute
overlap. The runtime issues every weight upload on the same stream its kernels
use and synchronizes it (kernels/cuda/backend.cu), so no compute is ever in
flight during an upload and a deeper queue has nothing to recover. This probe
supplies the arm that was never run.

Arms per device, medians over REPETITIONS:
  copy_only     cold slices, synchronize per copy
  kernel_only   calibrated to cost about what a copy costs
  shared_stream both on one stream -- what the runtime does today
  split_stream  copy on its own stream, kernel on the compute stream

Decision metric: overlap_efficiency = (shared - split) / min(copy, kernel).
  ~0.0  a copy stream is worth nothing here; the mechanism is dead.
  ~1.0  the smaller term disappears completely.

Environment overrides:
  MODEL_DIR, CHECKPOINT_FILE, RESULT_DIR, CUDA_DEVICES, NUMA_NODES,
  REPETITIONS, COLD_COPIES, COLD_SLICE_BYTES, OVERLAP_COPIES
EOF
    exit 0
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
checkpoint_file=${CHECKPOINT_FILE:-}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-copy-kernel-overlap"}
devices=${CUDA_DEVICES:-0,1,2}
numa_nodes=${NUMA_NODES:-0,1}
repetitions=${REPETITIONS:-3}
cold_copies=${COLD_COPIES:-512}
cold_slice_bytes=${COLD_SLICE_BYTES:-4456448}
overlap_copies=${OVERLAP_COPIES:-256}

# Stages other than the overlap arm are not under test here, so their extents
# are the smallest the probe accepts. The cold-slice arms stay at their 0026
# settings and serve as a control that the machine is in the same state.
transfer_bytes=${TRANSFER_BYTES:-16M}
activation_bytes=${ACTIVATION_BYTES:-4M}
io_bytes=${IO_BYTES:-64M}
io_block_bytes=${IO_BLOCK_BYTES:-4M}
queue_depths=${QUEUE_DEPTHS:-1}
prefill_rows=${PREFILL_ROWS:-30}

if [[ -z "${checkpoint_file}" ]]; then
    checkpoint_file=$(find "${model_dir}" -maxdepth 1 -type f \
        -name 'model-*.safetensors' -printf '%s %p\n' | sort -nr | awk 'NR == 1 {print $2}')
fi
if [[ ! -f "${checkpoint_file}" ]]; then
    echo "error: checkpoint range source does not exist: ${checkpoint_file}" >&2
    exit 1
fi
if [[ "${repetitions}" -lt 3 ]]; then
    echo "error: the charter requires at least three repetitions" >&2
    exit 1
fi
for command in jq nvidia-smi numactl; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: ${command} is required" >&2
        exit 1
    fi
done

mkdir -p "${result_dir}"
cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_SANITIZERS=OFF \
    -DSTRATA_ENABLE_CUDA=ON
cmake --build "${repo_root}/build" --target strata-topology-probe --parallel

{
    date --iso-8601=seconds
    uname -a
    numactl --hardware
    nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current \
        --format=csv
} >"${result_dir}/system.txt"

"${repo_root}/build/strata-topology-probe" \
    --checkpoint-file "${checkpoint_file}" \
    --devices "${devices}" \
    --numa-nodes "${numa_nodes}" \
    --repetitions "${repetitions}" \
    --transfer-bytes "${transfer_bytes}" \
    --activation-bytes "${activation_bytes}" \
    --io-bytes "${io_bytes}" \
    --io-block-bytes "${io_block_bytes}" \
    --queue-depths "${queue_depths}" \
    --prefill-rows "${prefill_rows}" \
    --cold-slice-bytes "${cold_slice_bytes}" \
    --cold-copies "${cold_copies}" \
    --overlap-copies "${overlap_copies}" \
    >"${result_dir}/probe.json" \
    2>"${result_dir}/probe.log"

jq -e '
    .schema == "strata.topology_probe" and
    (.copy_kernel_overlap | length) > 0 and
    all(.copy_kernel_overlap[]; .verified)
' "${result_dir}/probe.json" >/dev/null

jq '{
    configuration,
    cold_slice_transfers,
    copy_kernel_overlap
}' "${result_dir}/probe.json" >"${result_dir}/overlap.json"

cat "${result_dir}/overlap.json"
