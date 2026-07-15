#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
usage: scripts/run_glm52_topology_probe.sh

Environment overrides:
  MODEL_DIR, CHECKPOINT_FILE, RESULT_DIR, CUDA_DEVICES, NUMA_NODES,
  REPETITIONS, TRANSFER_BYTES, ACTIVATION_BYTES, IO_BYTES, IO_BLOCK_BYTES,
  QUEUE_DEPTHS, PREFILL_ROWS

Run the full T1 matrix in a named tmux session, for example:
  tmux new-session -d -s strata-glm52-t1 \
    'scripts/run_glm52_topology_probe.sh >results/glm52-topology/session.log 2>&1'
EOF
    exit 0
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/glm52"}
checkpoint_file=${CHECKPOINT_FILE:-}
result_dir=${RESULT_DIR:-"${repo_root}/results/glm52-topology/full-v1"}
devices=${CUDA_DEVICES:-0,1,2}
numa_nodes=${NUMA_NODES:-0,1}
repetitions=${REPETITIONS:-3}
transfer_bytes=${TRANSFER_BYTES:-64M}
activation_bytes=${ACTIVATION_BYTES:-8M}
io_bytes=${IO_BYTES:-256M}
io_block_bytes=${IO_BLOCK_BYTES:-4M}
queue_depths=${QUEUE_DEPTHS:-1,4,8}
prefill_rows=${PREFILL_ROWS:-30}
expected_index_sha256=43298345833417b1ad2a8b76d012a83d4f2275d532e5ab38e118566f1ac7b12b

if [[ -z "${checkpoint_file}" ]]; then
    checkpoint_file=$(find "${model_dir}" -maxdepth 1 -type f \
        -name 'model-*.safetensors' -printf '%s %p\n' | sort -nr | awk 'NR == 1 {print $2}')
fi
if [[ ! -f "${checkpoint_file}" ]]; then
    echo "error: checkpoint range source does not exist: ${checkpoint_file}" >&2
    exit 1
fi
if [[ "${repetitions}" -lt 3 ]]; then
    echo "error: T1 requires at least three repetitions" >&2
    exit 1
fi
for command in jq nvidia-smi numactl findmnt lsblk; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "error: ${command} is required" >&2
        exit 1
    fi
done

mkdir -p "${result_dir}"
actual_index_sha256=$(sha256sum "${model_dir}/model.safetensors.index.json" | awk '{print $1}')
if [[ "${actual_index_sha256}" != "${expected_index_sha256}" ]]; then
    echo "error: checkpoint index SHA-256 does not match the pinned revision" >&2
    exit 1
fi

cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_SANITIZERS=OFF \
    -DSTRATA_ENABLE_CUDA=ON
cmake --build "${repo_root}/build" --target strata-topology-probe --parallel

model_source=$(findmnt -no SOURCE --target "${checkpoint_file}")
block_name=$(lsblk -no PKNAME "${model_source}" | head -n 1)
if [[ -z "${block_name}" ]]; then
    block_name=$(basename "${model_source}")
fi
block_stat="/sys/class/block/${block_name}/stat"
if [[ ! -r "${block_stat}" ]]; then
    echo "error: cannot read block counters for ${block_name}" >&2
    exit 1
fi

{
    date --iso-8601=seconds
    uname -a
    lscpu
    numactl --hardware
    free -h
    findmnt --target "${checkpoint_file}"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,compute_cap,driver_version,memory.total,memory.free,pcie.link.gen.current,pcie.link.width.current \
        --format=csv
    nvidia-smi topo -m
} >"${result_dir}/system.txt"

read -r _ _ read_sectors _ _ _ write_sectors _ <"${block_stat}"
/usr/bin/time -v \
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
    >"${result_dir}/probe.json" \
    2>"${result_dir}/probe.log" &
probe_pid=$!

: >"${result_dir}/vram-samples.csv"
while kill -0 "${probe_pid}" 2>/dev/null; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        >>"${result_dir}/vram-samples.csv" || true
    sleep 0.25
done
wait "${probe_pid}"
read -r _ _ final_read_sectors _ _ _ final_write_sectors _ <"${block_stat}"

jq -e --argjson repetitions "${repetitions}" '
    .schema == "strata.topology_probe" and .version == 1 and
    all(.transfers[]; (.samples | length) == $repetitions and
        all(.samples[]; .verified)) and
    all(.device_transfers[]; (.samples | length) == $repetitions and
        all(.samples[]; .verified)) and
    all(.kernels[]; .verified and
        (.kernel_seconds | length) == $repetitions) and
    all(.row1_expert_service[]; .verified and
        (.samples_seconds | length) == $repetitions) and
    all(.checkpoint_reads[]; (.samples | length) == $repetitions and
        all(.samples[]; .verified)) and
    all(.checkpoint_reads[] | select(.mode == "direct_physical") | .samples[];
        .physical_read_bytes == .bytes)
' "${result_dir}/probe.json" >/dev/null

maximum_resident_kib=$(awk -F: '/Maximum resident set size/ {
    gsub(/^[[:space:]]+/, "", $2); print $2
}' "${result_dir}/probe.log")
awk -F, '{
    gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2);
    if ($2 > maximum[$1]) maximum[$1] = $2
} END {
    for (device in maximum) print device "\t" maximum[device]
}' "${result_dir}/vram-samples.csv" | sort -n >"${result_dir}/peak-vram.tsv"
jq -Rn '[inputs | split("\t") |
    {device: (.[0] | tonumber), peak_used_mib: (.[1] | tonumber)}
]' <"${result_dir}/peak-vram.tsv" >"${result_dir}/peak-vram.json"

physical_read_bytes=$(((final_read_sectors - read_sectors) * 512))
physical_write_bytes=$(((final_write_sectors - write_sectors) * 512))
jq -n \
    --arg block_device "${block_name}" \
    --arg checkpoint_index_sha256 "${actual_index_sha256}" \
    --argjson physical_read_bytes "${physical_read_bytes}" \
    --argjson physical_write_bytes "${physical_write_bytes}" \
    --argjson maximum_resident_kib "${maximum_resident_kib}" \
    --slurpfile peak_vram "${result_dir}/peak-vram.json" \
    '{
        block_device: $block_device,
        checkpoint_index_sha256: $checkpoint_index_sha256,
        physical_read_bytes: $physical_read_bytes,
        physical_write_bytes: $physical_write_bytes,
        maximum_resident_kib: $maximum_resident_kib,
        peak_vram: $peak_vram[0]
    }' >"${result_dir}/resources.json"

jq -s '.[0] + {resources: .[1]}' \
    "${result_dir}/probe.json" "${result_dir}/resources.json" \
    >"${result_dir}/results.json"
jq '{
    schema, version, identity, devices, vram_peak,
    pinned_h2d: [.transfers[] | select(.memory == "pinned" and .direction == "h2d") |
        {device, numa_node, median_gb_s, samples}],
    row1_expert_service,
    kernels,
    device_transfers,
    checkpoint_reads,
    allocation,
    synchronization,
    resources
}' "${result_dir}/results.json" >"${result_dir}/cost-matrix.json"

cat "${result_dir}/cost-matrix.json"
