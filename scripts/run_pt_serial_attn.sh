#!/usr/bin/env bash
# Reproduction: Portuguese prompt with --serial-host-attention
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir="${repo_root}/results/deepseek-v4-pt-br-corruption"
devices=${CUDA_DEVICES:-0,1,2}
max_context=2048
max_new=64
prompt='voce fala portugues? o que e caipirinha?'
mkdir -p "${result_dir}"

run_serial_attn() {
    local tag=$1
    "${repo_root}/build/strata-deepseek-run" \
        --model "${model_dir}" \
        --devices "${devices}" \
        --max-context "${max_context}" \
        --max-new "${max_new}" \
        --serial-host-attention \
        --logit-trace --logit-trace-top-k 20 \
        --layer-hash-trace \
        --json \
        --prompt "${prompt}" 2>"${result_dir}/${tag}.stderr" \
        >"${result_dir}/${tag}.json"
    echo "[done] ${tag} exit=$?"
}

echo "=== Run 1/3: serial-host-attention ==="
run_serial_attn pt-br-serial-attn-01
echo "=== Run 2/3: serial-host-attention ==="
run_serial_attn pt-br-serial-attn-02
echo "=== Run 3/3: serial-host-attention ==="
run_serial_attn pt-br-serial-attn-03

echo "=== All serial-host-attention runs complete ==="
