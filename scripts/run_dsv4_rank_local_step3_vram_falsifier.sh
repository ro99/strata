#!/usr/bin/env bash
# One-step rank-local allocation/admission falsifier. This is intentionally
# cheaper than a timing matrix: setup is the mechanism under test, followed by
# one exact decode step to prove the admitted objects are usable.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner=${RUNNER:-"${repo_root}/build-landing-nccl/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-rank-local-main-landing/step3-vram-falsifier"}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}
devices=${DEVICES:-1,2}
maximum_context=${MAX_CONTEXT:-4096}
expected_kv_bytes=${EXPECTED_KV_BYTES:-28627968}
device_ceiling=22548578304
host_ceiling=231928233984

mkdir -p "${result_dir}"
set +e
"${runner}" \
    --model "${model_dir}" \
    --prompt "${prompt}" \
    --max-new 2 \
    --devices "${devices}" \
    --max-context "${maximum_context}" \
    --host-memory 216G \
    --vram-fraction 0.95 \
    --device-resident-runtime \
    --decode-topology rank-local-tp2 \
    --quiet --json \
    >"${result_dir}/generation.json" \
    2>"${result_dir}/generation.log"
status=$?
set -e
printf '%d\n' "${status}" >"${result_dir}/generation.exit"
if [[ ${status} -ne 0 ]]; then
    exit "${status}"
fi

jq -e \
    --argjson expected_kv "${expected_kv_bytes}" \
    --argjson device_ceiling "${device_ceiling}" \
    --argjson host_ceiling "${host_ceiling}" '
    .generated_token_ids == [671, 22510] and
    .decode_steps == 1 and
    .decode_checkpoint_read_bytes == 0 and
    .generation_checkpoint_read_bytes > 0 and
    .rss_bytes <= $host_ceiling and
    .rank_local_memory.admitted_host_bytes ==
        .memory_plan.required_host_bytes and
    .rank_local_memory.admitted_host_bytes <= $host_ceiling and
    (.device_vram_used_bytes | length) == 2 and
    ([.device_vram_used_bytes[] <= $device_ceiling] | all) and
    (.memory_plan.per_device_kv_cache_bytes ==
        [$expected_kv, $expected_kv]) and
    (.rank_local_memory.initial_device_vram_bytes | length) == 2 and
    (.rank_local_memory.weight_bytes | length) == 2 and
    ([.rank_local_memory.weight_bytes[] > 0] | all) and
    (.rank_local_memory.expert_cache_capacity_bytes | length) == 2 and
    (.rank_local_memory.admitted_device_bytes | length) == 2 and
    ([.rank_local_memory.admitted_device_bytes[] <= $device_ceiling] | all) and
    ([range(0; 2) as $rank |
        .weight_cache.capacity_bytes[$rank] ==
            (.weight_cache.pinned_bytes[$rank] +
             .rank_local_memory.expert_cache_capacity_bytes[$rank])] | all)
' "${result_dir}/generation.json" >/dev/null

jq \
    --argjson expected_kv "${expected_kv_bytes}" \
    --argjson device_ceiling "${device_ceiling}" '
    {
        maximum_context: .memory_plan.maximum_context_tokens,
        expected_replicated_kv_bytes_per_rank: $expected_kv,
        generated_token_ids,
        decode_steps,
        decode_checkpoint_read_bytes,
        rss_bytes,
        device_ceiling_bytes: $device_ceiling,
        device_vram_used_bytes,
        per_device_kv_cache_bytes:
            .memory_plan.per_device_kv_cache_bytes,
        rank_local_memory,
        live_weight_cache_capacity_bytes: .weight_cache.capacity_bytes,
        live_weight_cache_pinned_bytes: .weight_cache.pinned_bytes,
        kv_device_peak_bytes: .kv_cache.device_peak_bytes
    }
' "${result_dir}/generation.json" >"${result_dir}/summary.json"

sha256sum \
    "${result_dir}/generation.json" \
    "${result_dir}/summary.json" \
    >"${result_dir}/sha256.txt"
