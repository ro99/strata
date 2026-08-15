#!/usr/bin/env bash
# Exactness gate for page-major physical prefill. The candidate arm visits
# layers outermost over a page of prompt rows, each row owning one device mHC
# slot; the reference arm is the accepted token-major page-1 path. Every
# observable must match bit for bit.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-page-major-prefill-correctness"}
runner=${RUNNER:-"${repo_root}/build-pagemajor/strata-deepseek-run"}
devices=${DEVICES:-"1,2"}
page_tokens=${PAGE_TOKENS:-4}
prompt_repetitions=${PROMPT_REPETITIONS:-4}

candidate="page${page_tokens}"
mkdir -p "${result_dir}/page1" "${result_dir}/${candidate}"
prompt=""
for ((repetition=0; repetition<prompt_repetitions; ++repetition)); do
    prompt+=" x"
done

run_arm() {
    local name=$1
    local arm_page_tokens=$2
    "${runner}" \
        --model "${model_dir}" \
        --devices "${devices}" \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --max-context 4096 \
        --device-resident-runtime \
        --decode-topology rank-local-tp2 \
        --prefill-page-tokens "${arm_page_tokens}" \
        --max-new 1 \
        --prompt "${prompt}" \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --logit-trace \
        --layer-hash-trace \
        --detailed-timing \
        --quiet \
        --json \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

run_arm page1 1
run_arm "${candidate}" "${page_tokens}"

page1_route_sha=$(sha256sum "${result_dir}/page1/routes.jsonl" | cut -d' ' -f1)
candidate_route_sha=$(sha256sum "${result_dir}/${candidate}/routes.jsonl" | cut -d' ' -f1)
jq -n \
    --slurpfile page1 "${result_dir}/page1/generation.json" \
    --slurpfile candidate "${result_dir}/${candidate}/generation.json" \
    --arg page1_route_sha "${page1_route_sha}" \
    --arg candidate_route_sha "${candidate_route_sha}" \
    --arg page_tokens "${page_tokens}" '
    {
        page_tokens: ($page_tokens | tonumber),
        prompt_tokens: [$page1[0].prompt_tokens, $candidate[0].prompt_tokens],
        generated_tokens_equal:
            ($page1[0].generated_token_ids == $candidate[0].generated_token_ids),
        logits_equal:
            ($page1[0].diagnostics.logits == $candidate[0].diagnostics.logits),
        layer_hashes_equal:
            ($page1[0].diagnostics.layer_hidden_hashes ==
             $candidate[0].diagnostics.layer_hidden_hashes),
        operation_hashes_equal:
            ($page1[0].diagnostics.operation_hashes ==
             $candidate[0].diagnostics.operation_hashes),
        routes_equal: ($page1_route_sha == $candidate_route_sha),
        zero_decode_checkpoint_reads:
            ($page1[0].decode_checkpoint_read_bytes == 0 and
             $candidate[0].decode_checkpoint_read_bytes == 0),
        candidate_maximum_page:
            $candidate[0].phases.prefill.graph.prefill_max_page_tokens,
        page1_prefill_seconds: $page1[0].prefill_seconds,
        candidate_prefill_seconds: $candidate[0].prefill_seconds,
        page1_routed_cpu_seconds:
            $page1[0].phases.prefill.device_moe_runtime.routed_cpu_seconds,
        candidate_routed_cpu_seconds:
            $candidate[0].phases.prefill.device_moe_runtime.routed_cpu_seconds,
        page1_host_callbacks:
            $page1[0].phases.prefill.device_moe_runtime.host_callback_batches,
        candidate_host_callbacks:
            $candidate[0].phases.prefill.device_moe_runtime.host_callback_batches
    }
    | .correctness_pass =
        (.generated_tokens_equal and .logits_equal and
         .layer_hashes_equal and .operation_hashes_equal and
         .routes_equal and .zero_decode_checkpoint_reads and
         .candidate_maximum_page > 1)
    ' >"${result_dir}/summary.json"

cat "${result_dir}/summary.json"
jq -e '.correctness_pass' "${result_dir}/summary.json" >/dev/null
