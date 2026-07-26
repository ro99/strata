#!/usr/bin/env bash
set -euo pipefail

# Exact old/new oracle for three-query DeepSeek attention at negligible prefill.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/DeepSeek-V4-Flash-DSpark"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-batched-attention-correctness"}
baseline=${BASELINE_RUNNER:-"${repo_root}/results/deepseek-v4-batched-attention-gate/main-baseline-runner"}
candidate=${CANDIDATE_RUNNER:-"${repo_root}/build/strata-deepseek-run"}
prompt=${PROMPT:-"x x x x x x x x"}

mkdir -p "${result_dir}/baseline" "${result_dir}/candidate"
cp --reflink=auto "${baseline}" "${result_dir}/baseline-runner"
cp --reflink=auto "${candidate}" "${result_dir}/candidate-runner"
baseline="${result_dir}/baseline-runner"
candidate="${result_dir}/candidate-runner"

run_arm() {
    local arm=$1 runner=$2
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.85 --max-context 2048 \
        --prefill-page-tokens 3 --max-new 2 --prompt "${prompt}" \
        --pin-resident-arena --block-kv-cache \
        --flash-attention --flash-attention-minimum-rows 0 \
        --route-trace "${result_dir}/${arm}/routes.jsonl" \
        --logit-trace --layer-hash-trace --detailed-timing --quiet --json \
        >"${result_dir}/${arm}/generation.json" \
        2>"${result_dir}/${arm}/generation.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" status --short --branch
    git -C "${repo_root}" diff --stat
    sha256sum "${baseline}" "${candidate}"
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

run_arm baseline "${baseline}"
run_arm candidate "${candidate}"

jq -n \
    --slurpfile baseline "${result_dir}/baseline/generation.json" \
    --slurpfile candidate "${result_dir}/candidate/generation.json" '
    ($baseline[0]) as $b | ($candidate[0]) as $c |
    {
      baseline_prefill_seconds: $b.prefill_seconds,
      candidate_prefill_seconds: $c.prefill_seconds,
      baseline_flash_calls: $b.phases.prefill.cuda.flash_attention_calls,
      candidate_flash_calls: $c.phases.prefill.cuda.flash_attention_calls,
      gates: {
        prompt_is_twelve_tokens:
          ($b.prompt_tokens == 12 and $c.prompt_tokens == 12),
        batching_exercised:
          ($c.phases.prefill.cuda.flash_attention_calls <
           $b.phases.prefill.cuda.flash_attention_calls),
        generated_tokens_equal:
          ($b.generated_token_ids == $c.generated_token_ids),
        logits_equal:
          ($b.diagnostics.logits == $c.diagnostics.logits),
        layer_hashes_equal:
          ($b.diagnostics.layer_hidden_hashes ==
           $c.diagnostics.layer_hidden_hashes),
        operation_hashes_equal:
          ($b.diagnostics.operation_hashes ==
           $c.diagnostics.operation_hashes),
        zero_checkpoint_reads:
          ($b.phases.prefill.checkpoint_read_bytes == 0 and
           $c.phases.prefill.checkpoint_read_bytes == 0 and
           $b.decode_checkpoint_read_bytes == 0 and
           $c.decode_checkpoint_read_bytes == 0),
        bounded_memory:
          ($b.rss_bytes <= $b.memory_plan.required_host_bytes and
           $c.rss_bytes <= $c.memory_plan.required_host_bytes)
      }
    }
    | .acceptance_pass = ([.gates[]] | all)
    ' >"${result_dir}/summary.json"

cmp "${result_dir}/baseline/routes.jsonl" \
    "${result_dir}/candidate/routes.jsonl"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
cat "${result_dir}/summary.json"
