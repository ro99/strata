#!/usr/bin/env bash
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_type=${MODEL_TYPE:-deepseek}
if [[ "${model_type}" == deepseek ]]; then
    default_model_dir="${repo_root}/models/DeepSeek-V4-Flash-DSpark"
else
    default_model_dir="${repo_root}/models/glm52"
fi
model_dir=${MODEL_DIR:-"${default_model_dir}"}
runner=${RUNNER:-"${repo_root}/build/strata-chat"}
result_dir=${RESULT_DIR:-"${repo_root}/results/incremental-kv-chat-${model_type}"}
devices=${CUDA_DEVICES:-0,1,2}
maximum_context=${MAX_CONTEXT_TOKENS:-2048}
maximum_new=${MAX_NEW_TOKENS:-8}
repetitions=${REPETITIONS:-3}
cache_mode=${CACHE_MODE:-scalar}
first_prompt=${FIRST_PROMPT:-Remember the exact code 7319.}
second_prompt=${SECOND_PROMPT:-What exact code did I ask you to remember?}

if [[ "${model_type}" != deepseek && "${model_type}" != glm ]]; then
    printf 'MODEL_TYPE must be deepseek or glm\n' >&2
    exit 2
fi
if [[ "${cache_mode}" != scalar && "${cache_mode}" != block ]]; then
    printf 'CACHE_MODE must be scalar or block\n' >&2
    exit 2
fi
if [[ "${model_type}" == glm && "${cache_mode}" == block ]]; then
    printf 'block cache mode is DeepSeek-only\n' >&2
    exit 2
fi

mkdir -p "${result_dir}"
cp --reflink=auto "${runner}" "${result_dir}/strata-chat"
runner="${result_dir}/strata-chat"
request_one=$(jq -cn --arg text "${first_prompt}" '{command:"prompt",text:$text}')
request_two=$(jq -cn --arg text "${second_prompt}" '{command:"prompt",text:$text}')

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'model_type=%s\nmodel_dir=%s\ndevices=%s\n' \
        "${model_type}" "${model_dir}" "${devices}"
    printf 'maximum_context=%s\nmaximum_new=%s\nrepetitions=%s\ncache_mode=%s\n' \
        "${maximum_context}" "${maximum_new}" "${repetitions}" "${cache_mode}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"
git -C "${repo_root}" diff --binary >"${result_dir}/candidate.diff"

run_variant() {
    local repetition=$1
    local variant=$2
    local run_dir="${result_dir}/repetition-${repetition}/${variant}"
    local flags=()
    mkdir -p "${run_dir}"
    if [[ "${variant}" == full ]]; then flags+=(--full-reprefill); fi
    if [[ "${cache_mode}" == block ]]; then flags+=(--block-kv-cache); fi

    printf '%s\n%s\n' "${request_one}" "${request_two}" |
        /usr/bin/time -v "${runner}" \
            --model "${model_dir}" \
            --model-type "${model_type}" \
            --devices "${devices}" \
            --context-size "${maximum_context}" \
            --max-new "${maximum_new}" \
            --temperature 0 \
            --protocol jsonl \
            "${flags[@]}" \
            >"${run_dir}/events.jsonl" 2>"${run_dir}/run.log"

    jq -s '
        {
            turns: [.[] | select(.event == "turn_done") | {
                prompt_tokens,
                prefill_tokens,
                reused_prompt_tokens,
                incremental_kv_continuation,
                prefill_seconds,
                decode_seconds,
                generated_token_ids
            }]
        }
    ' "${run_dir}/events.jsonl" >"${run_dir}/summary.json"
}

for ((repetition=1; repetition<=repetitions; ++repetition)); do
    if ((repetition % 2 == 1)); then
        variants=(full incremental)
    else
        variants=(incremental full)
    fi
    for variant in "${variants[@]}"; do
        run_variant "${repetition}" "${variant}"
    done
    repetition_dir="${result_dir}/repetition-${repetition}"
    jq -n \
        --slurpfile full "${repetition_dir}/full/summary.json" \
        --slurpfile incremental "${repetition_dir}/incremental/summary.json" '
        {
            full: $full[0],
            incremental: $incremental[0],
            gates: {
                generated_tokens_equal:
                    ([ $full[0].turns[].generated_token_ids ] ==
                     [ $incremental[0].turns[].generated_token_ids ]),
                two_turns:
                    (($full[0].turns | length == 2) and
                     ($incremental[0].turns | length == 2)),
                full_reprefill_used:
                    (all($full[0].turns[];
                        (.incremental_kv_continuation | not) and
                        .reused_prompt_tokens == 0 and
                        .prefill_tokens == .prompt_tokens)),
                incremental_second_turn_used:
                    ($incremental[0].turns[1].incremental_kv_continuation and
                     $incremental[0].turns[1].reused_prompt_tokens > 0 and
                     $incremental[0].turns[1].prefill_tokens <
                        $incremental[0].turns[1].prompt_tokens and
                     $incremental[0].turns[1].prefill_tokens +
                        $incremental[0].turns[1].reused_prompt_tokens ==
                        $incremental[0].turns[1].prompt_tokens)
            }
        }
        | .acceptance_pass = ([.gates[]] | all)
    ' >"${repetition_dir}/comparison.json"
done

jq -s '
    def median: sort | .[length / 2 | floor];
    {
        repetitions: .,
        full_second_turn_prefill_median_seconds:
            ([.[].full.turns[1].prefill_seconds] | median),
        incremental_second_turn_prefill_median_seconds:
            ([.[].incremental.turns[1].prefill_seconds] | median),
        acceptance_pass: ([.[].acceptance_pass] | all)
    }
' "${result_dir}"/repetition-*/comparison.json >"${result_dir}/summary.json"

jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
printf 'summary: %s\n' "${result_dir}/summary.json"
