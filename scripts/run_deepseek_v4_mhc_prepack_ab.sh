#!/usr/bin/env bash
set -euo pipefail

# Exact mHC projection-layout A/B at the single-stream chat operating point.
# The 18-token prompt keeps setup/prefill short; decode is the term under test.
# Budget at 152 tokens: about 2 minutes/arm, 6 arms, about 12 minutes total.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-mhc-prepack-ab"}
maximum_new_tokens=${MAX_NEW_TOKENS:-152}
repetitions=${REPETITIONS:-3}
trace=${TRACE:-0}
minimum_speedup=${MIN_SPEEDUP:-1.05}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}

if [[ -n "$(git -C "${repo_root}" status --porcelain)" ]]; then
    echo "error: A/B evidence requires a clean frozen revision" >&2
    exit 1
fi

mkdir -p "${result_dir}"
cp --reflink=auto "${runner}" "${result_dir}/strata-deepseek-run"
runner="${result_dir}/strata-deepseek-run"

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    printf 'maximum_new_tokens=%s\nrepetitions=%s\ntrace=%s\nminimum_speedup=%s\n' \
        "${maximum_new_tokens}" "${repetitions}" "${trace}" "${minimum_speedup}"
    sha256sum "${runner}"
    nvidia-smi --query-gpu=index,name,pci.bus_id,memory.free,memory.total \
        --format=csv
} >"${result_dir}/system.txt"

run_case() {
    local arm=$1 repetition=$2
    local name="${arm}/run-$(printf '%02d' "${repetition}")"
    local -a candidate=() diagnostics=()
    [[ "${arm}" == candidate ]] && candidate=(--prepack-mhc)
    [[ "${trace}" == 1 ]] && diagnostics=(--logit-trace --layer-hash-trace)
    mkdir -p "${result_dir}/${name}"
    /usr/bin/time -v "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory 216G --vram-fraction 0.95 \
        --max-context 32768 --max-new "${maximum_new_tokens}" \
        --prompt "${prompt}" --flash-attention --pin-resident-arena \
        --route-trace "${result_dir}/${name}/routes.jsonl" \
        --detailed-timing --quiet --json \
        "${diagnostics[@]}" "${candidate[@]}" \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log"
}

for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    if ((repetition % 2 == 1)); then
        run_case reference "${repetition}"
        run_case candidate "${repetition}"
    else
        run_case candidate "${repetition}"
        run_case reference "${repetition}"
    fi
    cmp "${result_dir}/reference/run-$(printf '%02d' "${repetition}")/routes.jsonl" \
        "${result_dir}/candidate/run-$(printf '%02d' "${repetition}")/routes.jsonl"
done

jq -s '.' "${result_dir}"/reference/run-*/generation.json >"${result_dir}/reference.json"
jq -s '.' "${result_dir}"/candidate/run-*/generation.json >"${result_dir}/candidate.json"

jq -n \
  --argjson trace "$([[ "${trace}" == 1 ]] && echo true || echo false)" \
  --argjson minimum_speedup "${minimum_speedup}" \
  --slurpfile r "${result_dir}/reference.json" \
  --slurpfile c "${result_dir}/candidate.json" '
  def median: sort | .[(length - 1) / 2 | floor];
  def med($xs; f): [$xs[0][] | f] | median;
  {
    repetitions: ($r[0] | length),
    median: {
      reference_tok_s: med($r; .decode_steps / .decode_seconds),
      candidate_tok_s: med($c; .decode_steps / .decode_seconds),
      reference_decode_seconds: med($r; .decode_seconds),
      candidate_decode_seconds: med($c; .decode_seconds),
      reference_mhc_pre_ms_per_step:
        med($r; 1000 * .phases.decode.graph.mhc_pre_seconds / .decode_steps),
      candidate_mhc_pre_ms_per_step:
        med($c; 1000 * .phases.decode.graph.mhc_pre_seconds / .decode_steps),
      reference_demand_wait_ms_per_step:
        med($r; 1000 * .phases.decode.cache.demand_wait_seconds / .decode_steps),
      candidate_demand_wait_ms_per_step:
        med($c; 1000 * .phases.decode.cache.demand_wait_seconds / .decode_steps)
    },
    memory: {
      reference_rss_bytes: med($r; .rss_bytes),
      candidate_rss_bytes: med($c; .rss_bytes),
      prepack_bytes: $c[0][0].memory_plan.mhc_prepack_bytes
    },
    gates: {
      flag_and_calls_exercised:
        ([$r[0][] | .prepack_mhc == false and
                     .phases.decode.graph.mhc_prepacked_calls == 0] | all) and
        ([$c[0][] | .prepack_mhc == true and
                     .phases.decode.graph.mhc_prepacked_calls ==
                         (86 * .decode_steps)] | all),
      generated_tokens_equal:
        ([range(0; $r[0] | length) as $i |
          $r[0][$i].generated_token_ids == $c[0][$i].generated_token_ids] | all),
      diagnostics_equal:
        (($trace | not) or
         ([range(0; $r[0] | length) as $i |
           $r[0][$i].diagnostics == $c[0][$i].diagnostics] | all)),
      zero_decode_checkpoint_reads:
        ([$r[0][], $c[0][] | .decode_checkpoint_read_bytes == 0] | all),
      exact_memory_accounting:
        ($r[0][0].memory_plan.mhc_prepack_bytes == 0 and
         $c[0][0].memory_plan.mhc_prepack_bytes == 135266304 and
         $c[0][0].memory_plan.required_host_bytes ==
             ($r[0][0].memory_plan.required_host_bytes + 135266304)),
      within_host_ceiling: ([$r[0][], $c[0][] | .rss_bytes <= 216 * 1073741824] | all),
      material_speedup:
        ((med($c; .decode_steps / .decode_seconds) /
          med($r; .decode_steps / .decode_seconds)) >= $minimum_speedup)
    }
  }
  | .speedup = (.median.candidate_tok_s / .median.reference_tok_s)
  | .acceptance_pass = ([.gates[]] | all)
  ' >"${result_dir}/summary.json"

cat "${result_dir}/summary.json"
jq -e '.acceptance_pass == true' "${result_dir}/summary.json" >/dev/null
