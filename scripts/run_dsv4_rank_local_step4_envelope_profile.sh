#!/usr/bin/env bash
# Step 4 attribution of the accepted rank-local short-context dependent
# non-CPU envelope (51.921 ms/token at the step2-acceptance-r2 median).
#
# The handoff requires the cheapest production-shaped measurement that can
# decompose that envelope. Aggregate rank-local counters cannot separate
# dependent GPU kernel time from cross-engine handoff gaps, and the 0087 trace
# predates the accepted queued topology. One Nsight Systems capture of the
# accepted production arm does both, and needs no hot-path instrumentation.
#
# Two arms at the identical accepted operating point:
#   control  unprofiled, establishes this arm's own steady median
#   profile  same command under nsys, so the perturbation is measured and
#            reported rather than assumed (0079: per-layer event arms perturbed
#            the chain by 4-5 ms and were rejected as causal evidence)
#
# Budget: initialization dominates and cannot be shortened -- the checkpoint is
# 156 GB. Each arm is about 126 s initialization, 3.4 s prefill and 1.4 s of
# measured decode, so roughly 2.5 min per arm and under 10 min in total.
# Decode is shortened from the acceptance runner's 32 tokens to 12 because
# attribution needs steady-state proportions, not a new timing gate.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner=${RUNNER:-"${repo_root}/build-landing-nccl/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-rank-local-main-landing/step4-envelope-profile"}
rank_local_oracle=${RANK_LOCAL_ORACLE:-"${repo_root}/results/dsv4-rank-local-main-landing/step2-sequential-oracle-32/sequential.json"}
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}
devices=${DEVICES:-1,2}
max_new=${MAX_NEW:-12}

mkdir -p "${result_dir}"

if [[ ! -f "${rank_local_oracle}" ]]; then
    printf 'missing rank-local sequential oracle: %s\n' "${rank_local_oracle}" >&2
    exit 2
fi

run_args=(
    --model "${model_dir}"
    --prompt "${prompt}"
    --max-new "${max_new}"
    --devices "${devices}"
    --max-context 4096
    --host-memory 216G
    --vram-fraction 0.95
    --device-resident-runtime
    --decode-topology rank-local-tp2
    --detailed-timing --quiet --json
)

# Every profiled arm must still produce the accepted token sequence. The
# oracle holds 32 IDs; this arm produces a prefix of it, so compare the prefix.
validate_arm() {
    local stem=$1
    jq -e --slurpfile oracle "${rank_local_oracle}" \
        --argjson tokens "${max_new}" '
        (.decode_steps == ($tokens - 1)) and
        ((.generated_token_ids | length) == $tokens) and
        (.decode_checkpoint_read_bytes == 0) and
        (.generated_token_ids == ($oracle[0].generated_token_ids[0:$tokens]))
    ' "${stem}.json" >/dev/null
}

steady_median_ms() {
    jq -r '[.decode_step_seconds[1:][] * 1000] | sort as $s |
           if ($s | length) % 2 == 1 then $s[($s|length)/2|floor]
           else ($s[($s|length)/2-1] + $s[($s|length)/2]) / 2 end' \
        "$1.json"
}

control_stem="${result_dir}/control"
if [[ ! -s "${control_stem}.json" ]]; then
    set +e
    "${runner}" "${run_args[@]}" \
        >"${control_stem}.json" 2>"${control_stem}.log"
    printf '%d\n' $? >"${control_stem}.exit"
    set -e
fi
[[ "$(cat "${control_stem}.exit")" == 0 ]] || exit 3
validate_arm "${control_stem}"

profile_stem="${result_dir}/profile"
if [[ ! -s "${profile_stem}.json" ]]; then
    set +e
    # nsys writes its own progress banner to stdout, so the runner's JSON is
    # redirected inside the traced process rather than around nsys. Sharing one
    # stream produces a file that is not valid JSON.
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --cuda-memory-usage=false \
        --force-overwrite=true \
        --output "${result_dir}/trace" \
        bash -c 'exec "$@" > "$0"' \
            "${profile_stem}.json" "${runner}" "${run_args[@]}" \
        >"${profile_stem}.nsys.log" 2>"${profile_stem}.log"
    printf '%d\n' $? >"${profile_stem}.exit"
    set -e
fi
[[ "$(cat "${profile_stem}.exit")" == 0 ]] || exit 4
validate_arm "${profile_stem}"

nsys export --type sqlite --force-overwrite=true \
    --output "${result_dir}/trace.sqlite" "${result_dir}/trace.nsys-rep" \
    >"${result_dir}/export.log" 2>&1

{
    printf 'rank-local Step 4 dependent-envelope attribution\n'
    printf 'runner sha256   %s\n' \
        "$(sha256sum "${runner}" | cut -d' ' -f1)"
    printf 'decode tokens   %s\n' "${max_new}"
    printf 'control steady median ms/token   %s\n' \
        "$(steady_median_ms "${control_stem}")"
    printf 'profiled steady median ms/token  %s\n' \
        "$(steady_median_ms "${profile_stem}")"
} >"${result_dir}/summary.txt"

sha256sum "${result_dir}"/control.json "${result_dir}"/profile.json \
    "${result_dir}"/trace.nsys-rep >"${result_dir}/sha256.txt"

cat "${result_dir}/summary.txt"
