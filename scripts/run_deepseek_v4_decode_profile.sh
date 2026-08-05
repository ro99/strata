#!/usr/bin/env bash
# Decode subphase profile for DeepSeek-V4-Flash at the chat operating point.
#
# Operating point is experiment 0034's, reproduced so numbers are comparable:
# 18-token chat prompt, three GPUs, 216 GiB host ceiling, 0.95 VRAM fraction,
# 32,768-token context ceiling. Decode tokens are a parameter because decode is
# what is under test and setup is fixed cost: at ~250 ms/step, 64 tokens is 16 s
# of measured window against ~60 s of setup.
#
# Usage: ARM=name FLAGS="--flash-attention ..." scripts/run_deepseek_v4_decode_profile.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner=${RUNNER:-"${repo_root}/build/strata-deepseek-run"}
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-decode-profile"}

arm=${ARM:-baseline}
flags=${FLAGS:---flash-attention --pin-resident-arena}
max_new=${MAX_NEW:-64}
max_context=${MAX_CONTEXT:-32768}
vram_fraction=${VRAM_FRACTION:-0.95}
host_memory=${HOST_MEMORY:-216G}
repetitions=${REPETITIONS:-1}
# Unset means detailed timing; DETAIL= means explicitly none. The distinction
# matters: detailed timing adds about 2,000 synchronizing API calls per step,
# which is most of the step at this operating point, so an A/B that leaves it
# on is not measuring the production configuration.
detail=${DETAIL---detailed-timing}

# Experiment 0034's prompt, verbatim, so the operating point is the same one.
prompt=${PROMPT:-"what is the closes star to us, and how far is it?"}

mkdir -p "${result_dir}"

# RUN_OFFSET lets an interleaving driver place a single repetition into an
# arbitrary run slot, so arms can alternate instead of running back to back.
run_offset=${RUN_OFFSET:-0}

for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    name="${arm}/run-$(printf '%02d' "$((repetition + run_offset))")"
    mkdir -p "${result_dir}/${name}"
    printf 'arm=%s\nflags=%s\nmax_new=%s\ndetail=%s\n' \
        "${arm}" "${flags}" "${max_new}" "${detail}" \
        >"${result_dir}/${name}/arm.txt"
    # shellcheck disable=SC2086
    "${runner}" \
        --model "${model_dir}" --devices 0,1,2 \
        --host-memory "${host_memory}" --vram-fraction "${vram_fraction}" \
        --max-context "${max_context}" \
        --max-new "${max_new}" --prompt "${prompt}" \
        ${flags} ${detail} --quiet --json \
        >"${result_dir}/${name}/generation.json" \
        2>"${result_dir}/${name}/generation.log" || {
            echo "arm ${name} failed; see ${result_dir}/${name}/generation.log" >&2
            exit 1
        }
    echo "done ${name}"
done
