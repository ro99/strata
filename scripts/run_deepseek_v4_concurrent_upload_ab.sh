#!/usr/bin/env bash
# A/B: concurrent routed-expert demand upload against the serial one.
#
# Reference arm is --serial-expert-upload, which waits out each upload where it
# is issued -- the pre-0052 behaviour. Treatment arm is the default, which
# leaves a layer's copies in flight and waits once per device. Identical bytes,
# identical routing, identical numerics: the generated token sequences must
# match exactly, and the script fails if they do not.
#
# Arms are interleaved so thermal and page-cache drift cancels. Budget at the
# chat operating point: ~90 s per arm (about 70 s of it model staging, ~15 s of
# measured decode), 2 arms x 3 repetitions = 6 arms, roughly 9 minutes total.
# The 64-token decode window is the part under test; it is not shortened
# further because three repetitions of a ~15 s window is what makes the median
# meaningful against the observed ~5% spread.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir=${RESULT_DIR:-"${repo_root}/results/deepseek-v4-concurrent-upload-ab"}
repetitions=${REPETITIONS:-3}
base_flags=${BASE_FLAGS:---flash-attention --pin-resident-arena}
detail=${DETAIL---detailed-timing}

export RESULT_DIR="${result_dir}"
export DETAIL="${detail}"
export MAX_NEW=${MAX_NEW:-64}

for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    for arm in serial concurrent; do
        if [[ ${arm} == serial ]]; then
            flags="${base_flags} --serial-expert-upload"
        else
            flags="${base_flags}"
        fi
        ARM="${arm}" FLAGS="${flags}" REPETITIONS=1 \
            RUN_OFFSET="$((repetition - 1))" \
            "${repo_root}/scripts/run_deepseek_v4_decode_profile.sh" \
            >/dev/null
        echo "done ${arm} repetition ${repetition}"
    done
done

python3 "${repo_root}/scripts/summarize_concurrent_upload_ab.py" "${result_dir}"
