#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-0106-numa-attribution"}
model_dir=${MODEL_DIR:-"/home/rodrigo/Developer/strata/models/dsv4f"}
runner=${RUNNER:-"${repo_root}/results/dsv4-0105-fp8-tensor-projections/strata-deepseek-run"}

mkdir -p "${result_dir}"
for output in attribution.json attribution.log numa-maps.log thread-cpus.log; do
    if [[ -e "${result_dir}/${output}" ]]; then
        echo "refusing to overwrite preserved ${output}" >&2
        exit 2
    fi
done

words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)
prompt=""
for ((index=0; index<420; ++index)); do
    prompt+="${words[index % ${#words[@]}]} "
done

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    sha256sum "${runner}"
    nvidia-smi topo -m
    numactl --hardware
} >"${result_dir}/system.txt"

"${runner}" \
    --model "${model_dir}" \
    --devices 1,2 \
    --host-memory 216G \
    --vram-fraction 0.95 \
    --max-context 4096 \
    --device-resident-runtime \
    --decode-topology rank-local-tp2 \
    --prefill-page-tokens 8192 \
    --max-new 4 \
    --prompt "${prompt}" \
    --detailed-timing \
    --quiet \
    --json \
    --dsv4-fp8-tensor-page \
    >"${result_dir}/attribution.json" \
    2>"${result_dir}/attribution.log" &
model_pid=$!
printf '%s\n' "${model_pid}" >"${result_dir}/pid.txt"

while kill -0 "${model_pid}" 2>/dev/null; do
    process_state=$(awk '{print $3}' "/proc/${model_pid}/stat" 2>/dev/null || true)
    [[ "${process_state}" != "Z" ]] || break
    timestamp=$(date +%s.%N)
    if [[ -r "/proc/${model_pid}/numa_maps" ]]; then
        awk -v timestamp="${timestamp}" '
            /anon=/ {
                anon = 0
                n0 = 0
                n1 = 0
                for (field = 1; field <= NF; ++field) {
                    if ($field ~ /^anon=/) {
                        split($field, pair, "=")
                        anon = pair[2]
                    } else if ($field ~ /^N0=/) {
                        split($field, pair, "=")
                        n0 = pair[2]
                    } else if ($field ~ /^N1=/) {
                        split($field, pair, "=")
                        n1 = pair[2]
                    }
                }
                if (anon >= 100000) {
                    print timestamp, "anon_pages=" anon, "N0=" n0,
                          "N1=" n1, $0
                }
            }
        ' "/proc/${model_pid}/numa_maps" >>"${result_dir}/numa-maps.log"
    fi
    for stat_file in /proc/"${model_pid}"/task/*/stat; do
        [[ -r "${stat_file}" ]] || continue
        awk -v timestamp="${timestamp}" '{
            cpu = $39
            node = (cpu >= 14 && cpu <= 27) || (cpu >= 42 && cpu <= 55) ? 1 : 0
            print timestamp, "tid=" $1, "cpu=" cpu, "node=" node
        }' "${stat_file}" >>"${result_dir}/thread-cpus.log"
    done
    sleep 0.25
done

wait "${model_pid}"
