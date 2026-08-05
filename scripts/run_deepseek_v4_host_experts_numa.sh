#!/usr/bin/env bash
# Production host-experts run with NUMA placement sampling.
# Reports the in-situ host MoE rate and the arena's node placement.
set -uo pipefail

repo_root=/home/rodrigo/Developer/strata
result_dir=${repo_root}/results/dsv4-host-experts-numa
mkdir -p "${result_dir}"

sentences=(
    "The harbour master kept a ledger of every vessel that crossed the bar at dawn."
    "Salt crusted the iron railings and the gulls argued over a torn net."
    "In the archive room a clerk copied tide tables onto onionskin paper."
    "A surveyor measured the drift of the sandbank against last winter's chart."
    "The lighthouse keeper replaced the mantle and trimmed the wick before dusk."
    "Fishermen mended trawl lines while the tide ran out across the flats."
    "An engineer argued that the breakwater needed a deeper foundation."
    "The customs officer weighed a crate of tin and signed the manifest twice."
    "Rain moved in from the west and the barometer fell through the evening."
    "A cartographer redrew the channel where the current had cut a new mouth."
    "The pilot boat waited beyond the buoys for the grain ship to answer."
    "Children counted the bells that marked each watch from the seawall."
)
prompt_sentences=${PROMPT_SENTENCES:-34}
index=0
text=""
while ((index < prompt_sentences)); do
    text+="${sentences[index % ${#sentences[@]}]} "
    ((++index))
done

runner=${result_dir}/strata-deepseek-run
cp --reflink=auto "${repo_root}/build/strata-deepseek-run" "${runner}"

args=(
    --model "${repo_root}/models/dsv4f"
    --devices 0,1,2
    --host-memory 216G
    --vram-fraction 0.95
    --max-context 32768
    --max-new 64
    --flash-attention
    --pin-resident-arena
    --host-experts
    --quiet
    --json
    --prompt "${text}"
)

"${runner}" "${args[@]}" > "${result_dir}/run.json" 2> "${result_dir}/run.err" &
bench_pid=$!
echo "benchmark pid=$bench_pid"

# Sample the arena's node placement while the model is resident.
for i in $(seq 1 60); do
    if ! kill -0 "${bench_pid}" 2>/dev/null; then break; fi
    sample=$(grep -E "^[0-9a-f]+ " /proc/${bench_pid}/numa_maps 2>/dev/null | \
        python3 -c '
import re, sys
n=[0,0,0,0]
total=0
for line in sys.stdin:
    total+=1
    for tok in ("N0=","N1=","N2=","N3="):
        at=line.find(tok)
        if at>=0:
            n[int(tok[1])]+=int(re.match(r"\d+", line[at+3:]).group())
print("N0=%d N1=%d N2=%d N3=%d vmas=%d" % (n[0],n[1],n[2],n[3],total))
' 2>/dev/null)
    echo "t=$((i*5))s ${sample}" >> "${result_dir}/numa_samples.log"
    sleep 5
done

wait "${bench_pid}"
rc=$?
echo "benchmark rc=${rc}"
if [[ -s "${result_dir}/run.json" ]]; then
    python3 -c '
import json
d=json.load(open("'${result_dir}'/run.json"))
dev=d.get("device_moe_runtime") or {}
print("decode_seconds=%.2f steps=%d" % (d.get("decode_seconds",0), d.get("decode_steps",0)))
print("device_moe_runtime:", json.dumps(dev))
dec=d.get("phases",{}).get("decode",{}).get("graph",{})
print("decode moe_seconds=%.3f attention_seconds=%.3f mhc_pre=%.3f mhc_post=%.3f" % (
    dec.get("moe_seconds",0), dec.get("attention_seconds",0),
    dec.get("mhc_pre_seconds",0), dec.get("mhc_post_seconds",0)))
print("resident_stage_seconds=%.2f rss=%d" % (d.get("resident_staging_seconds",0), d.get("rss_bytes",0)))
' 2>&1
fi
