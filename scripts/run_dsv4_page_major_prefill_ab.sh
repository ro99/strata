#!/usr/bin/env bash
# Prompt-throughput A/B for page-major physical prefill at the production
# operating point. Arms are interleaved and repeated; no diagnostic trace is
# enabled, because the trace path disables the fused branch the production
# path uses.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-page-major-prefill-ab"}
runner=${RUNNER:-"${repo_root}/build-pagemajor/strata-deepseek-run"}
devices=${DEVICES:-"1,2"}
page_tokens=${PAGE_TOKENS:-64}
repetitions=${REPETITIONS:-3}
prompt_words=${PROMPT_WORDS:-360}

mkdir -p "${result_dir}"

# A deterministic prompt of roughly 500 tokens. Word choice is fixed so every
# arm and every repetition prefills exactly the same route sequence.
prompt=""
words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)
for ((index=0; index<prompt_words; ++index)); do
    prompt+="${words[index % ${#words[@]}]} "
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
        --detailed-timing \
        --quiet \
        --json \
        >"${result_dir}/${name}.json" \
        2>"${result_dir}/${name}.log"
}

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" status --short --branch
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
} >"${result_dir}/system.txt"

for ((repetition=1; repetition<=repetitions; ++repetition)); do
    run_arm "page1-${repetition}" 1
    run_arm "page${page_tokens}-${repetition}" "${page_tokens}"
done

python3 - "${result_dir}" "${page_tokens}" "${repetitions}" <<'PY'
import json, statistics, sys
directory, page_tokens, repetitions = sys.argv[1], sys.argv[2], int(sys.argv[3])

def load(name):
    with open(f"{directory}/{name}.json") as handle:
        text = handle.read()
    return json.loads(text[text.find('{'):])

def phase(document, *path):
    node = document
    for key in path:
        node = node.get(key) or {}
    return node

arms = {}
for arm in ("page1", f"page{page_tokens}"):
    runs = [load(f"{arm}-{index}") for index in range(1, repetitions + 1)]
    prefill = [run["prefill_seconds"] for run in runs]
    moe = [phase(run, "phases", "prefill", "device_moe_runtime") for run in runs]
    graph = [phase(run, "phases", "prefill", "graph") for run in runs]
    cache = [phase(run, "phases", "prefill", "cache") for run in runs]
    cuda = [phase(run, "phases", "prefill", "cuda") for run in runs]
    arms[arm] = {
        "prompt_tokens": runs[0]["prompt_tokens"],
        "prefill_seconds": prefill,
        "prefill_seconds_median": statistics.median(prefill),
        "prefill_tokens_per_second_median":
            runs[0]["prompt_tokens"] / statistics.median(prefill),
        "routed_cpu_seconds_median":
            statistics.median(entry.get("routed_cpu_seconds", 0.0) for entry in moe),
        "routed_gate_up_seconds_median":
            statistics.median(entry.get("routed_gate_up_seconds", 0.0) for entry in moe),
        "routed_down_seconds_median":
            statistics.median(entry.get("routed_down_seconds", 0.0) for entry in moe),
        "host_callbacks": moe[0].get("host_callback_batches", 0),
        "routed_experts": moe[0].get("routed_experts", 0),
        "attention_seconds_median":
            statistics.median(entry.get("attention_seconds", 0.0) for entry in graph),
        "mhc_post_seconds_median":
            statistics.median(entry.get("mhc_post_seconds", 0.0) for entry in graph),
        "moe_seconds_median":
            statistics.median(entry.get("moe_seconds", 0.0) for entry in graph),
        "maximum_page_tokens": graph[0].get("prefill_max_page_tokens", 0),
        "demand_h2d_bytes": cache[0].get("demand_h2d_bytes", 0),
        "demand_wait_seconds_median":
            statistics.median(entry.get("demand_wait_seconds", 0.0) for entry in cache),
        "cache_hits": cache[0].get("hits", 0),
        "cache_misses": cache[0].get("misses", 0),
        "cache_evictions": cache[0].get("evictions", 0),
        "checkpoint_read_bytes":
            runs[0]["phases"]["prefill"].get("checkpoint_read_bytes", 0),
        "weight_h2d_bytes": cuda[0].get("weight_h2d_bytes", 0),
        "rss_bytes": runs[0].get("rss_bytes", 0),
        "device_vram_used_bytes": runs[0].get("device_vram_used_bytes", []),
        "generated_token_ids": runs[0].get("generated_token_ids", []),
    }

reference = arms["page1"]
candidate = arms[f"page{page_tokens}"]
summary = {
    "arms": arms,
    "speedup": reference["prefill_seconds_median"] /
               candidate["prefill_seconds_median"],
    "routed_cpu_speedup":
        (reference["routed_cpu_seconds_median"] /
         candidate["routed_cpu_seconds_median"])
        if candidate["routed_cpu_seconds_median"] else None,
    "generated_tokens_equal":
        reference["generated_token_ids"] == candidate["generated_token_ids"],
}
with open(f"{directory}/summary.json", "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY
