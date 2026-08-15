#!/usr/bin/env bash
# Prefill throughput sweep at the reference stack's own prompt lengths. Each
# arm is one process, so model setup is not in the measured window; the window
# is prefill only. PAGES and PROMPT_WORDS select the grid.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=${MODEL_DIR:-"${repo_root}/models/dsv4f"}
result_dir=${RESULT_DIR:-"${repo_root}/results/dsv4-prefill-sweep"}
runner=${RUNNER:-"${repo_root}/build-pagemajor/strata-deepseek-run"}
devices=${DEVICES:-"1,2"}
pages=${PAGES:-"64 512 4096"}
prompt_words=${PROMPT_WORDS:-420}
context=${CONTEXT:-16384}
extra=${EXTRA:-}

mkdir -p "${result_dir}"

words=(alpha bravo charlie delta echo foxtrot golf hotel india juliet
       kilo lima mike november oscar papa quebec romeo sierra tango)
prompt=""
for ((index=0; index<prompt_words; ++index)); do
    prompt+="${words[index % ${#words[@]}]} "
done

{
    date --iso-8601=seconds
    git -C "${repo_root}" rev-parse HEAD
    nvidia-smi --query-gpu=index,name,memory.free --format=csv
} >"${result_dir}/system.txt"

for page in ${pages}; do
    name="page${page}-w${prompt_words}"
    echo "arm ${name}"
    # shellcheck disable=SC2086
    "${runner}" \
        --model "${model_dir}" \
        --devices "${devices}" \
        --host-memory 216G \
        --vram-fraction 0.95 \
        --max-context "${context}" \
        --device-resident-runtime \
        --decode-topology rank-local-tp2 \
        --prefill-page-tokens "${page}" \
        --max-new 1 \
        --prompt "${prompt}" \
        --detailed-timing \
        --quiet \
        --json ${extra} \
        >"${result_dir}/${name}.json" \
        2>"${result_dir}/${name}.log" || {
            echo "  FAILED: $(tail -1 "${result_dir}/${name}.log")"
            continue
        }
    python3 - "${result_dir}/${name}.json" "${name}" <<'PY'
import json, sys
path, name = sys.argv[1], sys.argv[2]
text = open(path).read()
d = json.loads(text[text.find('{'):])
p = d["phases"]["prefill"]
g = p.get("graph") or {}
m = p.get("device_moe_runtime") or {}
c = p.get("cache") or {}
seconds = d["prefill_seconds"]
print(f'  {name}: {d["prompt_tokens"]} tok  {seconds:.2f} s  '
      f'{d["prompt_tokens"]/seconds:.1f} tok/s')
print(f'    attention {g.get("attention_seconds",0):.2f}  '
      f'moe {g.get("moe_seconds",0):.2f}  '
      f'routed_cpu {m.get("routed_cpu_seconds",0):.2f}  '
      f'mhc_post {g.get("mhc_post_seconds",0):.2f}')
print(f'    expert H2D {c.get("demand_h2d_bytes",0)/1e9:.1f} GB  '
      f'wait {c.get("demand_wait_seconds",0):.2f} s  '
      f'misses {c.get("misses",0)}  evictions {c.get("evictions",0)}  '
      f'max page {g.get("prefill_max_page_tokens",0)}')
PY
done
