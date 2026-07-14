Implemented and executed successfully on all three GPUs.

### Baseline results

30-token prompt, 128 generated tokens per run:

| Run | Prefill tok/s | Decode tok/s | Physical NVMe reads |
|---|---:|---:|---:|
| 1 | 0.514 | 0.292 | 210.3 GB |
| 2 | 0.604 | 0.308 | 163.1 GB |
| 3 | 0.587 | 0.307 | 146.8 GB |
| Median | **0.587** | **0.307** | **163.1 GB** |

Additional median measurements:

- Initialization/resident load: 23.55 seconds
- Requested checkpoint reads: 906.9 GB
- Weight H2D traffic: 906.9 GB
- VRAM cache hit rate: approximately 59%
- Peak weight-cache usage: 13.0 GiB + 19.8 GiB + 19.8 GiB
- Process RSS: approximately 2.27 GB; host RAM primarily operated as filesystem page cache

The machine has approximately 64 GB raw VRAM across the three GPUs; the 256 GB figure corresponds to host memory.

### Inference answer

The longer diagnostic generation reached:

> The closest star to the Sun is Proxima Centauri, approximately 4.24 light-years away—about 40 trillion kilometers or 25 trillion miles.

The model continued a verbose reasoning/formulation section and did not emit EOS before the 220-token diagnostic ceiling.

### Correctness status

- Full 128-shard checkpoint validated: 177,569 tensors and 405,459,090,304 payload bytes.
- Real INT4 group-128 CUDA oracle: bit-identical on all three GPUs.
- Real INT8 group-128 oracle: maximum error 1.19×10⁻⁶.
- Real MTP channelwise INT8 oracle: bit-identical.
- `make check`: passed.

One important caveat: runs 1 and 3 produced identical tokens, while run 2 diverged at generated token 93. The diagnostic majority branch had a 0.700 top-two logit margin at that position. Therefore these numbers are a useful machine baseline, but provisional—not yet suitable for claiming an optimization win until fixed-token teacher forcing and sequential route traces isolate the nondeterminism.

Key files:

- [GLM runtime](/home/rodrigo/Developer/strata/src/glm_runtime.cpp:406)
- [CUDA kernels](/home/rodrigo/Developer/strata/kernels/cuda/backend.cu:71)
- [Inference CLI](/home/rodrigo/Developer/strata/apps/strata_run.cpp:143)
- [Benchmark script](/home/rodrigo/Developer/strata/scripts/run_glm52_baseline.sh:87)
- [Three-run summary](/home/rodrigo/Developer/strata/results/glm52-baseline/summary.json)
- [Longer answer run](/home/rodrigo/Developer/strata/results/glm52-answer/run-01.json)

The retained tmux session is `strata-glm52-baseline`. Existing unrelated worktree changes were preserved; no commit was made.