# DeepSeek's Long March

## Timeline

V1 (early 2024) was a 67B dense model with conventional architecture and no MoE. It was a useful first model but not architecturally differentiated from the rest of the field, and the team was primarily using it to build experience with the training and serving infrastructure.

V2 (May 2024) is where the architecture changed. DeepSeek-V2 introduced Multi-head Latent Attention (MLA) [2], which compresses the KV cache into a low-rank latent vector, and combined it with DeepSeekMoE, their sparse expert architecture. The important technical numbers from V2 were 236B total parameters with 21B active per token, a 128K context window, a 93.3% reduction in KV cache size, and 5.76x higher generation throughput compared to a dense 67B baseline. This is the point where the lab stopped optimizing for model size and started optimizing for inference cost.

V3 (December 2024) scaled the same approach to 671B total parameters with 37B active per token and pre-trained the entire forward and backward pass in FP8 from step zero [3]. The full training run took 2.788M H800 GPU-hours, which made V3 not the largest model in the world at the time but the most efficient frontier-scale training run anyone had published, and a direct counter to the "we need more H100s" story that dominated US AI conversation at the time.

R1 (January 2025) is when DeepSeek became widely known outside AI research. R1 [4] showed that reasoning capabilities could be incentivized through pure reinforcement learning without any labeled chain-of-thought data, and AIME 2024 pass@1 jumped from 15.6% to 71.0% as a result. Closed-API prices started dropping within weeks of the release, which made R1 function as a market event in addition to a model release.

V3.1 / V3.2-Exp (September 2025) introduced DeepSeek Sparse Attention (DSA) [5], the architectural change that cut the quadratic cost of long context down to roughly linear. V3.2-Exp was explicitly experimental, but the path to V4 was already obvious to anyone watching the arXiv feed.

V4 (April 24, 2026) [1] ships DSA as the production default, scales the MoE to 1.6T / 49B active, and makes 1M context the default tier across web, app, and API, available at the baseline pricing rather than as a premium add-on.

The six releases all serve the same underlying thesis, which is to make the parts of the model that are not doing work as small and as skippable as possible, and then scale the parts that remain.

## Start with MLA, not DSA

Multi-head Latent Attention (MLA) [2] shipped with DeepSeek-V2 in May 2024, two years before V4. Every later DeepSeek model is built on top of it, DSA included. If you want to understand why DeepSeek's inference cost keeps dropping while other labs' inference cost stays the same, this is the architectural choice to look at first.

The problem MLA solves: in a standard transformer with Multi-Head Attention (MHA), every attention head stores its own K and V vector per token in the KV cache. A 70B-class model with 64 heads stores 128 cache entries per token. Run that out to 128K context and the cache is in the hundreds of gigabytes per serving session. An H100 with 80GB of HBM (high-bandwidth memory) can fit maybe one conversation, which is the reason MLA was necessary as an architectural change rather than a serving optimization.

Llama 3 partially addressed this with Grouped-Query Attention (GQA), where several query heads share a single K/V head. DeepSeek-V2 used a different approach. It compresses K and V together into a single low-rank latent vector per token, and reconstructs the per-head K and V on demand at attention time.

A few non-obvious things follow from this design choice.

1M context is economically viable because of MLA, not DSA. Pre-MLA, the KV cache at a million tokens would have been hundreds of gigabytes per serving session, while the post-MLA cache is roughly a fifteenth of that size. MLA is the foundation that makes everything else affordable at long context, and DSA only delivers its full benefit when it sits on top of an already-compressed cache.

DSA works because MLA already shrunk the cache. DSA in V3.2 and V4 runs directly on top of MLA in multi-query mode. The top-k selector operates over the latent representations, and the lightning indexer can run cheaply in FP8 specifically because the cache is already compressed and shared across query heads. Apply DSA to standard MHA instead, and the indexer has to re-rank every head independently; the kernel implementation becomes substantially harder to make efficient.

Adding MLA support is the hardest part of porting any DeepSeek model. Production MLA kernels in vLLM and SGLang landed over a substantial multi-release effort after V2 shipped rather than in a single drop, because MLA changes the memory layout, the projection geometry, and the math at the matmul boundary all at once.

In our own work porting MLA-class attention to the Spyre AIU, the friction point we ran into is that MLA's pattern of storing a compressed latent and projecting up to per-head K and V at attention time does not map cleanly onto the cache-and-attend kernel shapes that exist in most serving runtimes. The fused decompress-then-project step has to be a new primitive, with its own tiling decisions and its own numerics, and that one change ripples through the inference graph in several places at once: the memory layout for the latent cache, the way the indexer reads it in DSA mode, and the placement of the dequantize-and-project work relative to the softmax. MLA is where most of the kernel engineering goes when a team is bringing up a non-NVIDIA accelerator for these models.

## What DSA actually does, and why it is a hardware story

Sparse attention has a long history that includes Longformer, BigBird, and the sliding-window patterns used in Gemma 3 and Olmo 3. The DeepSeek contribution is not the underlying idea but rather two specific engineering choices that make DSA, to my knowledge, one of the first openly documented sparse-attention systems deployed as the default path at frontier scale.

The two-stage indexer. Standard attention computes a similarity score between each token and every other token. DSA replaces the expensive part with a fast scoring pass: a small indexer model running in FP8 computes a quick relevance score between the current token and every prior token. A top-k selector then picks the k=2,048 highest-scoring tokens. Expensive full attention runs only over those.

The complexity story is that standard attention is O(L²) in sequence length, while DSA reduces the dominant quadratic behavior by running the expensive softmax-and-matmul work over only ~2,048 tokens instead of all L. There is still O(L) work in the indexer pass and the top-k step, but those run in FP8 at much lower per-token cost, and the math works out to effectively near-linear scaling in practice for the configurations DeepSeek targets.

The hardware co-design. What most coverage misses is that DSA is not a software-only change that any serving stack can adopt. The indexer runs in FP8 because Hopper tensor cores expose an FP8 path. The variable per-query sparsity pattern requires custom CUDA kernels, because standard attention kernels cannot handle "different k tokens per query" efficiently. The whole design is built on top of MLA in multi-query mode so that the KV cache stays compressed across queries. For short contexts it falls back to dense attention because the indexer overhead is not worth paying below a threshold length.

That last detail matters more than it sounds. DSA uses sparse attention when sparse attention is faster and dense attention when it is not, and the switch happens inside a kernel that is designed for the accelerator's memory hierarchy. Retrofitting this behavior into vLLM or TensorRT-LLM requires substantial kernel-level changes to those runtimes, which is not impossible but is significantly more involved than flipping a configuration flag.

DSA shipped first in V3.2-Exp in September 2025, was hardened in V3.2 in December 2025, and continued to evolve through V4 in April 2026. The seven-month progression from experimental release to production default is the kind of cadence that comes from a team that treats kernel engineering as production work rather than as paper figures.

## What V4 actually changed beyond V3.2's DSA

The description above is V3.2-class DSA, which is the foundation that V4 builds on. V4 went further than V3.2, and the official V4 technical report describes a hybrid attention scheme that interleaves two related mechanisms across the layers of the model. (A note on what is documented versus inferred: the mechanism descriptions below are taken from DeepSeek's V4 technical report. Some of the serving-stack implications I draw at the end of this section are my own reading of the architecture rather than something DeepSeek explicitly documented.)

Compressed Sparse Attention (CSA) is the direct descendant of V3.2's DSA, but the indexer now scores over already-compressed entries. A learned token-level compressor with stride 4 condenses every neighborhood of tokens into a single entry, and the lightning indexer picks roughly 128 of those compressed entries per query for the expensive softmax-and-matmul step. The indexer itself moved from FP8 in V3.2 down to FP4 (MXFP4) in V4, with FP4 quantization-aware training keeping accuracy stable. This represents another halving of bytes per indexer activation on top of everything DSA already did.

Heavily Compressed Attention (HCA) is more aggressive. It consolidates KV entries with stride 128 — at 1M context, that turns the cache from ~1M positions into roughly 8K compressed entries. The cache is small enough at that point that dense attention over the entire compressed stream is feasible again. HCA gives the model a holistic, low-resolution view of the full context that complements CSA's precise sparse retrieval over selected regions.

V4-Pro uses HCA for its first two layers and then alternates CSA and HCA through the rest of the model. V4-Flash starts with sliding-window attention before falling into the same CSA/HCA alternation. Different layers do different things: some get the precise sparse retrieval pattern of CSA, others get the holistic global view of HCA.

The practical consequence for anyone building inference infrastructure on top of V4: the cache is now multi-level. Raw KV entries, CSA-compressed entries, and HCA-compressed entries all coexist, and the serving stack has to manage three different staleness and precision tiers. The kernel surface area is meaningfully larger than V3.2's, and the FP4 indexer in particular needs hardware that handles MXFP4 efficiently — which currently means Blackwell or Hopper-class with software emulation.

## Memory bandwidth is the common target

The four innovations cut cost on different axes, but the underlying point of all of them is the same. Modern LLM serving is increasingly memory-bandwidth-bound rather than compute-bound: generating a token spends most of its time streaming weights and KV state from HBM into the compute units, not doing math. Each architectural choice in V4 reduces bytes moved per generated token. MoE activates only a small fraction of the parameters, so fewer weights cross HBM. MLA compresses the KV cache by ~15×, so fewer cache bytes cross HBM per attention step. FP8 halves the bytes per tensor for both weights and activations. DSA cuts the number of KV entries that the expensive attention path actually touches. Fewer FLOPs is a side effect; reducing memory traffic is what each of them is really doing.

## How the gains compound

Each of these innovations would be a respectable paper on its own. The reason V4 performs as well as it does is that they target different bottlenecks and compose. MoE acts on active compute. MLA acts on KV cache size and the memory bandwidth that pays for it. FP8 acts on bytes per tensor for both weights and activations. DSA acts on attention's scaling in context length.

DeepSeek's V4 release notes state that V4-Pro needs 27% of the single-token inference FLOPs and 10% of the KV cache of V3.2 at 1M context. DSA alone is not 4× better than what came before, but the gain compounds because each earlier choice had already removed a layer of cost. MLA had already compressed the cache, MoE had already gated 97% of the parameters off, and FP8 had already halved the bytes per tensor. DSA on top of all of that then turns the remaining quadratic context cost linear. Each layer of the stack depends on the one below it, and removing any single layer would cause the savings to disappear.

When people ask me whether closed labs can match this, my answer keeps coming back to the same number. V4-Pro activates 49 billion parameters out of 1.6 trillion per token, which is roughly 3% of the model, and that ratio is what determines the inference economics. The model retains the capability of a 1.6T system while running at roughly the inference cost of a well-engineered 49B dense model.

This is the structural reason DeepSeek's aggressive sparsity strategy puts downward pressure on closed-API pricing. Closed labs likely run their own sparsity and MoE optimizations internally, but the available open weights at comparable capability put a hard floor under what any commodity-tier API can charge. The 2026 Stanford HAI AI Index documents the pattern: every time DeepSeek ships a competitive open model, closed-API prices drop within weeks.

## What this means for accelerator design

For anyone designing or evaluating an AI accelerator in 2026, the V4 architecture is a useful stress test. A chip that is optimized for the previous generation's workload of dense GEMMs over big tensors, long contiguous reads from HBM, and regular access patterns will struggle with the V4 hot paths. The performance-critical kernels for a DSA-style stack live somewhere else entirely.

A few of the new pressure points worth designing for:

    - KV cache hierarchy. With MLA, the cache is small enough that more of it can live in higher levels of the memory hierarchy (HBM today, on-package SRAM tomorrow). The accelerators that win on long-context inference will be the ones that can keep more of the working KV state closer to compute.
    - Sparse gather/scatter. Once top-k picks 2,048 tokens out of a million, the actual attention step needs to pull K and V vectors from non-contiguous positions in the cache. Standard contiguous-read DMA engines do not handle this access pattern well, so the design either copies the selected entries to a contiguous staging area first or relies on a gather-capable load unit.
    - Top-k as a first-class operation. A fast top-k over a long sequence dimension is non-trivial. Accelerators with hardware sort or median-selection primitives have a real advantage here; everyone else implements it as a stack of bitonic sorts or approximate top-k.
    - FP8 accumulation. Hopper's design choice — FP8 inputs, FP32 accumulation in the matmul — is now a baseline expectation. Accelerators that can only accumulate in BF16 either have to restructure the math or accept measurable precision loss in long training runs.
    - Compiler and runtime work. The decompress-then-project step in MLA, the fused indexer-plus-top-k in DSA, and the variable per-query sparsity pattern do not slot into the abstractions that PyTorch's compiler stack was built around. Backend compiler and runtime teams end up writing new primitives and arguing with their schedulers about how to place them.

The systems thesis underneath all of this is that skipping work has become a first-class architectural goal rather than a clever optimization layered on top of dense compute. The accelerator stacks that internalize this design principle will keep up with the workload, and the stacks that do not will spend the next two years catching up.