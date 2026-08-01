# GLM W4A16 capability parity ledger

This ledger compares the validated DeepSeek baseline and its retained shared
infrastructure with the GLM runtime. Experimental policies rejected by their
own gates are listed explicitly and are not treated as parity requirements.

| Capability | Classification | GLM status and action |
|---|---|---|
| Immutable zero-rewrite checkpoint reads and exact source validation | already shared | GLM uses the shared Safetensors/checkpoint primitives with its W4A16 manifest. |
| Chat, sampling, streaming, server, and stop handling | already shared | GLM uses the common runtime boundary, including future-entropy sampling. |
| Runtime device admission and capacity-weighted scheduling | already shared | GLM uses the common device planner and an explicit 0.85 VRAM budget. |
| Process RSS, per-device VRAM, checkpoint, CUDA, and graph-phase metrics | already shared | The current worktree shares the generic RSS/VRAM helpers and adds GLM phase attribution. |
| Route tracing and decode-only placement replay | already shared | GLM emits sequential routes; the retained simulator can warm on prefill and measure decode. |
| Layer-major bounded batched prefill | already shared | GLM advances the admitted prompt rows through each layer and batches its linear projections. |
| Generic FlashAttention | already shared | GLM uses the common exact backend for both causal full attention and reconstructed sparse selections. |
| Incremental KV continuation and speculative rewind | already shared | GLM retains exact prefix continuation and rewinds every cache component after lookahead. |
| Bounded reusable CUDA activation workspaces, streams, and events | already shared | GLM now uses the backend-owned per-device MoE buffers instead of three synchronous matmuls. |
| Persistent CUDA weight arena | already shared | The backend capability exists, but GLM does not select it: the equal-budget GLM gate rejected it. It is not restored here. |
| Page-locked resident weight arena | already shared | This is valid for DeepSeek's admitted resident arena. GLM has no equivalent whole-checkpoint resident arena, and its prior gate was rejected. |
| Trace-driven expert prefetch | already shared | Predictor/simulation infrastructure exists. GLM replay increased SSD bytes, so no GLM runtime prefetch policy is enabled. |
| Persistent routed-expert dispatch workers | applicable and missing | Implemented: GLM uses the existing persistent worker pool; no per-layer `std::async` tasks remain. |
| Device-resident gate/up/activation/down expert execution | applicable and missing | Implemented in the common CUDA MoE command for exact offset-packed INT4 triplets. |
| Device-side SwiGLU and routed/shared expert batching | applicable and missing | Implemented with two device kernels per command; decode batches routed experts and the shared expert while retaining deterministic host aggregation. |
| Multi-device enqueue/collect overlap | applicable and missing | Implemented with one persistent task per device and cache leases held through collection. |
| Device-resident activations across the complete graph | applicable and missing | The experimental DeepSeek branch is not in the validated baseline. Limit this integration to the requested MoE boundary; broader graph residency needs its own correctness contract. |
| DeepSeek compact/tiered KV block manager | model-semantic and requiring a GLM implementation | Implemented as GLM-native latent-512 plus RoPE-64 storage, with 128-dimensional index keys only on full-index layers. At the model ceiling the retained cache is 199,716,831,232 bytes (186.0 GiB). |
| Learned sparse index selection | model-semantic and requiring a GLM implementation | Implemented with 32 heads by 128 dimensions, ReLU scores, learned head weights, causal stable top-2048, and interleaved RoPE. |
| Cross-layer index sharing | model-semantic and requiring a GLM implementation | Implemented: full layers are 0, 1, 2, then 6, 10, ..., 74; intervening layers reuse the latest selection. |
| Sparse attention KV reconstruction | model-semantic and requiring a GLM implementation | Implemented: selected compact latents are reconstructed through each layer's `kv_b_proj`; expanded per-head history is never retained. |
| GLM sigmoid/`noaux_tc` routing and routed scale | model-semantic and requiring a GLM implementation | Already implemented and retained; DeepSeek `sqrtsoftplus` behavior is not substituted. |
| GLM shared-expert semantics | model-semantic and requiring a GLM implementation | Implemented in the common device MoE command without changing routed/shared host accumulation order. |
| mHC projection and residual state | impossible because the checkpoint lacks required tensors | GLM declares ordinary residual connections, not mHC; no behavior is fabricated. |
| DSpark verification and rollback | impossible because the checkpoint lacks required tensors | The checkpoint has no DSpark stages or heads. |
| MTP execution | impossible because the checkpoint lacks required tensors | The pinned base checkpoint has zero MTP tensors; MTP remains disabled explicitly. |

The integrated correctness order is binding: common MoE execution, GLM index
selection, compact KV reconstruction, target-layout fixtures, `make check`, and
then one full-checkpoint generation/profile screening run.
