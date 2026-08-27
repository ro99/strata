ValidationResult DeepSeekV4Runtime::Impl::index_positions(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::uint32_t position,
    std::vector<std::uint32_t>& selected,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores, bool device_prepared_source) {
    ValidationResult result;
    selected.clear();
    auto& compressor_state = attention_state[layer].indexer_compressor;
    if (compressor_state.ratio == 0U) {
        result.errors.emplace_back(
            "DeepSeek sparse indexer was not admitted for a long context");
        return result;
    }
    result = compress_state(
        layer, compressor_state,
        layer_prefix(layer) + "attn.indexer.compressor.", input, position,
        attention_state[layer].frequencies, prepared_values, prepared_scores);
    if (!result.ok()) return result;
    return index_select(layer, input, query_rank, position, selected,
                        device_prepared_source);
}

ValidationResult DeepSeekV4Runtime::Impl::index_select(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::uint32_t position,
    std::vector<std::uint32_t>& selected, bool device_prepared_source) {
    ValidationResult result;
    selected.clear();
    const auto record_selection = [&] {
        auto hash = diagnostics.index_selection_trace_hash;
        hash = diagnostic_hash_u32(hash, layer);
        hash = diagnostic_hash_u32(hash, position);
        hash = diagnostic_hash_u32(
            hash, static_cast<std::uint32_t>(selected.size()));
        for (const auto selected_position : selected) {
            hash = diagnostic_hash_u32(hash, selected_position);
        }
        diagnostics.index_selection_trace_hash = hash;
        ++diagnostics.index_selection_count;
    };
    const auto& state = attention_state[layer].indexer_compressor;
    if (state.ratio == 0U) {
        result.errors.emplace_back(
            "DeepSeek sparse indexer was not admitted for a long context");
        return result;
    }
    const auto prefix = layer_prefix(layer) + "attn.indexer.";

    const auto compressed_count = (position + 1U) / state.ratio;
    if (compressed_count <= kIndexTopK) {
        selected.resize(compressed_count);
        std::iota(selected.begin(), selected.end(), 0U);
        record_selection();
        return result;
    }
    ++graph_stats.attention_index_queries;
    graph_stats.attention_index_candidates += compressed_count;
    graph_stats.attention_index_selected += kIndexTopK;

    const auto slot = layer_device(layer);
    std::vector<float> queries(
        static_cast<std::size_t>(kIndexHeads) * kIndexHeadDim);
    std::vector<float> index_weights(kIndexHeads);
    constexpr float index_scale = kIndexQueryScale;
    const auto& frequencies = attention_state[layer].frequencies;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        device_prepared_source) {
        // Both projections run on the device against the preparation command's
        // own activations, so neither the query rank nor the layer input is
        // sent back across the bus. The rotation angles are still evaluated
        // here rather than in the kernel because host libm and device
        // trigonometry differ in the last ulp, and selection is a hard top-k
        // where that becomes a different candidate set. They depend only on the
        // position and the layer frequencies, so they are knowable before the
        // call -- which is also what will let this run inside the queued chain.
        if (frequencies.size() < kRopeDim / 2U) {
            result.errors.emplace_back(
                "DeepSeek index rope frequencies are too short");
            return result;
        }
        index_rope_cosines.resize(kRopeDim / 2U);
        index_rope_sines.resize(kRopeDim / 2U);
        for (std::size_t pair = 0U; pair < index_rope_cosines.size(); ++pair) {
            const float angle =
                static_cast<float>(position) * frequencies[pair];
            index_rope_cosines[pair] = std::cos(angle);
            index_rope_sines[pair] = std::sin(angle);
        }
        auto cuda_demand = weights->demand();
        Dsv4WeightCache::Lease query_projection;
        Dsv4WeightCache::Lease weight_projection;
        result = weights->acquire(slot, prefix + "wq_b",
                                  kIndexHeads * kIndexHeadDim, kQueryRank,
                                  query_projection);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "weights_proj", kIndexHeads,
                                  kHidden, weight_projection);
        if (!result.ok()) return result;
        CudaDsv4IndexProjectionRequest projection;
        projection.query_projection = &query_projection.weight();
        projection.weight_projection = &weight_projection.weight();
        projection.rope_cosines = index_rope_cosines;
        projection.rope_sines = index_rope_sines;
        projection.heads = kIndexHeads;
        projection.head_dim = kIndexHeadDim;
        projection.rope_dim = kRopeDim;
        projection.weight_scale = index_scale;
        result = cuda.dsv4_index_projections(devices[slot], projection, queries,
                                             index_weights);
        if (!result.ok()) return result;
    } else {
        result = linear(slot, prefix + "wq_b", kIndexHeads * kIndexHeadDim,
                        kQueryRank, query_rank, queries);
        if (!result.ok()) return result;
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            // The batched prefill page projects many rows before any of them
            // reaches selection, so it leaves no per-row preparation state on
            // the device. Its query still crosses the same rotation,
            // rounding and half-up quantization, one uploaded row at a time.
            if (frequencies.size() < kRopeDim / 2U) {
                result.errors.emplace_back(
                    "DeepSeek index rope frequencies are too short");
                return result;
            }
            index_rope_cosines.resize(kRopeDim / 2U);
            index_rope_sines.resize(kRopeDim / 2U);
            for (std::size_t pair = 0U; pair < index_rope_cosines.size();
                 ++pair) {
                const float angle =
                    static_cast<float>(position) * frequencies[pair];
                index_rope_cosines[pair] = std::cos(angle);
                index_rope_sines[pair] = std::sin(angle);
            }
            result = cuda.dsv4_index_query_rope_quantize(
                devices[slot], queries, index_rope_cosines, index_rope_sines,
                kIndexHeads, kIndexHeadDim, kRopeDim, true);
            if (!result.ok()) return result;
        } else {
            for (std::uint32_t head = 0U; head < kIndexHeads; ++head) {
                auto query = std::span<float>(queries).subspan(
                    static_cast<std::size_t>(head) * kIndexHeadDim,
                    kIndexHeadDim);
                apply_rope(query.last(kRopeDim), position, frequencies);
                round_bf16(query.last(kRopeDim));
                if (!config.enable_gpu_lightning_indexer) {
                    result = dsv4_hadamard_rotate_f32(query);
                    if (!result.ok()) return result;
                    result = dsv4_fp4_e2m1_simulate_f32(query, 32U);
                    if (!result.ok()) return result;
                }
            }
        }
        result = linear(slot, prefix + "weights_proj", kIndexHeads, kHidden,
                        input, index_weights);
        if (!result.ok()) return result;
        for (auto& weight : index_weights) {
            weight = round_bf16(weight * index_scale);
        }
    }

    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        // Physical KV keeps the learned index as E4M3 rows with one f32 scale
        // each, which the FP4 Lightning Indexer above cannot read. The scalar
        // path below is exact but scores every candidate on the host: at the
        // declared 1,048,576-token context that is 262,144 candidates over 64
        // heads of 128 dimensions per layer, 1.85e11 FLOP per token across the
        // 21 ratio-4 layers, which no CPU budget reaches. Device selection is
        // therefore the only viable path here, not an optimization of one.
        auto cuda_demand = weights->demand();
        auto& blocks = physical_index_blocks;
        result = kv_cache->block_table_into(
            active_sequence, Dsv4KvBlockKind::LearnedIndex, layer, blocks);
        if (!result.ok()) return result;
        std::vector<Dsv4KvDeviceLease> leases;
        std::vector<CudaDsv4PhysicalIndexPage> pages;
        try {
            leases.reserve(blocks.size());
            pages.reserve(blocks.size());
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate physical Lightning Indexer page metadata");
            return result;
        }
        std::uint32_t remaining = compressed_count;
        for (const auto& block : blocks) {
            if (remaining == 0U) break;
            const auto logical_row = block.logical_begin /
                                     block.compression_ratio;
            auto lease = kv_cache->acquire_device(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                logical_row, slot);
            if (!lease.ok()) {
                append_errors(
                    result, std::move(lease.errors),
                    "DeepSeek physical index page lease layer " +
                        std::to_string(layer) + " logical row " +
                        std::to_string(logical_row));
                return result;
            }
            const auto rows = std::min(remaining, block.used_rows);
            leases.push_back(std::move(lease.value));
            // A physical device lease holds the block-major payload alone;
            // acquire_device strips the header on upload.
            pages.push_back(CudaDsv4PhysicalIndexPage{
                leases.back().buffer(), 0U, block.capacity_rows, rows});
            remaining -= rows;
        }
        if (remaining != 0U) {
            result.errors.emplace_back(
                "physical Lightning Indexer device history is incomplete");
            return result;
        }
        selected.resize(kIndexTopK);
        CudaDsv4PhysicalIndexRequest request;
        request.queries = queries;
        request.weights = index_weights;
        request.pages = pages;
        request.heads = kIndexHeads;
        request.head_dim = kIndexHeadDim;
        request.top_k = kIndexTopK;
        result = cuda.dsv4_physical_lightning_index(
            devices[slot], request, selected);
        if (!result.ok()) {
            selected.clear();
            return result;
        }
        ++graph_stats.attention_index_cuda_dispatches;
        record_selection();
        return result;
    }

    if (config.enable_gpu_lightning_indexer) {
        auto cuda_demand = weights->demand();
        std::vector<CudaLightningIndexSegment> segments;
        std::vector<Dsv4KvDeviceLease> leases;
        const bool device_resident =
            !config.device_kv_cache_bytes.empty() &&
            config.device_kv_cache_bytes[slot] != 0U;
        if (device_resident) {
            auto& blocks = physical_index_blocks;
            result = kv_cache->block_table_into(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                blocks);
            if (!result.ok()) return result;
            try {
                segments.reserve(blocks.size());
                leases.reserve(blocks.size());
            } catch (const std::bad_alloc&) {
                result.errors.emplace_back(
                    "cannot allocate Lightning Indexer block metadata");
                return result;
            }
            std::uint32_t remaining = compressed_count;
            for (const auto& block : blocks) {
                if (remaining == 0U) break;
                const auto logical_row = block.logical_begin /
                                         block.compression_ratio;
                auto lease = kv_cache->acquire_device(
                    active_sequence, Dsv4KvBlockKind::LearnedIndex,
                    layer, logical_row, slot);
                if (!lease.ok()) {
                    append_errors(result, std::move(lease.errors));
                    return result;
                }
                const auto rows = std::min(remaining, block.used_rows);
                leases.push_back(std::move(lease.value));
                segments.push_back(CudaLightningIndexSegment{
                    leases.back().buffer(), {}, kDsv4KvBlockHeaderBytes, rows});
                remaining -= rows;
            }
            if (remaining != 0U) {
                result.errors.emplace_back(
                    "Lightning Indexer device history is incomplete");
                return result;
            }
        } else {
            auto compact = kv_cache->learned_index_segments(
                active_sequence, layer, compressed_count);
            if (!compact.ok()) {
                append_errors(result, std::move(compact.errors));
                return result;
            }
            segments = std::move(compact.value);
        }
        selected.resize(kIndexTopK);
        CudaLightningIndexRequest request;
        request.queries = queries;
        request.weights = index_weights;
        request.segments = segments;
        request.heads = kIndexHeads;
        request.head_dim = kIndexHeadDim;
        request.top_k = kIndexTopK;
        result = cuda.lightning_index(devices[slot], request, selected);
        if (!result.ok()) {
            selected.clear();
            return result;
        }
        ++graph_stats.attention_index_cuda_dispatches;
        record_selection();
        return result;
    }

    std::vector<float> scores(compressed_count);
    const auto score_key = [&](std::size_t row, std::span<const float> key) {
        if (key.size() != kIndexHeadDim) {
            scores[row] = -std::numeric_limits<float>::infinity();
            return;
        }
        auto destination = std::span<float>(scores).subspan(row, 1U);
        const auto scored = dsv4_index_scores_f32(
            destination, queries, key, index_weights, kIndexHeads,
            kIndexHeadDim);
        if (!scored.ok()) scores[row] = -std::numeric_limits<float>::infinity();
    };
    const auto score_row = [&](std::size_t row) {
        score_key(row, state.compressed.row(row));
    };
    if (kv_cache != nullptr) {
        for (std::uint32_t row = 0U; row < compressed_count; ++row) {
            auto key = kv_row(layer, Dsv4KvBlockKind::LearnedIndex, row);
            if (!key.ok()) {
                append_errors(result, std::move(key.errors));
                return result;
            }
            score_key(row, key.value);
        }
    } else if (attention_workers != nullptr && attention_workers->size() > 1U) {
        const auto workers = std::min<std::size_t>(
            attention_workers->size(), compressed_count);
        result = attention_workers->parallel_for(
            workers, [&](std::size_t worker) {
                const auto begin = compressed_count * worker / workers;
                const auto end = compressed_count * (worker + 1U) / workers;
                for (std::size_t row = begin; row < end; ++row) {
                    score_row(row);
                }
            });
        if (!result.ok()) return result;
    } else {
        for (std::size_t row = 0U; row < compressed_count; ++row) {
            score_row(row);
        }
    }

    auto topk = dsv4_index_topk_f32(scores, kIndexTopK);
    if (!topk.ok()) {
        append_errors(result, std::move(topk.errors));
        return result;
    }
    selected = std::move(topk.positions);
    ++graph_stats.attention_index_scalar_dispatches;
    record_selection();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position,
    std::span<float> output) {
    ValidationResult result;
    if (input.size() != kHidden ||
        (output.size() != kHidden &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           output.empty()))) {
        result.errors.emplace_back("DeepSeek attention spans have incompatible sizes");
        return result;
    }
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto cuda_demand = weights->demand();
        // The learned-index *row* can now be reserved and committed in stream
        // order with the sliding and compressed rows (see the index_append
        // handling below and in complete_physical_attention_prepare). What
        // still excludes the sparse indexer from the queued path is
        // *selection*, not the append: index_select() projects index queries
        // through wq_b and weights_proj, and linear() dispatches those to
        // CUDA. Selection can only run once query_rank exists host-side, which
        // in the queued path is inside the CUDA host callback -- where calling
        // a CUDA API is not permitted.
        //
        // Until index query projection, scoring and top-k are moved onto the
        // device inside the queued chain, this path must stay closed: enabling
        // it would reach physical_paged_attention with sparse == true and an
        // empty indexed_positions, which silently attends zero compressed
        // rows. Fail closed instead.
        const bool deferred_page_update =
            !config.enable_layer_hash_trace && kv_cache != nullptr &&
            attention_state[layer].indexer_compressor.ratio == 0U;
        Dsv4WeightCache::Lease query_a;
        Dsv4WeightCache::Lease query_b;
        Dsv4WeightCache::Lease key_value_weight;
        Dsv4WeightCache::Lease compressor_value_weight;
        Dsv4WeightCache::Lease compressor_gate_weight;
        Dsv4WeightCache::Lease index_compressor_value_weight;
        Dsv4WeightCache::Lease index_compressor_gate_weight;
        result = weights->acquire(
            slot, prefix + "wq_a", kQueryRank, kHidden, query_a);
        if (!result.ok()) return result;
        result = weights->acquire(
            slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank,
            query_b);
        if (!result.ok()) return result;
        result = weights->acquire(
            slot, prefix + "wkv", kHeadDim, kHidden, key_value_weight);
        if (!result.ok()) return result;
        const auto compressor_dimensions =
            static_cast<std::size_t>(
                attention_state[layer].compressor.coefficient) *
            attention_state[layer].compressor.head_dim;
        std::vector<float> compressor_values;
        std::vector<float> compressor_scores;
        if (attention_state[layer].compressor.ratio != 0U) {
            result = weights->acquire(
                slot, prefix + "compressor.wkv", compressor_dimensions,
                kHidden, compressor_value_weight);
            if (!result.ok()) return result;
            result = weights->acquire(
                slot, prefix + "compressor.wgate", compressor_dimensions,
                kHidden, compressor_gate_weight);
            if (!result.ok()) return result;
            compressor_values.resize(compressor_dimensions);
            compressor_scores.resize(compressor_dimensions);
        }
        const auto index_compressor_dimensions =
            static_cast<std::size_t>(
                attention_state[layer].indexer_compressor.coefficient) *
            attention_state[layer].indexer_compressor.head_dim;
        std::vector<float> index_compressor_values;
        std::vector<float> index_compressor_scores;
        if (attention_state[layer].indexer_compressor.ratio != 0U) {
            result = weights->acquire(
                slot, prefix + "indexer.compressor.wkv",
                index_compressor_dimensions, kHidden,
                index_compressor_value_weight);
            if (!result.ok()) return result;
            result = weights->acquire(
                slot, prefix + "indexer.compressor.wgate",
                index_compressor_dimensions, kHidden,
                index_compressor_gate_weight);
            if (!result.ok()) return result;
            index_compressor_values.resize(index_compressor_dimensions);
            index_compressor_scores.resize(index_compressor_dimensions);
        }
        auto query_norm = host_tensor(prefix + "q_norm.weight", kQueryRank);
        if (!query_norm.ok()) {
            append_errors(result, std::move(query_norm.errors));
            return result;
        }
        auto key_value_norm = host_tensor(
            prefix + "kv_norm.weight", kHeadDim);
        if (!key_value_norm.ok()) {
            append_errors(result, std::move(key_value_norm.errors));
            return result;
        }
        std::array<float, kRopeDim / 2U> cosines{};
        std::array<float, kRopeDim / 2U> sines{};
        for (std::size_t index = 0U; index < cosines.size(); ++index) {
            const float angle = static_cast<float>(position) *
                                attention_state[layer].frequencies[index];
            cosines[index] = std::cos(angle);
            sines[index] = std::sin(angle);
        }
        std::vector<float> query_rank(kQueryRank);
        std::vector<float> kv(kHeadDim);
        auto& deferred_context = physical_attention_context(
            host_moe_chain_row.value_or(layer));
        std::vector<CudaDsv4AttentionPageWrite> page_writes;
        if (deferred_page_update) {
            deferred_context.owner = this;
            deferred_context.layer = layer;
            deferred_context.position = position;
            deferred_context.result = {};
            deferred_context.invoked = false;
            deferred_context.key_value.resize(kHeadDim);
            deferred_context.sliding_append.reset();
            deferred_context.compressed_append.reset();
            deferred_context.index_append.reset();
            auto sliding = kv_cache->reserve_physical_append(
                active_sequence, Dsv4KvBlockKind::Sliding, layer, 1U,
                position, slot);
            if (!sliding.ok()) {
                append_errors(result, std::move(sliding.errors));
                return result;
            }
            deferred_context.sliding_append.emplace(
                std::move(sliding.value));
            const auto ratio = attention_state[layer].compressor.ratio;
            if (ratio != 0U && (position + 1U) % ratio == 0U) {
                auto compressed = kv_cache->reserve_physical_append(
                    active_sequence,
                    attention_state[layer].compressor.kind, layer, ratio,
                    position / ratio, slot);
                if (!compressed.ok()) {
                    append_errors(result, std::move(compressed.errors));
                    return result;
                }
                deferred_context.compressed_append.emplace(
                    std::move(compressed.value));
            }
            const auto add_write = [&](const Dsv4KvPhysicalAppend& append) {
                page_writes.push_back(CudaDsv4AttentionPageWrite{
                    append.buffer(), append.data_offset(),
                    append.scale_offset(), append.data_bytes(),
                    append.scale_bytes()});
            };
            const auto index_ratio =
                attention_state[layer].indexer_compressor.ratio;
            if (index_ratio != 0U && (position + 1U) % index_ratio == 0U) {
                auto index = kv_cache->reserve_physical_append(
                    active_sequence,
                    attention_state[layer].indexer_compressor.kind, layer,
                    index_ratio, position / index_ratio, slot);
                if (!index.ok()) {
                    append_errors(result, std::move(index.errors));
                    return result;
                }
                deferred_context.index_append.emplace(std::move(index.value));
            }
            add_write(*deferred_context.sliding_append);
            if (deferred_context.compressed_append.has_value()) {
                add_write(*deferred_context.compressed_append);
            }
            if (deferred_context.index_append.has_value()) {
                add_write(*deferred_context.index_append);
            }
        }
        CudaDsv4AttentionPrepareRequest request;
        request.query_a = &query_a.weight();
        request.query_b = &query_b.weight();
        request.key_value = &key_value_weight.weight();
        if (pending_mhc_attention_transition) {
            request.mhc_transition = &device_mhc_weights[layer][0U];
        }
        if (!compressor_values.empty()) {
            request.compressor_value = &compressor_value_weight.weight();
            request.compressor_gate = &compressor_gate_weight.weight();
        }
        if (!index_compressor_values.empty()) {
            request.index_compressor_value =
                &index_compressor_value_weight.weight();
            request.index_compressor_gate =
                &index_compressor_gate_weight.weight();
        }
        request.query_norm = *query_norm.value;
        request.key_value_norm = *key_value_norm.value;
        request.rope_cosines = cosines;
        request.rope_sines = sines;
        request.mhc_device = devices[mhc_slot];
        if (devices[slot] != devices[mhc_slot] &&
            !pending_mhc_attention_transition &&
            (layer != 0U || config.enable_layer_hash_trace)) {
            request.cross_device_input = input;
        }
        if (deferred_page_update) {
            request.host_callback = physical_attention_prepare_callback;
            request.host_callback_context = &deferred_context;
            request.page_writes = page_writes;
        }
        request.maximum_workspace_bytes = 1ULL << 20U;
        auto subphase_started = std::chrono::steady_clock::now();
        const auto prepared_compressor_matmuls =
            static_cast<std::uint64_t>(!compressor_values.empty()) * 2U +
            static_cast<std::uint64_t>(!index_compressor_values.empty()) * 2U;
        graph_stats.attention_projection_matmul_calls +=
            3U + prepared_compressor_matmuls;
        graph_stats.attention_projection_matmul_rows +=
            3U + prepared_compressor_matmuls;
        result = cuda.dsv4_prepare_attention(
            devices[slot], request, query_rank, kv,
            compressor_values, compressor_scores,
            index_compressor_values, index_compressor_scores);
        graph_stats.attention_query_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
        if (!result.ok()) return result;
        pending_mhc_attention_transition = false;
        if (deferred_page_update) {
            pending_attention_weights.push_back(std::move(query_a));
            pending_attention_weights.push_back(std::move(query_b));
            pending_attention_weights.push_back(
                std::move(key_value_weight));
            if (!compressor_values.empty()) {
                pending_attention_weights.push_back(
                    std::move(compressor_value_weight));
                pending_attention_weights.push_back(
                    std::move(compressor_gate_weight));
            }
            if (!index_compressor_values.empty()) {
                pending_attention_weights.push_back(
                    std::move(index_compressor_value_weight));
                pending_attention_weights.push_back(
                    std::move(index_compressor_gate_weight));
            }
            auto sink = host_tensor(prefix + "attn_sink", kHeads);
            if (!sink.ok()) {
                append_errors(result, std::move(sink.errors));
                return result;
            }
            return physical_paged_attention(
                layer, {}, *sink.value, position, {}, output);
        }
        std::span<const float> resident_queries;
        // The preparation command above ran on this layer's device and its
        // activations are still the most recent there, so selection may
        // project from them.
        return attention_prepared(
            layer, input, query_rank, resident_queries, kv, position, output,
            compressor_values, compressor_scores,
            index_compressor_values, index_compressor_scores, true);
    }
    auto subphase_started = std::chrono::steady_clock::now();
    std::vector<float> query_rank(kQueryRank);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wq_a", kQueryRank, kHidden, input, query_rank);
    if (!result.ok()) return result;
    result = norm(query_rank, query_rank, prefix + "q_norm.weight");
    if (!result.ok()) return result;
    std::vector<float> queries(static_cast<std::size_t>(kHeads) * kHeadDim);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank,
                    query_rank, queries);
    if (!result.ok()) return result;
    const auto normalize_query = [&](std::uint32_t head) {
        auto query = std::span<float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        double square_sum = 0.0;
        for (const float value : query) {
            square_sum += static_cast<double>(value) * value;
        }
        const float reciprocal = 1.0F / std::sqrt(
            static_cast<float>(square_sum / kHeadDim) + kRmsEpsilon);
        for (auto& value : query) value = round_bf16(value * reciprocal);
        apply_rope(query.last(kRopeDim), position,
                   attention_state[layer].frequencies);
        round_bf16(query.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                normalize_query(static_cast<std::uint32_t>(head));
            });
        if (!result.ok()) return result;
    } else {
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            normalize_query(head);
        }
    }
    graph_stats.attention_query_nanoseconds += elapsed_nanoseconds(subphase_started);

    subphase_started = std::chrono::steady_clock::now();
    std::vector<float> kv(kHeadDim);
    ++graph_stats.attention_projection_matmul_calls;
    ++graph_stats.attention_projection_matmul_rows;
    result = linear(slot, prefix + "wkv", kHeadDim, kHidden, input, kv);
    if (!result.ok()) return result;
    result = norm(kv, kv, prefix + "kv_norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(kv).last(kRopeDim), position,
               attention_state[layer].frequencies);
    round_bf16(std::span<float>(kv).last(kRopeDim));
    if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        quantize_activation_in_place(
            std::span<float>(kv).first(kHeadDim - kRopeDim), 64U);
    }
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);
    return attention_prepared(layer, input, query_rank, queries, kv, position,
                              output);
}

ValidationResult DeepSeekV4Runtime::Impl::physical_paged_attention(
    std::uint32_t layer, std::span<const float> queries,
    std::span<const float> sinks, std::uint32_t position,
    std::span<const std::uint32_t> indexed_positions,
    std::span<float> diagnostic_branch,
    PhysicalAttentionPageSet* page_set) {
    ValidationResult result;
    if (kv_cache == nullptr ||
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        result.errors.emplace_back(
            "DeepSeek physical paged attention requires its physical cache");
        return result;
    }
    const auto slot = layer_device(layer);
    // Process-lifetime buffers: these tables reach 4,096 blocks at the declared
    // context and are rebuilt once per kind per layer per token, so they must
    // not allocate on the timed path.
    auto& sliding = physical_sliding_blocks;
    auto& compressed = physical_compressed_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::Sliding, layer, sliding);
    if (!result.ok()) return result;
    const auto ratio = attention_state[layer].compressor.ratio;
    compressed.clear();
    if (ratio != 0U) {
        result = kv_cache->block_table_into(
            active_sequence, attention_state[layer].compressor.kind, layer,
            compressed);
        if (!result.ok()) return result;
    }

    std::uint64_t page_set_build_nanoseconds = 0U;
    const bool builds_page_set =
        page_set == nullptr || page_set->pages.empty();
    if (builds_page_set) ++graph_stats.attention_page_set_builds;
    const auto page_set_setup_started = std::chrono::steady_clock::now();
    std::vector<Dsv4KvDeviceLease> local_leases;
    std::vector<CudaDsv4PhysicalPage> local_pages;
    std::unordered_map<std::uint64_t, std::uint32_t> local_page_indices;
    auto& leases = page_set != nullptr ? page_set->leases : local_leases;
    auto& pages = page_set != nullptr ? page_set->pages : local_pages;
    auto& page_indices = page_set != nullptr
        ? page_set->page_indices : local_page_indices;
    if (page_set != nullptr && page_set->pages.empty()) {
        const auto reserve = sliding.size() + compressed.size();
        page_set->leases.reserve(reserve);
        page_set->pages.reserve(reserve);
        page_set->page_indices.reserve(reserve);
    }
    if (builds_page_set) {
        page_set_build_nanoseconds +=
            elapsed_nanoseconds(page_set_setup_started);
    }
    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "DeepSeek physical attention candidate page is unavailable");
            return;
        }
        const auto found = table.begin() + static_cast<std::ptrdiff_t>(located);
        const auto begin = found->logical_begin / found->compression_ratio;
        auto page = page_indices.find(found->id);
        if (page == page_indices.end()) {
            const auto page_build_started = std::chrono::steady_clock::now();
            auto lease = kv_cache->acquire_device(
                active_sequence, kind, layer, logical_row, slot);
            if (!lease.ok()) {
                append_errors(
                    result, std::move(lease.errors),
                    "DeepSeek physical attention page lease layer " +
                        std::to_string(layer) + " kind " +
                        std::to_string(static_cast<unsigned>(kind)) +
                        " logical row " + std::to_string(logical_row));
                return;
            }
            const auto index = static_cast<std::uint32_t>(pages.size());
            leases.push_back(std::move(lease.value));
            pages.push_back({leases.back().buffer(), found->capacity_rows});
            page = page_indices.emplace(found->id, index).first;
            ++graph_stats.attention_page_set_pages;
            page_set_build_nanoseconds +=
                elapsed_nanoseconds(page_build_started);
        }
        candidate.page = page->second;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
    };

    const auto compressed_count = ratio == 0U
        ? 0U : (position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        attention_state[layer].indexer_compressor.ratio == 4U;
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    std::vector<CudaDsv4AttentionCandidate> candidates(
        static_cast<std::size_t>(compressed_width) + sliding_width);
    const auto attended_compressed = sparse
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    if (attended_compressed > compressed_width) {
        result.errors.emplace_back(
            "DeepSeek physical attention compressed candidates exceed their fixed region");
        return result;
    }
    const auto candidate_resolution_started = std::chrono::steady_clock::now();
    graph_stats.attention_candidate_resolutions +=
        static_cast<std::uint64_t>(attended_compressed);
    for (std::uint32_t item = 0U; item < attended_compressed; ++item) {
        const auto logical_row = sparse ? indexed_positions[item] : item;
        locate(attention_state[layer].compressor.kind, compressed,
               logical_row, candidates[item]);
        if (!result.ok()) return result;
    }
    const auto window_count = std::min(position + 1U, kWindow);
    graph_stats.attention_candidate_resolutions += window_count;
    for (std::uint32_t item = 0U; item < window_count; ++item) {
        const auto logical_row = position + 1U - window_count + item;
        locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
               candidates[static_cast<std::size_t>(compressed_width) + item]);
        if (!result.ok()) return result;
    }
    const auto candidate_resolution_nanoseconds =
        elapsed_nanoseconds(candidate_resolution_started);
    graph_stats.attention_page_set_build_nanoseconds +=
        page_set_build_nanoseconds;
    graph_stats.attention_candidate_resolution_nanoseconds +=
        candidate_resolution_nanoseconds > page_set_build_nanoseconds
            ? candidate_resolution_nanoseconds - page_set_build_nanoseconds
            : 0U;

    const auto prefix = layer_prefix(layer) + "attn.";
    const auto weight_started = std::chrono::steady_clock::now();
    Dsv4WeightCache::Lease output_a;
    Dsv4WeightCache::Lease output_b;
    Dsv4WeightCache::Lease router;
    result = weights->acquire(
        slot, prefix + "wo_a", kOutputGroups * kOutputRank,
        kHeads * kHeadDim / kOutputGroups, output_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wo_b", kHidden,
        kOutputGroups * kOutputRank, output_b);
    if (!result.ok()) return result;
    graph_stats.attention_page_weight_acquire_nanoseconds +=
        elapsed_nanoseconds(weight_started);
    std::array<float, kRopeDim / 2U> cosines{};
    std::array<float, kRopeDim / 2U> sines{};
    for (std::size_t index = 0U; index < cosines.size(); ++index) {
        const float angle = static_cast<float>(position) *
                            attention_state[layer].frequencies[index] * -1.0F;
        cosines[index] = std::cos(angle);
        sines[index] = std::sin(angle);
    }
    CudaDsv4PagedAttentionMhcRequest request;
    request.attention.queries = queries;
    request.attention.head_sinks = sinks;
    request.attention.pages = pages;
    request.attention.candidates = candidates;
    request.attention.scale = kAttentionScale;
    request.attention.maximum_workspace_bytes = 4ULL << 20U;
    request.inverse_rope_cosines = cosines;
    request.inverse_rope_sines = sines;
    request.output_a = &output_a.weight();
    request.output_b = &output_b.weight();
    request.mhc_device = devices[mhc_slot];
    const bool combine_mhc_transition = diagnostic_branch.empty();
    if (combine_mhc_transition) {
        result = weights->acquire(
            mhc_slot, layer_prefix(layer) + "ffn.gate", kExperts, kHidden,
            router);
        if (!result.ok()) return result;
        request.mhc_transition = &device_mhc_weights[layer][1U];
        request.router = &router.weight();
        request.defer_host_moe_input = true;
    }
    result = cuda.dsv4_paged_attention_to_mhc(
        devices[slot], request, diagnostic_branch);
    if (result.ok()) {
        completed_attention_mhc_transition = combine_mhc_transition;
        completed_router_projection = combine_mhc_transition;
        deferred_attention_moe_input = combine_mhc_transition;
        if (combine_mhc_transition) {
            pending_attention_leases.insert(
                pending_attention_leases.end(),
                std::make_move_iterator(leases.begin()),
                std::make_move_iterator(leases.end()));
            pending_attention_weights.push_back(std::move(output_a));
            pending_attention_weights.push_back(std::move(output_b));
            pending_attention_weights.push_back(std::move(router));
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::physical_paged_attention_page(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::span<const float> sinks, std::uint32_t position_base,
    std::span<const std::uint32_t> row_slots,
    std::span<float> diagnostic_branches) {
    ValidationResult result;
    const auto rows64 = input.size() / kHidden;
    const auto query_stride = static_cast<std::size_t>(kHeads) * kHeadDim;
    if (kv_cache == nullptr ||
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice ||
        rows64 < 2U || rows64 > std::numeric_limits<std::uint32_t>::max() ||
        input.size() != rows64 * kHidden ||
        query_rank.size() != rows64 * kQueryRank ||
        (!queries.empty() && queries.size() != rows64 * query_stride) ||
        sinks.size() != kHeads || row_slots.size() != rows64 ||
        diagnostic_branches.size() != rows64 * kHidden) {
        result.errors.emplace_back(
            "DeepSeek physical attention page spans are incompatible");
        return result;
    }
    const auto rows = static_cast<std::uint32_t>(rows64);
    const auto slot = layer_device(layer);
    auto& layer_state = attention_state[layer];
    const auto ratio = layer_state.compressor.ratio;
    const auto last_position = position_base + rows - 1U;

    auto& sliding = physical_sliding_blocks;
    auto& compressed = physical_compressed_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::Sliding, layer, sliding);
    if (!result.ok()) return result;
    compressed.clear();
    if (ratio != 0U) {
        result = kv_cache->block_table_into(
            active_sequence, layer_state.compressor.kind, layer, compressed);
        if (!result.ok()) return result;
    }

    ++graph_stats.attention_page_set_builds;
    const auto page_build_started = std::chrono::steady_clock::now();
    std::vector<Dsv4KvDeviceLease> leases;
    std::vector<CudaDsv4PhysicalPage> pages;
    std::unordered_map<std::uint64_t, std::uint32_t> page_indices;
    const auto reserve = compressed.size() + sliding.size();
    leases.reserve(reserve);
    pages.reserve(reserve);
    page_indices.reserve(reserve);
    const auto lease_table = [&](Dsv4KvBlockKind kind,
                                 const std::vector<Dsv4KvBlockInfo>& table) {
        for (const auto& block : table) {
            if (block.compression_ratio == 0U) {
                result.errors.emplace_back(
                    "DeepSeek physical attention page has a zero-ratio block");
                return;
            }
            // The lease names a block, not a row: every row inside the block
            // resolves to the same device buffer. Blocks are retired whole,
            // 256 rows at a time, while the sliding window retires rows one at
            // a time, so a block that is still live can already have its first
            // row outside the retained window -- which is every prefill page
            // based at or beyond kWindow. Name the block's most recent row
            // instead; it is retained for exactly as long as the block is.
            const auto first_row =
                block.logical_begin / block.compression_ratio;
            const auto logical_row = static_cast<std::uint32_t>(
                block.used_rows == 0U
                    ? first_row
                    : first_row + block.used_rows - 1U);
            auto lease = kv_cache->acquire_device(
                active_sequence, kind, layer, logical_row, slot);
            if (!lease.ok()) {
                append_errors(result, std::move(lease.errors));
                return;
            }
            const auto page = static_cast<std::uint32_t>(pages.size());
            leases.push_back(std::move(lease.value));
            pages.push_back({leases.back().buffer(), block.capacity_rows});
            page_indices.emplace(block.id, page);
            ++graph_stats.attention_page_set_pages;
        }
    };
    if (ratio != 0U) lease_table(layer_state.compressor.kind, compressed);
    if (!result.ok()) return result;
    lease_table(Dsv4KvBlockKind::Sliding, sliding);
    if (!result.ok()) return result;
    graph_stats.attention_page_set_build_nanoseconds +=
        elapsed_nanoseconds(page_build_started);

    const auto compressed_count = ratio == 0U
        ? 0U : (last_position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        layer_state.indexer_compressor.ratio == 4U;
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    const auto candidate_width = compressed_width + sliding_width;
    std::vector<CudaDsv4AttentionCandidate> candidates(
        static_cast<std::size_t>(rows) * candidate_width);

    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "DeepSeek physical attention page candidate is unavailable");
            return;
        }
        const auto& block = table[located];
        const auto found = page_indices.find(block.id);
        if (found == page_indices.end()) {
            result.errors.emplace_back(
                "DeepSeek physical attention page lease is unavailable");
            return;
        }
        const auto begin = block.logical_begin / block.compression_ratio;
        candidate.page = found->second;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
        static_cast<void>(kind);
    };

    const auto candidate_started = std::chrono::steady_clock::now();
    std::uint64_t selection_nanoseconds = 0U;
    std::vector<std::uint32_t> selected;
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto position = position_base + row;
        const auto row_compressed_count = ratio == 0U
            ? 0U : (position + 1U) / ratio;
        if (sparse) {
            const auto selection_started = std::chrono::steady_clock::now();
            result = index_select(
                layer,
                input.subspan(static_cast<std::size_t>(row) * kHidden,
                              kHidden),
                query_rank.subspan(
                    static_cast<std::size_t>(row) * kQueryRank, kQueryRank),
                position, selected, false);
            selection_nanoseconds += elapsed_nanoseconds(selection_started);
            if (!result.ok()) return result;
        } else {
            selected.resize(row_compressed_count);
            std::iota(selected.begin(), selected.end(), 0U);
        }
        if (selected.size() > compressed_width) {
            result.errors.emplace_back(
                "DeepSeek physical attention page selection exceeds its region");
            return result;
        }
        auto row_candidates = std::span<CudaDsv4AttentionCandidate>(candidates)
            .subspan(static_cast<std::size_t>(row) * candidate_width,
                     candidate_width);
        graph_stats.attention_candidate_resolutions += selected.size();
        for (std::size_t item = 0U; item < selected.size(); ++item) {
            locate(layer_state.compressor.kind, compressed, selected[item],
                   row_candidates[item]);
            if (!result.ok()) return result;
        }
        const auto window_count = std::min(position + 1U, kWindow);
        graph_stats.attention_candidate_resolutions += window_count;
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto logical_row = position + 1U - window_count + item;
            locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
                   row_candidates[compressed_width + item]);
            if (!result.ok()) return result;
        }
    }
    graph_stats.attention_page_index_selection_nanoseconds +=
        selection_nanoseconds;
    graph_stats.attention_candidate_resolution_nanoseconds +=
        elapsed_nanoseconds(candidate_started) - selection_nanoseconds;

    const auto prefix = layer_prefix(layer) + "attn.";
    const auto weight_started = std::chrono::steady_clock::now();
    Dsv4WeightCache::Lease output_a;
    Dsv4WeightCache::Lease output_b;
    Dsv4WeightCache::Lease query_b;
    if (queries.empty()) {
        result = weights->acquire(
            slot, prefix + "wq_b", query_stride, kQueryRank, query_b);
        if (!result.ok()) return result;
    }
    result = weights->acquire(
        slot, prefix + "wo_a", kOutputGroups * kOutputRank,
        kHeads * kHeadDim / kOutputGroups, output_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wo_b", kHidden,
        kOutputGroups * kOutputRank, output_b);
    if (!result.ok()) return result;
    graph_stats.attention_page_weight_acquire_nanoseconds +=
        elapsed_nanoseconds(weight_started);

    std::vector<float> cosines(static_cast<std::size_t>(rows) * kRopeDim / 2U);
    std::vector<float> sines(cosines.size());
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto position = position_base + row;
        for (std::size_t index = 0U; index < kRopeDim / 2U; ++index) {
            const float angle = static_cast<float>(position) *
                                layer_state.frequencies[index] * -1.0F;
            cosines[static_cast<std::size_t>(row) * kRopeDim / 2U + index] =
                std::cos(angle);
            sines[static_cast<std::size_t>(row) * kRopeDim / 2U + index] =
                std::sin(angle);
        }
    }

    // A scheduling page may be wider than the bounded CUDA workspace. Derive
    // its row admission from the backend's exact allocation layout, then slice
    // only the query dimension. Pages and weights remain shared by every
    // slice, matching the reference stack's budgeted prefill chunks.
    constexpr std::uint64_t maximum_workspace_bytes = 384ULL << 20U;
    auto admitted = cuda.dsv4_paged_attention_to_mhc_page_maximum_rows(
        pages, rows, candidate_width, maximum_workspace_bytes,
        queries.empty());
    if (!admitted.ok()) return {std::move(admitted.errors)};
    const auto maximum_rows = admitted.value;
    if (const char* trace = std::getenv("STRATA_TRACE_ATTENTION_LAYOUT");
        trace != nullptr && *trace == '1') {
        static std::atomic<int> emitted{0};
        if (emitted.fetch_add(1) < 2) {
            auto at = [&](std::uint32_t probe) -> std::uint64_t {
                auto bytes = cuda
                    .dsv4_paged_attention_to_mhc_page_workspace_bytes(
                        pages, probe, candidate_width, queries.empty());
                return bytes.ok() ? bytes.value : 0U;
            };
            std::fprintf(stderr,
                "page admission: requested=%u admitted=%u chunks=%u "
                "ws(admitted)=%.2f MB ws(requested)=%.2f MB cap=%.2f MB\n",
                rows, maximum_rows,
                (rows + maximum_rows - 1U) / maximum_rows,
                at(maximum_rows) / 1048576.0, at(rows) / 1048576.0,
                maximum_workspace_bytes / 1048576.0);
        }
    }

    const auto rope_stride = static_cast<std::size_t>(kRopeDim) / 2U;
    std::uint32_t begin = 0U;
    while (begin < rows) {
        const auto remaining = rows - begin;
        auto chunk_rows = std::min(maximum_rows, remaining);
        // The page command uses the single-row path when rows == 1, which has
        // a different mHC state contract. Rebalance the preceding chunk so
        // every page slice retains at least two rows.
        if (remaining - chunk_rows == 1U) --chunk_rows;
        if (chunk_rows < 2U) {
            result.errors.emplace_back(
                "DeepSeek attention page workspace split produced a singleton slice");
            return result;
        }

        CudaDsv4PagedAttentionMhcRequest request;
        request.attention.queries = queries.subspan(
            queries.empty() ? 0U
                            : static_cast<std::size_t>(begin) * query_stride,
            queries.empty() ? 0U
                            : static_cast<std::size_t>(chunk_rows) *
                                  query_stride);
        request.attention.head_sinks = sinks;
        request.attention.pages = pages;
        request.attention.candidates = std::span<const CudaDsv4AttentionCandidate>(
            candidates).subspan(
                static_cast<std::size_t>(begin) * candidate_width,
                static_cast<std::size_t>(chunk_rows) * candidate_width);
        request.attention.rows = chunk_rows;
        request.attention.candidate_width = candidate_width;
        request.attention.scale = kAttentionScale;
        request.attention.maximum_workspace_bytes = maximum_workspace_bytes;
        request.mhc_slots = row_slots.subspan(begin, chunk_rows);
        if (queries.empty()) {
            request.page_query_projection = &query_b.weight();
            request.page_query_rank = query_rank.subspan(
                static_cast<std::size_t>(begin) * kQueryRank,
                static_cast<std::size_t>(chunk_rows) * kQueryRank);
        }
        request.inverse_rope_cosines = std::span<const float>(cosines).subspan(
            static_cast<std::size_t>(begin) * rope_stride,
            static_cast<std::size_t>(chunk_rows) * rope_stride);
        request.inverse_rope_sines = std::span<const float>(sines).subspan(
            static_cast<std::size_t>(begin) * rope_stride,
            static_cast<std::size_t>(chunk_rows) * rope_stride);
        request.output_a = &output_a.weight();
        request.output_b = &output_b.weight();
        request.mhc_device = devices[mhc_slot];

        result = cuda.dsv4_paged_attention_to_mhc(
            devices[slot], request, diagnostic_branches.subspan(
                static_cast<std::size_t>(begin) * kHidden,
                static_cast<std::size_t>(chunk_rows) * kHidden));
        if (!result.ok()) return result;
        begin += chunk_rows;
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_append_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> kv, std::uint32_t position,
    std::span<const float> compressor_values,
    std::span<const float> compressor_scores,
    std::span<const float> index_compressor_values,
    std::span<const float> index_compressor_scores,
    bool append_index_compressor,
    std::uint64_t sliding_retention_floor) {
    ValidationResult result;
    if (input.size() != kHidden || kv.size() != kHeadDim) {
        result.errors.emplace_back("DeepSeek prepared attention append spans have incompatible sizes");
        return result;
    }
    const auto prefix = layer_prefix(layer) + "attn.";
    auto subphase_started = std::chrono::steady_clock::now();
    auto& layer_state = attention_state[layer];
    if (kv_cache != nullptr) {
        result = kv_cache->append(active_sequence,
                                  Dsv4KvBlockKind::Sliding, layer,
                                  1U, position, kv,
                                  sliding_retention_floor);
        if (!result.ok()) return result;
    } else {
        std::copy(kv.begin(), kv.end(),
                  layer_state.sliding.begin() +
                      static_cast<std::size_t>(position % kWindow) * kHeadDim);
    }
    result = compressor(layer, input, position,
                        compressor_values, compressor_scores);
    if (!result.ok()) return result;
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);
    if (append_index_compressor && layer_state.indexer_compressor.ratio != 0U) {
        subphase_started = std::chrono::steady_clock::now();
        result = compress_state(
            layer, layer_state.indexer_compressor,
            layer_prefix(layer) + "attn.indexer.compressor.", input, position,
            layer_state.frequencies, index_compressor_values,
            index_compressor_scores);
        if (!result.ok()) return result;
        graph_stats.attention_index_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_attend_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::uint32_t position, std::span<float> output,
    bool index_compressor_prepared, bool device_prepared_source,
    PhysicalAttentionPageSet* page_set) {
    ValidationResult result;
    if (input.size() != kHidden || query_rank.size() != kQueryRank ||
        (queries.size() != static_cast<std::size_t>(kHeads) * kHeadDim &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           queries.empty())) ||
        (output.size() != kHidden &&
         !(config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
           output.empty()))) {
        result.errors.emplace_back(
            "DeepSeek prepared attention spans have incompatible sizes");
        return result;
    }
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    auto& layer_state = attention_state[layer];
    std::vector<std::uint32_t> indexed_positions;
    const bool use_sparse_indexer =
        layer_state.compressor.ratio == 4U &&
        layer_state.indexer_compressor.ratio == 4U;
    if (use_sparse_indexer) {
        const auto subphase_started = std::chrono::steady_clock::now();
        if (index_compressor_prepared) {
            result = index_select(layer, input, query_rank, position,
                                  indexed_positions, device_prepared_source);
        } else {
            result = index_positions(layer, input, query_rank, position,
                                     indexed_positions, {}, {},
                                     device_prepared_source);
        }
        if (!result.ok()) return result;
        const auto selection_nanoseconds = elapsed_nanoseconds(subphase_started);
        graph_stats.attention_index_nanoseconds += selection_nanoseconds;
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            graph_stats.attention_page_index_selection_nanoseconds +=
                selection_nanoseconds;
        }
    }

    auto sink = host_tensor(prefix + "attn_sink", kHeads);
    if (!sink.ok()) {
        append_errors(result, std::move(sink.errors));
        return result;
    }
    std::vector<float> attended(
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice
            ? 0U : static_cast<std::size_t>(kHeads) * kHeadDim,
        0.0F);
    const auto window_count = std::min<std::uint32_t>(position + 1U, kWindow);
    const auto ratio = layer_state.compressor.ratio;
    const auto compressed_count = ratio == 0U ? 0U : (position + 1U) / ratio;
    const auto attended_compressed_count = use_sparse_indexer
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    const auto score_stride = static_cast<std::size_t>(window_count) +
                              attended_compressed_count;
    std::vector<std::vector<float>> block_rows;
    if (kv_cache != nullptr &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
        block_rows.reserve(score_stride);
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            auto row = kv_row(layer, Dsv4KvBlockKind::Sliding, absolute);
            if (!row.ok()) {
                append_errors(result, std::move(row.errors));
                return result;
            }
            block_rows.push_back(std::move(row.value));
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item] : item;
            auto row = kv_row(layer, layer_state.compressor.kind, cache_row);
            if (!row.ok()) {
                append_errors(result, std::move(row.errors));
                return result;
            }
            block_rows.push_back(std::move(row.value));
        }
    }
    const bool use_cuda_attention = should_dispatch_flash_attention_cuda(
        config.enable_flash_attention, score_stride,
        config.flash_attention_minimum_rows);
    auto subphase_started = std::chrono::steady_clock::now();
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto cuda_demand = weights->demand();
        ++graph_stats.attention_cuda_dispatches;
        result = physical_paged_attention(
            layer, queries, *sink.value, position, indexed_positions,
            output, page_set);
        if (!result.ok()) return result;
    } else if (use_cuda_attention) {
        auto cuda_demand = weights->demand();
        ++graph_stats.attention_cuda_dispatches;
        std::vector<std::uint32_t> sliding_rows(window_count);
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            sliding_rows[item] = absolute % kWindow;
        }
        std::vector<FlashAttentionSegment> segments;
        if (kv_cache != nullptr) {
            segments.reserve(block_rows.size());
            for (const auto& row : block_rows) {
                segments.push_back({std::span<const float>(row), {}, {}});
            }
        } else {
            segments.reserve(
                static_cast<std::size_t>(attended_compressed_count) + 1U);
            if (window_count != 0U) {
                segments.push_back({layer_state.sliding, {}, sliding_rows});
            }
            for (std::uint32_t item = 0U;
                 item < attended_compressed_count; ++item) {
                const auto cache_row = use_sparse_indexer
                    ? indexed_positions[item] : item;
                const auto row = layer_state.compressor.compressed.row(cache_row);
                if (row.size() != kHeadDim) {
                    result.errors.emplace_back(
                        "DeepSeek FlashAttention compressed row is unavailable");
                    return result;
                }
                segments.push_back({row, {}, {}});
            }
        }
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.head_sinks = *sink.value;
        request.query_rows = 1U;
        request.query_heads = kHeads;
        request.key_value_heads = 1U;
        request.query_key_dim = kHeadDim;
        request.value_dim = kHeadDim;
        request.scale = kAttentionScale;
        request.numerics =
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
        request.maximum_workspace_bytes = kDeviceWorkspaceReserve;
        result = cuda.flash_attention(devices[slot], request, attended);
        if (!result.ok()) return result;
        const auto finish_head = [&](std::uint32_t head) {
            auto destination = std::span<float>(attended).subspan(
                static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
            round_bf16(destination);
            apply_rope(destination.last(kRopeDim), position,
                       layer_state.frequencies, true);
            round_bf16(destination.last(kRopeDim));
        };
        if (attention_workers != nullptr) {
            result = attention_workers->parallel_for(
                kHeads, [&](std::size_t head) {
                    finish_head(static_cast<std::uint32_t>(head));
                });
            if (!result.ok()) return result;
        } else {
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                finish_head(head);
            }
        }
    } else {
        ++graph_stats.attention_scalar_dispatches;
    const auto attend_head = [&](std::uint32_t head,
                                 std::span<float> scores) {
        const auto query = std::span<const float>(queries).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t next_score = 0U;
        float maximum = (*sink.value)[head];
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto key = kv_cache == nullptr
                ? std::span<const float>(layer_state.sliding).subspan(
                      static_cast<std::size_t>(absolute % kWindow) * kHeadDim,
                      kHeadDim)
                : std::span<const float>(block_rows[item]);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item]
                : item;
            const auto key = kv_cache == nullptr
                ? layer_state.compressor.compressed.row(cache_row)
                : std::span<const float>(
                      block_rows[static_cast<std::size_t>(window_count) + item]);
            double dot = 0.0;
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                dot += static_cast<double>(query[dimension]) * key[dimension];
            }
            const float score = static_cast<float>(dot) * kAttentionScale;
            scores[next_score++] = score;
            maximum = std::max(maximum, score);
        }
        double denominator = std::exp(
            static_cast<double>((*sink.value)[head] - maximum));
        for (const float score : scores) {
            denominator += std::exp(static_cast<double>(score - maximum));
        }
        auto destination = std::span<float>(attended).subspan(
            static_cast<std::size_t>(head) * kHeadDim, kHeadDim);
        std::size_t score_index = 0U;
        for (std::uint32_t item = 0U; item < window_count; ++item) {
            const auto absolute = position + 1U - window_count + item;
            const auto value = kv_cache == nullptr
                ? std::span<const float>(layer_state.sliding).subspan(
                      static_cast<std::size_t>(absolute % kWindow) * kHeadDim,
                      kHeadDim)
                : std::span<const float>(block_rows[item]);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        for (std::uint32_t item = 0U; item < attended_compressed_count; ++item) {
            const auto cache_row = use_sparse_indexer
                ? indexed_positions[item]
                : item;
            const auto value = kv_cache == nullptr
                ? layer_state.compressor.compressed.row(cache_row)
                : std::span<const float>(
                      block_rows[static_cast<std::size_t>(window_count) + item]);
            const float probability = static_cast<float>(
                std::exp(static_cast<double>(scores[score_index++] - maximum)) /
                denominator);
            for (std::uint32_t dimension = 0U; dimension < kHeadDim; ++dimension) {
                destination[dimension] += probability * value[dimension];
            }
        }
        round_bf16(destination);
        apply_rope(destination.last(kRopeDim), position,
                   layer_state.frequencies, true);
        round_bf16(destination.last(kRopeDim));
    };
    if (attention_workers != nullptr) {
        std::vector<float> parallel_scores(
            static_cast<std::size_t>(kHeads) * score_stride);
        result = attention_workers->parallel_for(
            kHeads, [&](std::size_t head) {
                attend_head(
                    static_cast<std::uint32_t>(head),
                    std::span<float>(parallel_scores).subspan(
                        head * score_stride, score_stride));
            });
        if (!result.ok()) return result;
    } else {
        std::vector<float> scores(score_stride);
        for (std::uint32_t head = 0U; head < kHeads; ++head) {
            attend_head(head, scores);
        }
    }
    }
    graph_stats.attention_score_nanoseconds += elapsed_nanoseconds(subphase_started);

    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        return result;
    }

    subphase_started = std::chrono::steady_clock::now();
    std::vector<float> output_rank(static_cast<std::size_t>(kOutputGroups) *
                                   kOutputRank);
    result = weights->grouped(slot, prefix + "wo_a", kOutputGroups * kOutputRank,
                              kHeads * kHeadDim / kOutputGroups, attended,
                              kOutputGroups, kOutputRank, output_rank);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "wo_b", kHidden,
                    kOutputGroups * kOutputRank, output_rank, output);
    graph_stats.attention_output_nanoseconds += elapsed_nanoseconds(subphase_started);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::attention_prepared(
    std::uint32_t layer, std::span<const float> input,
    std::span<const float> query_rank, std::span<const float> queries,
    std::span<const float> kv, std::uint32_t position,
    std::span<float> output,
    std::span<const float> compressor_values,
    std::span<const float> compressor_scores,
    std::span<const float> index_compressor_values,
    std::span<const float> index_compressor_scores,
    bool device_prepared_source) {
    auto result = attention_append_prepared(
        layer, input, kv, position, compressor_values, compressor_scores,
        index_compressor_values, index_compressor_scores, true);
    if (!result.ok()) return result;
    return attention_attend_prepared(
        layer, input, query_rank, queries, position, output, true,
        device_prepared_source);
}

ValidationResult DeepSeekV4Runtime::Impl::attention_page(
    std::uint32_t layer, std::span<const float> input,
    std::uint32_t position_base, std::span<float> output,
    std::span<const std::uint32_t> row_slots) {
    ValidationResult result;
    if (input.empty() || input.size() % kHidden != 0U ||
        output.size() != input.size() ||
        (!row_slots.empty() && row_slots.size() != input.size() / kHidden)) {
        result.errors.emplace_back(
            "DeepSeek attention page spans have incompatible sizes");
        return result;
    }
    const auto row_count = input.size() / kHidden;
    if (row_count > std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back("DeepSeek attention page row count overflows");
        return result;
    }
    const auto rows = static_cast<std::uint32_t>(row_count);
    if (rows > config.prefill_page_tokens ||
        rows > config.maximum_context_tokens ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek attention page exceeds the admitted context bounds");
        return result;
    }

    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";
    const bool device_page_query =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        config.enable_dsv4_batched_page_attention && rows > 1U &&
        !row_slots.empty();
    const auto add_matmul_profile = [](const CudaMatmulProfile& profile,
                                       std::uint64_t& weight_acquisition,
                                       std::uint64_t& issue,
                                       std::uint64_t& finish,
                                       std::uint64_t& synchronization,
                                       std::uint64_t& h2d,
                                       std::uint64_t& kernel,
                                       std::uint64_t& d2h) {
        weight_acquisition += profile.weight_acquisition_nanoseconds;
        issue += profile.issue_nanoseconds;
        finish += profile.finish_nanoseconds;
        synchronization += profile.synchronization_nanoseconds;
        h2d += profile.h2d_nanoseconds;
        kernel += profile.kernel_nanoseconds;
        d2h += profile.d2h_nanoseconds;
    };
    auto subphase_started = std::chrono::steady_clock::now();
    auto allocation_started = std::chrono::steady_clock::now();
    auto query_rank = attention_page_query_rank_scratch.acquire(
        row_count * kQueryRank);
    graph_stats.attention_query_allocation_nanoseconds +=
        elapsed_nanoseconds(allocation_started);
    ++graph_stats.attention_projection_matmul_calls;
    graph_stats.attention_projection_matmul_rows += rows;
    CudaMatmulProfile query_a_profile;
    result = linear_rows(slot, prefix + "wq_a", kQueryRank, kHidden, input,
                         rows, query_rank, true, &query_a_profile,
                         config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    add_matmul_profile(
        query_a_profile,
        graph_stats.attention_query_weight_acquisition_nanoseconds,
        graph_stats.attention_query_matmul_issue_nanoseconds,
        graph_stats.attention_query_matmul_finish_nanoseconds,
        graph_stats.attention_query_matmul_sync_nanoseconds,
        graph_stats.attention_query_matmul_h2d_nanoseconds,
        graph_stats.attention_query_matmul_kernel_nanoseconds,
        graph_stats.attention_query_matmul_d2h_nanoseconds);
    const auto query_rank_norm_started = std::chrono::steady_clock::now();
    result = norm_rows(query_rank, query_rank, rows, kQueryRank,
                       prefix + "q_norm.weight");
    if (!result.ok()) return result;
    graph_stats.attention_query_rank_norm_nanoseconds +=
        elapsed_nanoseconds(query_rank_norm_started);

    const auto query_stride = static_cast<std::size_t>(kHeads) * kHeadDim;
    std::span<float> queries;
    if (!device_page_query) {
        allocation_started = std::chrono::steady_clock::now();
        queries = attention_page_query_scratch.acquire(
            row_count * query_stride);
        graph_stats.attention_query_allocation_nanoseconds +=
            elapsed_nanoseconds(allocation_started);
        ++graph_stats.attention_projection_matmul_calls;
        graph_stats.attention_projection_matmul_rows += rows;
        CudaMatmulProfile query_b_profile;
        result = linear_rows(slot, prefix + "wq_b", query_stride, kQueryRank,
                             query_rank, rows, queries, true, &query_b_profile,
                             config.enable_dsv4_fp8_tensor_page);
        if (!result.ok()) return result;
        add_matmul_profile(
            query_b_profile,
            graph_stats.attention_query_weight_acquisition_nanoseconds,
            graph_stats.attention_query_matmul_issue_nanoseconds,
            graph_stats.attention_query_matmul_finish_nanoseconds,
            graph_stats.attention_query_matmul_sync_nanoseconds,
            graph_stats.attention_query_matmul_h2d_nanoseconds,
            graph_stats.attention_query_matmul_kernel_nanoseconds,
            graph_stats.attention_query_matmul_d2h_nanoseconds);
        // One task per row rather than per (row, head). A head is about 1,500
        // flops, so at 64 heads a page of 677 rows dispatched 1.86 million
        // tasks across 43 layers and spent 11.2 s in pool overhead for work
        // that is nowhere near that size.
        std::atomic<std::uint64_t> query_rms_cpu_nanoseconds{0U};
        std::atomic<std::uint64_t> query_rope_cpu_nanoseconds{0U};
        const auto normalize_query_row = [&](std::size_t row) {
            auto cpu_started = std::chrono::steady_clock::now();
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                auto query = queries.subspan(
                    row * query_stride + head * kHeadDim, kHeadDim);
                double square_sum = 0.0;
                for (const float value : query) {
                    square_sum += static_cast<double>(value) * value;
                }
                const float reciprocal = 1.0F / std::sqrt(
                    static_cast<float>(square_sum / kHeadDim) + kRmsEpsilon);
                for (auto& value : query) {
                    value = round_bf16(value * reciprocal);
                }
            }
            query_rms_cpu_nanoseconds.fetch_add(
                elapsed_nanoseconds(cpu_started), std::memory_order_relaxed);
            cpu_started = std::chrono::steady_clock::now();
            for (std::uint32_t head = 0U; head < kHeads; ++head) {
                auto query = queries.subspan(
                    row * query_stride + head * kHeadDim, kHeadDim);
                apply_rope(query.last(kRopeDim),
                           position_base + static_cast<std::uint32_t>(row),
                           attention_state[layer].frequencies);
                round_bf16(query.last(kRopeDim));
            }
            query_rope_cpu_nanoseconds.fetch_add(
                elapsed_nanoseconds(cpu_started), std::memory_order_relaxed);
        };
        const auto query_finish_started = std::chrono::steady_clock::now();
        if (attention_workers != nullptr && row_count > 1U) {
            result = attention_workers->parallel_for(row_count,
                                                     normalize_query_row);
            if (!result.ok()) return result;
        } else {
            for (std::size_t row = 0U; row < row_count; ++row) {
                normalize_query_row(row);
            }
        }
        graph_stats.attention_query_finish_nanoseconds +=
            elapsed_nanoseconds(query_finish_started);
        graph_stats.attention_query_rms_cpu_nanoseconds +=
            query_rms_cpu_nanoseconds.load(std::memory_order_relaxed);
        graph_stats.attention_query_rope_cpu_nanoseconds +=
            query_rope_cpu_nanoseconds.load(std::memory_order_relaxed);
    } else {
        // The projection still executes once for this semantic page, but it is
        // now part of the bounded attention command and never materializes a
        // full host query tensor.
        ++graph_stats.attention_projection_matmul_calls;
        graph_stats.attention_projection_matmul_rows += rows;
    }
    graph_stats.attention_query_nanoseconds += elapsed_nanoseconds(subphase_started);

    subphase_started = std::chrono::steady_clock::now();
    allocation_started = std::chrono::steady_clock::now();
    auto kv = attention_page_kv_scratch.acquire(row_count * kHeadDim);
    graph_stats.attention_kv_allocation_nanoseconds +=
        elapsed_nanoseconds(allocation_started);
    ++graph_stats.attention_projection_matmul_calls;
    graph_stats.attention_projection_matmul_rows += rows;
    CudaMatmulProfile kv_profile;
    result = linear_rows(slot, prefix + "wkv", kHeadDim, kHidden, input, rows,
                         kv, true, &kv_profile,
                         config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    add_matmul_profile(
        kv_profile, graph_stats.attention_kv_weight_acquisition_nanoseconds,
        graph_stats.attention_kv_matmul_issue_nanoseconds,
        graph_stats.attention_kv_matmul_finish_nanoseconds,
        graph_stats.attention_kv_matmul_sync_nanoseconds,
        graph_stats.attention_kv_matmul_h2d_nanoseconds,
        graph_stats.attention_kv_matmul_kernel_nanoseconds,
        graph_stats.attention_kv_matmul_d2h_nanoseconds);
    const auto kv_norm_started = std::chrono::steady_clock::now();
    result = norm_rows(kv, kv, rows, kHeadDim, prefix + "kv_norm.weight");
    if (!result.ok()) return result;
    graph_stats.attention_kv_norm_nanoseconds +=
        elapsed_nanoseconds(kv_norm_started);
    const auto finish_kv = [&](std::size_t row) {
        auto kv_row = std::span<float>(kv).subspan(row * kHeadDim, kHeadDim);
        apply_rope(kv_row.last(kRopeDim),
                   position_base + static_cast<std::uint32_t>(row),
                   attention_state[layer].frequencies);
        round_bf16(kv_row.last(kRopeDim));
        if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            quantize_activation_in_place(
                kv_row.first(kHeadDim - kRopeDim), 64U);
        }
    };
    const auto kv_rope_started = std::chrono::steady_clock::now();
    if (attention_workers != nullptr && rows > 1U) {
        result = attention_workers->parallel_for(rows, finish_kv);
        if (!result.ok()) return result;
    } else {
        for (std::uint32_t row = 0U; row < rows; ++row) finish_kv(row);
    }
    graph_stats.attention_kv_rope_nanoseconds +=
        elapsed_nanoseconds(kv_rope_started);
    graph_stats.attention_kv_nanoseconds += elapsed_nanoseconds(subphase_started);

    auto& layer_state = attention_state[layer];
    const auto last_position = position_base + rows - 1U;
    const auto ratio = layer_state.compressor.ratio;
    const bool use_sparse_indexer =
        ratio == 4U && layer_state.indexer_compressor.ratio == 4U;
    const auto maximum_score_rows =
        std::min(last_position + 1U, kWindow) +
        (ratio == 0U ? 0U : (last_position + 1U) / ratio);
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        config.enable_dsv4_batched_page_attention) {
        // A physical KV block rejects mutation while any device lease is
        // outstanding. Append every row (including both compressor states)
        // before resolving or leasing a page, then attend the rows in order
        // against one page-set/map. The candidates remain row-local, so
        // causality and candidate order are unchanged.
        PhysicalAttentionPageSet page_set;
        const auto sliding_retention_floor =
            position_base + 1U > kWindow
                ? static_cast<std::uint64_t>(position_base + 1U - kWindow)
                : 0U;
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto kv_row_values = std::span<const float>(kv).subspan(
                static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
            result = attention_append_prepared(
                layer, input_row, kv_row_values,
                position_base + row, {}, {}, {}, {}, true,
                sliding_retention_floor);
            if (!result.ok()) return result;
        }
        if (rows > 1U && !row_slots.empty()) {
            auto sink = host_tensor(prefix + "attn_sink", kHeads);
            if (!sink.ok()) {
                append_errors(result, std::move(sink.errors));
                return result;
            }
            subphase_started = std::chrono::steady_clock::now();
            result = physical_paged_attention_page(
                layer, input, query_rank, queries, *sink.value,
                position_base, row_slots, output);
            graph_stats.attention_score_nanoseconds +=
                elapsed_nanoseconds(subphase_started);
            return result;
        }
        for (std::uint32_t row = 0U; row < rows; ++row) {
            if (!row_slots.empty()) {
                result = cuda.dsv4_mhc_select_slot(
                    devices[mhc_slot], row_slots[row]);
                if (!result.ok()) return result;
            }
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto query_rank_row = std::span<const float>(query_rank)
                .subspan(static_cast<std::size_t>(row) * kQueryRank, kQueryRank);
            const auto queries_row = std::span<const float>(queries).subspan(
                static_cast<std::size_t>(row) * query_stride, query_stride);
            auto output_row = output.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            result = attention_attend_prepared(
                layer, input_row, query_rank_row, queries_row,
                position_base + row, output_row, true, false, &page_set);
            if (!result.ok()) return result;
        }
        return result;
    }
    const bool batch_cuda =
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice &&
        rows > 1U && kv_cache != nullptr &&
        !use_sparse_indexer && should_dispatch_flash_attention_cuda(
            config.enable_flash_attention, maximum_score_rows,
            config.flash_attention_minimum_rows);
    if (batch_cuda) {
        const auto sliding_begin = position_base + 1U > kWindow
            ? position_base + 1U - kWindow : 0U;
        const auto sliding_end = last_position + 1U;
        const auto sliding_rows = sliding_end - sliding_begin;
        const auto historical_sliding_rows = position_base - sliding_begin;
        const auto compressed_rows = ratio == 0U
            ? 0U : (last_position + 1U) / ratio;
        const auto key_rows = sliding_rows + compressed_rows;
        std::vector<float> gathered(
            static_cast<std::size_t>(key_rows) * kHeadDim);
        for (std::uint32_t row = 0U; row < historical_sliding_rows; ++row) {
            auto source = kv_row(layer, Dsv4KvBlockKind::Sliding,
                                 sliding_begin + row);
            if (!source.ok()) {
                append_errors(result, std::move(source.errors));
                return result;
            }
            std::copy(source.value.begin(), source.value.end(),
                      gathered.begin() +
                          static_cast<std::ptrdiff_t>(row) * kHeadDim);
        }

        subphase_started = std::chrono::steady_clock::now();
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto position = position_base + row;
            const auto input_row = input.subspan(
                static_cast<std::size_t>(row) * kHidden, kHidden);
            const auto kv_row_values = std::span<const float>(kv).subspan(
                static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
            result = kv_cache->append(active_sequence,
                                      Dsv4KvBlockKind::Sliding, layer,
                                      1U, position, kv_row_values);
            if (!result.ok()) return result;
            result = compressor(layer, input_row, position);
            if (!result.ok()) return result;
            std::copy(kv_row_values.begin(), kv_row_values.end(),
                      gathered.begin() + static_cast<std::ptrdiff_t>(
                          historical_sliding_rows + row) * kHeadDim);
        }
        graph_stats.attention_kv_nanoseconds +=
            elapsed_nanoseconds(subphase_started);

        auto sink = host_tensor(prefix + "attn_sink", kHeads);
        if (!sink.ok()) {
            append_errors(result, std::move(sink.errors));
            return result;
        }
        for (std::uint32_t row = 0U; row < compressed_rows; ++row) {
            auto source = kv_row(layer, layer_state.compressor.kind, row);
            if (!source.ok()) {
                append_errors(result, std::move(source.errors));
                return result;
            }
            std::copy(source.value.begin(), source.value.end(),
                      gathered.begin() + static_cast<std::ptrdiff_t>(
                          sliding_rows + row) * kHeadDim);
        }

        std::vector<std::uint8_t> mask(
            static_cast<std::size_t>(rows) * key_rows, 0U);
        for (std::uint32_t row = 0U; row < rows; ++row) {
            const auto position = position_base + row;
            const auto window_count = std::min(position + 1U, kWindow);
            const auto window_begin = position + 1U - window_count;
            auto row_mask = std::span<std::uint8_t>(mask).subspan(
                static_cast<std::size_t>(row) * key_rows, key_rows);
            std::fill(row_mask.begin() +
                          static_cast<std::ptrdiff_t>(window_begin - sliding_begin),
                      row_mask.begin() + static_cast<std::ptrdiff_t>(
                          position + 1U - sliding_begin),
                      1U);
            const auto visible_compressed = ratio == 0U
                ? 0U : (position + 1U) / ratio;
            std::fill_n(row_mask.begin() + sliding_rows,
                        visible_compressed, 1U);
        }

        subphase_started = std::chrono::steady_clock::now();
        graph_stats.attention_cuda_dispatches += rows;
        const std::array<FlashAttentionSegment, 1> segments{{
            {gathered, {}, {}}}};
        std::vector<float> attended(
            row_count * static_cast<std::size_t>(kHeads) * kHeadDim);
        FlashAttentionRequest request;
        request.queries = queries;
        request.segments = segments;
        request.head_sinks = *sink.value;
        request.query_key_mask = mask;
        request.query_rows = rows;
        request.query_heads = kHeads;
        request.key_value_heads = 1U;
        request.query_key_dim = kHeadDim;
        request.value_dim = kHeadDim;
        request.scale = kAttentionScale;
        request.numerics =
            FlashAttentionNumerics::f64_dot_f32_score_f32_accum;
        request.maximum_workspace_bytes = kDeviceWorkspaceReserve;
        {
            auto cuda_demand = weights->demand();
            result = cuda.flash_attention(devices[slot], request, attended);
        }
        if (!result.ok()) return result;
        const auto finish_head = [&](std::size_t task) {
            const auto row = task / kHeads;
            auto destination = std::span<float>(attended).subspan(
                task * kHeadDim, kHeadDim);
            round_bf16(destination);
            apply_rope(destination.last(kRopeDim),
                       position_base + static_cast<std::uint32_t>(row),
                       layer_state.frequencies, true);
            round_bf16(destination.last(kRopeDim));
        };
        if (attention_workers != nullptr) {
            result = attention_workers->parallel_for(
                row_count * kHeads, finish_head);
            if (!result.ok()) return result;
        } else {
            for (std::size_t task = 0U; task < row_count * kHeads; ++task) {
                finish_head(task);
            }
        }
        graph_stats.attention_score_nanoseconds +=
            elapsed_nanoseconds(subphase_started);

        subphase_started = std::chrono::steady_clock::now();
        std::vector<float> output_rank(
            row_count * static_cast<std::size_t>(kOutputGroups) * kOutputRank);
        result = weights->grouped_rows(
            slot, prefix + "wo_a", kOutputGroups * kOutputRank,
            kHeads * kHeadDim / kOutputGroups, attended, rows,
            kOutputGroups, kOutputRank, output_rank);
        if (!result.ok()) return result;
        result = linear_rows(slot, prefix + "wo_b", kHidden,
                             kOutputGroups * kOutputRank, output_rank,
                             rows, output, true, nullptr,
                             config.enable_dsv4_fp8_tensor_page);
        graph_stats.attention_output_nanoseconds +=
            elapsed_nanoseconds(subphase_started);
        return result;
    }

    for (std::uint32_t row = 0U; row < rows; ++row) {
        if (!row_slots.empty()) {
            result = cuda.dsv4_mhc_select_slot(
                devices[mhc_slot], row_slots[row]);
            if (!result.ok()) return result;
        }
        const auto input_row = input.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        const auto query_rank_row = std::span<const float>(query_rank).subspan(
            static_cast<std::size_t>(row) * kQueryRank, kQueryRank);
        const auto queries_row = std::span<const float>(queries).subspan(
            static_cast<std::size_t>(row) * query_stride, query_stride);
        const auto kv_row = std::span<const float>(kv).subspan(
            static_cast<std::size_t>(row) * kHeadDim, kHeadDim);
        auto output_row = output.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        result = attention_prepared(layer, input_row, query_rank_row,
                                    queries_row, kv_row, position_base + row,
                                    output_row);
        if (!result.ok()) return result;
    }
    return result;
}

