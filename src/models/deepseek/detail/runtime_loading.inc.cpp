void DeepSeekV4Runtime::Impl::reset_diagnostics() {
    diagnostics = {};
    diagnostics.logit_trace_enabled = config.enable_logit_trace;
    diagnostics.layer_hash_trace_enabled = config.enable_layer_hash_trace;
    diagnostics.logit_top_k = config.logit_trace_top_k;
    diagnostics.index_selection_trace_hash = kDiagnosticFnvOffset;
    graph_stats.prefill_pages = 0U;
    graph_stats.prefill_max_page_tokens = 0U;
    graph_stats.prefill_max_workspace_bytes = 0U;
    pending_prefetch_predictions.clear();
    deferred_route_events.clear();
    defer_prefill_observability = false;
    if (config.enable_logit_trace) {
        diagnostics.logit_aggregate.trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, config.logit_trace_top_k);
        diagnostics.logits.reserve(config.maximum_context_tokens);
    }
    if (config.enable_layer_hash_trace) {
        diagnostics.layer_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
        diagnostics.layer_hashes.reserve(
            static_cast<std::size_t>(config.maximum_context_tokens) * kLayers);
        diagnostics.operation_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
    }
}

void DeepSeekV4Runtime::Impl::record_layer_hash(
    std::uint32_t position, std::uint32_t token, std::uint32_t layer,
    std::span<const float> hidden) {
    const auto hash = stable_bf16_hash(hidden);
    diagnostics.layer_hashes.push_back({position, token, layer, hash});
    auto aggregate = diagnostics.layer_hash_trace_hash;
    aggregate = diagnostic_hash_u32(aggregate, position);
    aggregate = diagnostic_hash_u32(aggregate, token);
    aggregate = diagnostic_hash_u32(aggregate, layer);
    diagnostics.layer_hash_trace_hash = diagnostic_hash_u64(aggregate, hash);
}

void DeepSeekV4Runtime::Impl::record_operation_hash(
    std::uint32_t position, std::uint32_t token,
    std::uint32_t layer, std::string_view operation,
    std::span<const float> values) {
    const auto hash = stable_bf16_hash(values);
    diagnostics.operation_hashes.push_back(
        {position, token, layer, std::string(operation), hash});
    auto aggregate = diagnostics.operation_hash_trace_hash;
    aggregate = diagnostic_hash_u32(aggregate, position);
    aggregate = diagnostic_hash_u32(aggregate, token);
    aggregate = diagnostic_hash_u32(aggregate, layer);
    for (const char ch : operation) {
        aggregate = diagnostic_hash_byte(aggregate, static_cast<std::uint8_t>(ch));
    }
    diagnostics.operation_hash_trace_hash = diagnostic_hash_u64(aggregate, hash);
}

void DeepSeekV4Runtime::Impl::record_logits(
    std::uint32_t position, std::uint32_t token, std::uint32_t selected,
    std::span<const float> logits) {
    auto analysis = analyze_logits(logits, config.logit_trace_top_k);
    const auto& summary = analysis.summary;
    auto& aggregate = diagnostics.logit_aggregate;
    ++aggregate.forward_count;
    aggregate.value_count += summary.value_count;
    aggregate.finite_count += summary.finite_count;
    aggregate.non_finite_count += summary.non_finite_count;
    aggregate.sum += summary.sum;
    aggregate.absolute_sum += summary.absolute_sum;
    aggregate.square_sum += summary.square_sum;
    if (summary.has_finite) {
        if (!aggregate.has_finite) {
            aggregate.minimum = summary.minimum;
            aggregate.maximum = summary.maximum;
            aggregate.has_finite = true;
        } else {
            aggregate.minimum = std::min(aggregate.minimum, summary.minimum);
            aggregate.maximum = std::max(aggregate.maximum, summary.maximum);
        }
    }
    auto hash = aggregate.trace_hash;
    hash = diagnostic_hash_u32(hash, position);
    hash = diagnostic_hash_u32(hash, token);
    hash = diagnostic_hash_u32(hash, selected);
    aggregate.trace_hash = diagnostic_hash_u64(hash, summary.raw_f32_hash);
    diagnostics.logits.push_back(
        {position, token, selected, summary, std::move(analysis.top)});
}

ValidationResult DeepSeekV4Runtime::Impl::warmup() {
    ValidationResult result;
    const auto preload = [this](ValidationResult& target, std::size_t slot,
                                const std::string& base, std::uint64_t rows,
                                std::uint64_t columns) {
        if (target.ok()) target = weights->preload(slot, base, rows, columns);
    };
    const auto load_host = [this](ValidationResult& target, const std::string& name,
                                  std::uint64_t ceiling) {
        if (!target.ok()) return;
        auto loaded = host_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(target, std::move(loaded.errors));
    };
    const auto load_raw = [this](ValidationResult& target, const std::string& name,
                                 std::uint64_t ceiling) {
        if (!target.ok()) return;
        auto loaded = raw_tensor(name, ceiling);
        if (!loaded.ok()) append_errors(target, std::move(loaded.errors));
    };
    const auto load_mhc = [this](ValidationResult& target,
                                 const std::string& name) {
        if (!target.ok()) return;
        auto loaded = host_tensor(name, kMix * kMhc * kHidden);
        if (!loaded.ok()) {
            append_errors(target, std::move(loaded.errors));
            return;
        }
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice ||
            !config.prepack_mhc_projection) {
            return;
        }
        try {
            std::vector<float> packed(loaded.value->size());
            auto status = dsv4_pack_mhc_projection_f32(
                packed, *loaded.value, kMix, kMhc * kHidden);
            if (!status.ok()) {
                append_errors(target, std::move(status.errors));
                return;
            }
            prepacked_mhc.emplace(name, std::move(packed));
        } catch (const std::bad_alloc&) {
            target.errors.emplace_back(
                "cannot allocate DeepSeek prepacked mHC projection");
        }
    };

    const auto ratios = deepseek_v4_flash_0731_spec().deepseek_v4.compression_ratios;
    const auto preload_layer = [this, &preload, &ratios](
        ValidationResult& target, std::uint32_t layer) {
        const auto slot = layer_device(layer);
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "attn.";
        preload(target, slot, attention + "wq_a", kQueryRank, kHidden);
        preload(target, slot, attention + "wq_b", kHeads * kHeadDim, kQueryRank);
        preload(target, slot, attention + "wkv", kHeadDim, kHidden);
        preload(target, slot, attention + "wo_a", kOutputGroups * kOutputRank,
                kHeads * kHeadDim / kOutputGroups);
        preload(target, slot, attention + "wo_b", kHidden,
                kOutputGroups * kOutputRank);
        preload(target, slot, prefix + "ffn.gate", kExperts, kHidden);
        for (const auto* operation : {"w1", "w3"}) {
            preload(target, slot, prefix + "ffn.shared_experts." + operation,
                    kExpertIntermediate, kHidden);
        }
        preload(target, slot, prefix + "ffn.shared_experts.w2", kHidden,
                kExpertIntermediate);
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto dimensions = coefficient * kHeadDim;
            preload(target, slot, attention + "compressor.wkv", dimensions, kHidden);
            preload(target, slot, attention + "compressor.wgate", dimensions, kHidden);
            if (ratio == 4U) {
                preload(target, slot, attention + "indexer.wq_b", 64U * 128U,
                        kQueryRank);
                preload(target, slot, attention + "indexer.weights_proj", 64U, kHidden);
                preload(target, slot, attention + "indexer.compressor.wkv", 2U * 128U,
                        kHidden);
                preload(target, slot, attention + "indexer.compressor.wgate", 2U * 128U,
                        kHidden);
            }
        }
    };
    const auto load_host_layer = [this, &load_host, &load_raw, &load_mhc, &ratios](
        ValidationResult& target, std::uint32_t layer) {
        const auto prefix = layer_prefix(layer);
        const auto attention = prefix + "attn.";
        load_host(target, attention + "q_norm.weight", kQueryRank);
        load_host(target, attention + "kv_norm.weight", kHeadDim);
        load_host(target, attention + "attn_sink", kHeads);
        load_host(target, prefix + "attn_norm.weight", kHidden);
        load_host(target, prefix + "ffn_norm.weight", kHidden);
        constexpr std::array<const char*, 2U> branches{"attn", "ffn"};
        for (std::size_t branch_index = 0U;
             branch_index < branches.size(); ++branch_index) {
            const std::string branch(branches[branch_index]);
            const auto projection_name = prefix + "hc_" + branch + "_fn";
            const auto scale_name = prefix + "hc_" + branch + "_scale";
            const auto base_name = prefix + "hc_" + branch + "_base";
            const auto norm_name = prefix + branch + "_norm.weight";
            load_mhc(target, projection_name);
            load_host(target, scale_name, 3U);
            load_host(target, base_name, kMix);
            if (!target.ok() ||
                config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
                continue;
            }
            auto projection = host_tensor(
                projection_name, kMix * kMhc * kHidden);
            auto scale = host_tensor(scale_name, 3U);
            auto base = host_tensor(base_name, kMix);
            auto norm_weight = host_tensor(norm_name, kHidden);
            if (!projection.ok()) {
                append_errors(target, std::move(projection.errors));
            }
            if (!scale.ok()) append_errors(target, std::move(scale.errors));
            if (!base.ok()) append_errors(target, std::move(base.errors));
            if (!norm_weight.ok()) {
                append_errors(target, std::move(norm_weight.errors));
            }
            if (!target.ok()) continue;
            target = cuda.upload_dsv4_mhc_weights(
                devices[mhc_slot], *projection.value, *scale.value,
                *base.value, *norm_weight.value,
                device_mhc_weights[layer][branch_index]);
            if (target.ok()) host_tensors.erase(projection_name);
        }
        if (layer < 3U) {
            load_raw(target, prefix + "ffn.gate.tid2eid",
                     8ULL * kVocabulary * kTopK);
        } else {
            load_host(target, prefix + "ffn.gate.bias", kExperts);
        }
        const auto ratio = ratios[layer];
        if (ratio != 0U) {
            const auto coefficient = ratio == 4U ? 2U : 1U;
            const auto dimensions = coefficient * kHeadDim;
            load_host(target, attention + "compressor.ape",
                      static_cast<std::uint64_t>(ratio) * dimensions);
            load_host(target, attention + "compressor.norm.weight", kHeadDim);
            if (ratio == 4U) {
                load_host(target, attention + "indexer.compressor.ape",
                          4U * 2U * 128U);
                load_host(target, attention + "indexer.compressor.norm.weight", 128U);
            }
        }
    };
    const auto preload_head = [this, &preload](ValidationResult& target) {
        preload(target, layer_device(kLayers - 1U), "head", kVocabulary, kHidden);
    };
    const auto load_host_head = [this, &load_host](ValidationResult& target) {
        load_host(target, "norm.weight", kHidden);
        load_host(target, "hc_head_fn", kMhc * kMhc * kHidden);
        load_host(target, "hc_head_scale", 1U);
        load_host(target, "hc_head_base", kMhc);
    };

    if (config.spine_warmup_workers == 1U) {
        for (std::uint32_t layer = 0U; layer < kLayers && result.ok(); ++layer) {
            preload_layer(result, layer);
            load_host_layer(result, layer);
            if (config.verbose) {
                std::cerr << "[deepseek-load] resident spine layer " << layer + 1U
                          << '/' << kLayers << '\n';
            }
        }
        if (result.ok()) {
            preload_head(result);
            load_host_head(result);
            if (result.ok() &&
                config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
                result = cuda.reserve_dsv4_mhc_head(
                    devices[mhc_slot], kVocabulary);
            }
        }
        return result;
    }

    std::vector<ValidationResult> device_results(devices.size());
    std::atomic<std::size_t> next_slot{};
    const auto worker = [&] {
        for (;;) {
            const auto slot = next_slot.fetch_add(1U, std::memory_order_relaxed);
            if (slot >= devices.size()) return;
            auto& target = device_results[slot];
            for (std::uint32_t layer = 0U; layer < kLayers && target.ok(); ++layer) {
                if (layer_device(layer) == slot) preload_layer(target, layer);
            }
            if (target.ok() && layer_device(kLayers - 1U) == slot) {
                preload_head(target);
            }
        }
    };
    const auto active_workers = std::min<std::size_t>(
        config.spine_warmup_workers, devices.size());
    std::vector<std::thread> workers;
    workers.reserve(active_workers);
    for (std::size_t index = 0U; index < active_workers; ++index) {
        workers.emplace_back(worker);
    }
    for (auto& thread : workers) thread.join();
    for (auto& device_result : device_results) {
        if (!device_result.ok()) append_errors(result, std::move(device_result.errors));
    }
    for (std::uint32_t layer = 0U; layer < kLayers && result.ok(); ++layer) {
        load_host_layer(result, layer);
        if (config.verbose) {
            std::cerr << "[deepseek-load] resident spine layer " << layer + 1U
                      << '/' << kLayers << '\n';
        }
    }
    if (result.ok()) load_host_head(result);
    if (result.ok() &&
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        result = cuda.reserve_dsv4_mhc_head(devices[mhc_slot], kVocabulary);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::mhc_pre(
    std::span<float> reduced, Dsv4MhcMix& mix,
    std::span<const float> hidden, const std::string& projection_name,
    std::span<const float> projection, std::span<const float> scale,
    std::span<const float> base, bool parallel_projection) {
    if (!config.prepack_mhc_projection) {
        return dsv4_mhc_pre_f32(
            reduced, mix, hidden, projection, scale, base);
    }
    const auto found = prepacked_mhc.find(projection_name);
    if (found == prepacked_mhc.end()) {
        ValidationResult result;
        result.errors.emplace_back(
            "DeepSeek prepacked mHC projection is unavailable: " +
            projection_name);
        return result;
    }
    ++graph_stats.mhc_prepacked_calls;
    Dsv4ParallelFor lanes;
    if (parallel_projection && attention_workers != nullptr &&
        attention_workers->size() > 1U) {
        lanes = [this](std::size_t tasks,
                       const std::function<void(std::size_t)>& body) {
            return attention_workers->parallel_for(tasks, body);
        };
    }
    return dsv4_mhc_prepacked_f32(
        reduced, mix, hidden, found->second, scale, base,
        kDsv4MhcMultiplier, kDsv4MhcSinkhornIterations, kDsv4NormEpsilon,
        lanes);
}

void DeepSeekV4Runtime::Impl::release_retained_kv_leases() noexcept {
    for (auto& scratch : rank_local_scratch) {
        scratch.compressed_block_leased.clear();
        scratch.leases.clear();
        scratch.index_leases.clear();
        for (auto& pages : scratch.pages) pages.clear();
        for (auto& pages : scratch.index_pages) pages.clear();
    }
    // The centralized path parks its leases here until the matching MoE
    // collect, which an ended generation never reaches.
    pending_attention_leases.clear();
}

ValidationResult DeepSeekV4Runtime::Impl::reset_sequence(
    std::uint32_t active_context_tokens) {
    ValidationResult result;
    reusable_sequence = false;
    cached_token_ids.clear();
    // Page slots and leases describe blocks of the sequence being discarded,
    // where the same indices will refer to different pages in the next one.
    release_retained_kv_leases();
    if (kv_cache != nullptr) {
        result = kv_cache->reset_sequence(active_sequence);
        if (!result.ok()) return result;
    }
    const auto ratios = deepseek_v4_flash_0731_spec().deepseek_v4.compression_ratios;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        auto& state = attention_state[layer];
        if (kv_cache == nullptr) {
            state.sliding.assign(static_cast<std::size_t>(kWindow) * kHeadDim,
                                 0.0F);
        } else {
            state.sliding.clear();
        }
        state.frequencies = rope_frequencies(ratios[layer]);
        auto& compressor_state = state.compressor;
        compressor_state = {};
        compressor_state.ratio = ratios[layer];
        if (compressor_state.ratio == 0U) continue;
        compressor_state.kind = compressor_state.ratio == 4U
            ? Dsv4KvBlockKind::Csa : Dsv4KvBlockKind::Hca;
        compressor_state.coefficient = compressor_state.ratio == 4U ? 2U : 1U;
        compressor_state.head_dim = kHeadDim;
        const auto rows = static_cast<std::size_t>(compressor_state.coefficient) *
                          compressor_state.ratio;
        const auto dimensions = static_cast<std::size_t>(
            compressor_state.coefficient) * compressor_state.head_dim;
        compressor_state.values.assign(rows * dimensions, 0.0F);
        compressor_state.scores.assign(
            rows * dimensions, -std::numeric_limits<float>::infinity());
        const auto compressed_rows =
            (static_cast<std::size_t>(config.maximum_context_tokens) +
             compressor_state.ratio - 1U) / compressor_state.ratio;
        if (kv_cache == nullptr &&
            !compressor_state.compressed.configure(compressed_rows,
                                                   compressor_state.head_dim)) {
            result.errors.emplace_back(
                "cannot reserve paged DeepSeek compressed KV metadata");
            return result;
        }
        state.indexer_compressor = {};
        if (compressor_state.ratio == 4U &&
            active_context_tokens > kIndexTopK * compressor_state.ratio) {
            auto& indexer = state.indexer_compressor;
            indexer.ratio = compressor_state.ratio;
            indexer.kind = Dsv4KvBlockKind::LearnedIndex;
            indexer.coefficient = 2U;
            indexer.head_dim = kIndexHeadDim;
            indexer.rotate_fp4 =
                config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice;
            const auto indexer_rows = static_cast<std::size_t>(indexer.coefficient) *
                                      indexer.ratio;
            const auto indexer_dimensions =
                static_cast<std::size_t>(indexer.coefficient) * indexer.head_dim;
            indexer.values.assign(indexer_rows * indexer_dimensions, 0.0F);
            indexer.scores.assign(
                indexer_rows * indexer_dimensions,
                -std::numeric_limits<float>::infinity());
            if (kv_cache == nullptr &&
                !indexer.compressed.configure(compressed_rows,
                                              indexer.head_dim)) {
                result.errors.emplace_back(
                    "cannot reserve paged DeepSeek sparse-index metadata");
                return result;
            }
        }
    }
    return result;
}

ParseResult<std::vector<float>> DeepSeekV4Runtime::Impl::kv_row(
    std::uint32_t layer, Dsv4KvBlockKind kind,
    std::uint64_t logical_row) {
    if (kv_cache != nullptr) {
        return kv_cache->row(active_sequence, kind, layer, logical_row);
    }
    ParseResult<std::vector<float>> result;
    const auto& state = attention_state[layer];
    if (kind == Dsv4KvBlockKind::Sliding) {
        const auto values = std::span<const float>(state.sliding).subspan(
            static_cast<std::size_t>(logical_row % kWindow) * kHeadDim,
            kHeadDim);
        result.value.assign(values.begin(), values.end());
        return result;
    }
    const auto& compressed = kind == Dsv4KvBlockKind::LearnedIndex
        ? state.indexer_compressor.compressed : state.compressor.compressed;
    const auto values = compressed.row(logical_row);
    if (values.empty()) {
        result.errors.emplace_back("DeepSeek scalar KV row is unavailable");
    } else {
        result.value.assign(values.begin(), values.end());
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::embed(std::uint32_t token,
                                                 std::span<float> output) {
    ValidationResult result;
    if (token >= kVocabulary || output.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek embedding token or output shape is invalid");
        return result;
    }
    const auto embedding = resident.find("embed.weight");
    const auto row_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    const auto offset = static_cast<std::size_t>(token) * row_bytes;
    if (embedding.size() < offset + row_bytes) {
        result.errors.emplace_back("DeepSeek resident embedding extent is incomplete");
        return result;
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        std::uint16_t encoded = 0U;
        std::memcpy(&encoded, embedding.data() + offset + column * sizeof(encoded),
                    sizeof(encoded));
        const float value = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded) << 16U);
        for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
            output[static_cast<std::size_t>(copy) * kHidden + column] = value;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::compressor(
    std::uint32_t layer, std::span<const float> input, std::uint32_t position,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores) {
    return compress_state(layer, attention_state[layer].compressor,
                          layer_prefix(layer) + "attn.compressor.", input,
                          position, attention_state[layer].frequencies,
                          prepared_values, prepared_scores);
}

bool DeepSeekV4Runtime::Impl::physical_attention_prepare_callback(
    void* opaque, const CudaDsv4AttentionPrepareHostView& view) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<PhysicalAttentionContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->complete_physical_attention_prepare(context, view);
}

bool DeepSeekV4Runtime::Impl::rank_local_page_patch_callback(
    void* opaque, const CudaDsv4AttentionPrepareHostView& view) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<RankLocalPageContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->complete_rank_local_page_patch(context, view);
}

bool DeepSeekV4Runtime::Impl::complete_rank_local_page_patch(
    RankLocalPageContext& context,
    const CudaDsv4AttentionPrepareHostView& view) {
    const auto started = std::chrono::steady_clock::now();
    context.invoked = true;
    context.result = {};
    auto* scratch = context.scratch;
    const auto complete = [&](bool success) {
        context.elapsed_nanoseconds = ::strata::elapsed_nanoseconds(started);
        return success;
    };
    const auto fail = [&](std::string message) {
        context.result.errors.push_back(std::move(message));
        return complete(false);
    };
    if (context.transaction == nullptr || scratch == nullptr ||
        context.layer >= kLayers || context.rank >= kDsv4RankLocalWorld ||
        view.query_rank.size() != kQueryRank ||
        view.key_value.size() != kHeadDim) {
        return fail("rank-local deferred page-patch shape is invalid");
    }

    const auto& state = attention_state[context.layer];
    const auto compressor_elements = static_cast<std::size_t>(
        state.compressor.coefficient) * state.compressor.head_dim;
    const auto index_elements = static_cast<std::size_t>(
        state.indexer_compressor.coefficient) *
        state.indexer_compressor.head_dim;
    if (view.compressor_values.size() != compressor_elements ||
        view.compressor_scores.size() != compressor_elements ||
        view.index_compressor_values.size() != index_elements ||
        view.index_compressor_scores.size() != index_elements ||
        view.page_patches.size() != scratch->replica_patch.size()) {
        return fail("rank-local canonical page-patch payload is invalid");
    }

    scratch->key_value.resize(kHeadDim);
    for (std::size_t index = 0U; index < kHeadDim; ++index) {
        scratch->key_value[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(view.key_value[index]) << 16U);
    }

    const auto prefix = layer_prefix(context.layer) + "attn.";
    context.result = compress_state(
        context.layer, attention_state[context.layer].compressor,
        prefix + "compressor.", {}, context.position,
        attention_state[context.layer].frequencies,
        view.compressor_values, view.compressor_scores, nullptr, {},
        &scratch->compressed_row);
    if (context.result.ok()) {
        context.result = compress_state(
            context.layer, attention_state[context.layer].indexer_compressor,
            prefix + "indexer.compressor.", {}, context.position,
            attention_state[context.layer].frequencies,
            view.index_compressor_values, view.index_compressor_scores,
            nullptr, {}, &scratch->index_row);
    }
    if (context.result.ok()) {
        std::array<std::span<std::byte>, kDsv4RankLocalWorld> patches{
            view.page_patches, scratch->replica_patch};
        context.result = context.transaction->commit_layer(
            context.layer, scratch->key_value, scratch->compressed_row,
            patches, scratch->index_row);
    }
    return complete(context.result.ok());
}

bool DeepSeekV4Runtime::Impl::complete_physical_attention_prepare(
    PhysicalAttentionContext& context,
    const CudaDsv4AttentionPrepareHostView& view) {
    context.invoked = true;
    context.result = {};
    if (context.layer >= kLayers || view.key_value.size() != kHeadDim ||
        !context.sliding_append.has_value()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred attention preparation shape is invalid");
        return false;
    }
    const auto decode = [](std::span<const std::uint16_t> source,
                           std::span<float> destination) {
        for (std::size_t index = 0U; index < source.size(); ++index) {
            destination[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(source[index]) << 16U);
        }
    };
    decode(view.key_value, context.key_value);
    std::size_t patch_cursor = 0U;
    const auto commit = [&](Dsv4KvPhysicalAppend& append,
                            std::span<const float> values) {
        const auto bytes = static_cast<std::size_t>(append.patch_bytes());
        if (bytes > view.page_patches.size() - patch_cursor) {
            context.result.errors.emplace_back(
                "DeepSeek deferred page-patch staging is truncated");
            return false;
        }
        auto committed = append.commit(
            values, view.page_patches.subspan(patch_cursor, bytes));
        patch_cursor += bytes;
        if (!committed.ok()) {
            append_errors(context.result, std::move(committed.errors));
            return false;
        }
        return true;
    };
    if (!commit(*context.sliding_append, context.key_value)) return false;

    auto& layer_state = attention_state[context.layer];
    auto* compressed_append = context.compressed_append.has_value()
        ? &*context.compressed_append : nullptr;
    std::span<std::byte> compressed_patch;
    if (compressed_append != nullptr) {
        const auto bytes = static_cast<std::size_t>(
            compressed_append->patch_bytes());
        if (bytes > view.page_patches.size() - patch_cursor) {
            context.result.errors.emplace_back(
                "DeepSeek deferred compressed-page staging is truncated");
            return false;
        }
        compressed_patch = view.page_patches.subspan(patch_cursor, bytes);
        patch_cursor += bytes;
    }
    context.result = compress_state(
        context.layer, layer_state.compressor,
        layer_prefix(context.layer) + "attn.compressor.", {},
        context.position, layer_state.frequencies,
        view.compressor_values, view.compressor_scores,
        compressed_append, compressed_patch);
    if (!context.result.ok()) return false;
    if (layer_state.indexer_compressor.ratio != 0U) {
        // The learned-index row is committed here, in the same stream order as
        // the sliding and compressed rows, so index_select() later performs
        // selection only and must not append it a second time.
        auto* index_append = context.index_append.has_value()
            ? &*context.index_append : nullptr;
        std::span<std::byte> index_patch;
        if (index_append != nullptr) {
            const auto bytes = static_cast<std::size_t>(
                index_append->patch_bytes());
            if (bytes > view.page_patches.size() - patch_cursor) {
                context.result.errors.emplace_back(
                    "DeepSeek deferred learned-index staging is truncated");
                return false;
            }
            index_patch = view.page_patches.subspan(patch_cursor, bytes);
            patch_cursor += bytes;
        }
        context.result = compress_state(
            context.layer, layer_state.indexer_compressor,
            layer_prefix(context.layer) + "attn.indexer.compressor.", {},
            context.position, layer_state.frequencies,
            view.index_compressor_values, view.index_compressor_scores,
            index_append, index_patch);
        if (!context.result.ok()) return false;
    } else if (!view.index_compressor_values.empty() ||
               !view.index_compressor_scores.empty() ||
               context.index_append.has_value()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred sparse-index preparation was supplied for a "
            "layer whose indexer is not admitted");
        return false;
    }
    if (patch_cursor != view.page_patches.size()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred page-patch staging has unused bytes");
        return false;
    }
    return true;
}

ValidationResult DeepSeekV4Runtime::Impl::compress_state(
    std::uint32_t layer, CompressorState& state, const std::string& prefix,
    std::span<const float> input, std::uint32_t position,
    std::span<const float> frequencies,
    std::span<const float> prepared_values,
    std::span<const float> prepared_scores,
    Dsv4KvPhysicalAppend* prepared_append,
    std::span<std::byte> prepared_patch,
    std::vector<float>* pooled_row) {
    ValidationResult result;
    if (pooled_row != nullptr) pooled_row->clear();
    if (state.ratio == 0U) return result;
    const auto dimensions = static_cast<std::size_t>(state.coefficient) *
                            state.head_dim;
    std::vector<float> values(dimensions);
    std::vector<float> scores(dimensions);
    if (!prepared_values.empty() || !prepared_scores.empty()) {
        if (prepared_values.size() != dimensions ||
            prepared_scores.size() != dimensions) {
            result.errors.emplace_back(
                "DeepSeek prepared compressor spans have incompatible sizes");
            return result;
        }
        std::copy(prepared_values.begin(), prepared_values.end(), values.begin());
        std::copy(prepared_scores.begin(), prepared_scores.end(), scores.begin());
    } else {
        const auto slot = layer_device(layer);
        result = linear(slot, prefix + "wkv", dimensions, kHidden, input,
                        values, false);
        if (!result.ok()) return result;
        result = linear(slot, prefix + "wgate", dimensions, kHidden, input,
                        scores, false);
        if (!result.ok()) return result;
    }
    auto ape = host_tensor(prefix + "ape",
                           static_cast<std::uint64_t>(state.ratio) * dimensions);
    if (!ape.ok()) {
        append_errors(result, std::move(ape.errors));
        return result;
    }
    const auto phase = position % state.ratio;
    const auto row = state.coefficient == 2U ? state.ratio + phase : phase;
    const auto row_offset = static_cast<std::size_t>(row) * dimensions;
    const auto ape_offset = static_cast<std::size_t>(phase) * dimensions;
    for (std::size_t dimension = 0U; dimension < dimensions; ++dimension) {
        state.values[row_offset + dimension] = values[dimension];
        state.scores[row_offset + dimension] =
            scores[dimension] + (*ape.value)[ape_offset + dimension];
    }
    if ((position + 1U) % state.ratio != 0U) return result;

    std::vector<float> pooled(state.head_dim, 0.0F);
    for (std::uint32_t dimension = 0U; dimension < state.head_dim; ++dimension) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_row = candidate < state.ratio ? candidate : candidate;
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(state.head_dim + dimension);
                index = static_cast<std::size_t>(source_row) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            maximum = std::max(maximum, state.scores[index]);
        }
        double denominator = 0.0;
        double numerator = 0.0;
        for (std::uint32_t candidate = 0U;
             candidate < state.coefficient * state.ratio; ++candidate) {
            std::size_t index = 0U;
            if (state.coefficient == 2U) {
                const auto source_dimension = candidate < state.ratio ? dimension :
                    static_cast<std::uint32_t>(state.head_dim + dimension);
                index = static_cast<std::size_t>(candidate) * dimensions +
                        source_dimension;
            } else {
                index = static_cast<std::size_t>(candidate) * dimensions + dimension;
            }
            const double weight = std::exp(
                static_cast<double>(state.scores[index] - maximum));
            denominator += weight;
            numerator += weight * static_cast<double>(state.values[index]);
        }
        pooled[dimension] = static_cast<float>(numerator / denominator);
    }
    if (state.coefficient == 2U) {
        const auto block_bytes = static_cast<std::size_t>(state.ratio) * dimensions;
        std::copy_n(state.values.begin() + block_bytes, block_bytes,
                    state.values.begin());
        std::copy_n(state.scores.begin() + block_bytes, block_bytes,
                    state.scores.begin());
    }
    result = norm(pooled, pooled, prefix + "norm.weight");
    if (!result.ok()) return result;
    apply_rope(std::span<float>(pooled).last(kRopeDim),
               position + 1U - state.ratio, frequencies);
    round_bf16(pooled);
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        // The live physical page encoder owns the accepted BF16-boundary,
        // power-of-two scale, and half-up E4M3 conversion.
    } else if (state.rotate_fp4) {
        result = dsv4_hadamard_rotate_f32(pooled);
        if (!result.ok()) return result;
        result = dsv4_fp4_e2m1_simulate_f32(pooled, 32U);
        if (!result.ok()) return result;
    } else {
        quantize_activation_in_place(
            std::span<float>(pooled).first(state.head_dim - kRopeDim), 64U);
    }
    const auto compressed_row = position / state.ratio;
    if (pooled_row != nullptr) {
        // The caller owns publication. Returning here keeps the accumulator
        // advance and the row encoding in one place while letting the
        // rank-local path lay the same bytes into both ranks' pages.
        *pooled_row = std::move(pooled);
        return result;
    }
    if (prepared_append != nullptr) {
        result = prepared_append->commit(pooled, prepared_patch);
        if (!result.ok()) return result;
    } else if (kv_cache != nullptr) {
        result = kv_cache->append(active_sequence, state.kind, layer,
                                  state.ratio, compressed_row, pooled);
        if (!result.ok()) return result;
    } else {
        auto destination = state.compressed.writable_row(compressed_row);
        if (destination.size() != pooled.size()) {
            result.errors.emplace_back("DeepSeek compressed cache allocation failed");
            return result;
        }
        std::copy(pooled.begin(), pooled.end(), destination.begin());
    }
    return result;
}

