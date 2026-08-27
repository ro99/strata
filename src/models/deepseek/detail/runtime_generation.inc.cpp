ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::forward_token(
    std::uint32_t token, std::uint32_t position, bool logits_required) {
    ParseResult<std::uint32_t> result;
    result.value = token;
    std::vector<float> hidden(static_cast<std::size_t>(kMhc) * kHidden);
    std::vector<float> fused_logits;
    auto validation = forward_hidden(
        token, position, hidden,
        logits_required ? &fused_logits : nullptr);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    if (!logits_required) {
        ++graph_stats.forward_tokens;
        result.value = token;
        return result;
    }

    return sample_hidden(
        token, position, hidden,
        fused_logits.empty() ? nullptr : &fused_logits);
}

// A one-token pass writes the compressor accumulators in place, may shift them
// when a block closes, and appends at most one row to each of the sliding and
// compressed caches. `position` is the position that pass will occupy, which
// fixes which single row of each cache it can reach.
SpeculativeState DeepSeekV4Runtime::Impl::capture_speculative_state(
    std::uint32_t position) const {
    SpeculativeState saved;
    saved.layers.resize(kLayers);
    saved.tokens = position;
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        const auto& state = attention_state[layer];
        auto& snapshot = saved.layers[layer];
        const auto capture = [&](const CompressorState& source,
                                 CompressorSnapshot& destination) {
            if (source.ratio == 0U) return;
            destination.values = source.values;
            destination.scores = source.scores;
            // With a block cache the compressed rows live in the KV cache and
            // are undone by truncating the sequence instead.
            if (kv_cache != nullptr) return;
            destination.compressed_index = position / source.ratio;
            const auto row = source.compressed.row(destination.compressed_index);
            if (row.empty()) return;
            destination.compressed_row.assign(row.begin(), row.end());
            destination.compressed_saved = true;
        };
        capture(state.compressor, snapshot.compressor);
        capture(state.indexer_compressor, snapshot.indexer);
        if (kv_cache == nullptr && !state.sliding.empty()) {
            snapshot.sliding_index =
                static_cast<std::size_t>(position % kWindow) * kHeadDim;
            snapshot.sliding_row.assign(
                state.sliding.begin() +
                    static_cast<std::ptrdiff_t>(snapshot.sliding_index),
                state.sliding.begin() +
                    static_cast<std::ptrdiff_t>(snapshot.sliding_index + kHeadDim));
            snapshot.sliding_saved = true;
        }
    }
    return saved;
}

ValidationResult DeepSeekV4Runtime::Impl::restore_speculative_state(
    const SpeculativeState& saved) {
    ValidationResult result;
    if (kv_cache != nullptr) {
        result = kv_cache->truncate_sequence(active_sequence, saved.tokens);
        if (!result.ok()) return result;
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        auto& state = attention_state[layer];
        const auto& snapshot = saved.layers[layer];
        const auto restore = [&](CompressorState& target,
                                 const CompressorSnapshot& source) {
            if (target.ratio == 0U) return;
            target.values = source.values;
            target.scores = source.scores;
            if (kv_cache != nullptr) return;
            if (source.compressed_saved) {
                auto row = target.compressed.writable_row(source.compressed_index);
                if (row.size() == source.compressed_row.size()) {
                    std::copy(source.compressed_row.begin(),
                              source.compressed_row.end(), row.begin());
                }
                return;
            }
            // The row was unallocated before the pass. Nothing ever reads past
            // the accepted compressed count, so zeroing an allocation the pass
            // made is observably the state it started in, and it cannot leave
            // a speculative row behind for a later read to find.
            if (target.compressed.row(source.compressed_index).empty()) return;
            auto row = target.compressed.writable_row(source.compressed_index);
            std::fill(row.begin(), row.end(), 0.0F);
        };
        restore(state.compressor, snapshot.compressor);
        restore(state.indexer_compressor, snapshot.indexer);
        if (snapshot.sliding_saved) {
            std::copy(snapshot.sliding_row.begin(), snapshot.sliding_row.end(),
                      state.sliding.begin() +
                          static_cast<std::ptrdiff_t>(snapshot.sliding_index));
        }
    }
    return result;
}

// One lookahead step per candidate: decode `c + w`, measure how open the
// resulting distribution is, then put the runtime back. Sequential rather than
// batched because the graph carries a single sequence, so each candidate costs
// a full decode step on top of the one that produced these logits.
ValidationResult DeepSeekV4Runtime::Impl::future_entropy(
    std::span<const std::uint32_t> candidates, std::uint32_t top_n,
    std::uint32_t position, std::span<double> normalized_entropy) {
    ValidationResult result;
    const auto speculative_position = position + 1U;
    if (speculative_position >= config.maximum_context_tokens) {
        result.errors.emplace_back(
            "future-entropy lookahead exceeds the configured context ceiling");
        return result;
    }
    const auto lookahead_started = std::chrono::steady_clock::now();
    const auto saved = capture_speculative_state(speculative_position);
    speculative_pass = true;
    std::vector<float> hidden(static_cast<std::size_t>(kMhc) * kHidden);
    std::vector<float> logits;
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        ++graph_stats.future_entropy_passes;
        result = forward_hidden(
            candidates[index], speculative_position, hidden,
            config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice
                ? &logits : nullptr);
        if (result.ok() &&
            config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            result = head_logits(hidden, logits);
        }
        // Restore whether or not the pass succeeded: a half-applied pass would
        // desynchronize the caches from the accepted sequence.
        auto restored = restore_speculative_state(saved);
        if (!result.ok()) break;
        if (!restored.ok()) {
            result = std::move(restored);
            break;
        }
        normalized_entropy[index] = normalized_top_n_entropy(logits, top_n);
    }
    speculative_pass = false;
    graph_stats.future_entropy_nanoseconds +=
        elapsed_nanoseconds(lookahead_started);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::head_logits(
    std::span<const float> hidden, std::vector<float>& logits) {
    ValidationResult validation;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        validation.errors.emplace_back(
            "DeepSeek output head received an invalid hidden-state shape");
        return validation;
    }

    auto head_projection = host_tensor("hc_head_fn", kMhc * kMhc * kHidden);
    auto head_scale = host_tensor("hc_head_scale", 1U);
    auto head_base = host_tensor("hc_head_base", kMhc);
    if (!head_projection.ok()) append_errors(validation, std::move(head_projection.errors));
    if (!head_scale.ok()) append_errors(validation, std::move(head_scale.errors));
    if (!head_base.ok()) append_errors(validation, std::move(head_base.errors));
    if (!validation.ok()) return validation;
    double square_sum = 0.0;
    for (const float value : hidden) square_sum += static_cast<double>(value) * value;
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum / static_cast<double>(hidden.size())) +
        kRmsEpsilon);
    std::vector<float> reduced(kHidden, 0.0F);
    for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
        double projected = 0.0;
        const auto row = static_cast<std::size_t>(copy) * hidden.size();
        for (std::size_t column = 0U; column < hidden.size(); ++column) {
            projected += static_cast<double>((*head_projection.value)[row + column]) *
                         hidden[column];
        }
        const float coefficient = sigmoid_f32(
            static_cast<float>(projected) * reciprocal * (*head_scale.value)[0] +
            (*head_base.value)[copy]) + kRmsEpsilon;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            reduced[column] += coefficient *
                hidden[static_cast<std::size_t>(copy) * kHidden + column];
        }
    }
    round_bf16(reduced);
    validation = norm(reduced, reduced, "norm.weight");
    if (!validation.ok()) return validation;
    logits.assign(kVocabulary, 0.0F);
    return linear(layer_device(kLayers - 1U), "head", kVocabulary,
                  kHidden, reduced, logits, false);
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::sample_hidden(
    std::uint32_t token, std::uint32_t position,
    std::span<const float> hidden,
    const std::vector<float>* prepared_logits) {
    ParseResult<std::uint32_t> result;
    const auto head_started = std::chrono::steady_clock::now();
    std::vector<float> logits;
    ValidationResult validation;
    if (prepared_logits == nullptr) {
        validation = head_logits(hidden, logits);
    }
    // Stop the head timer before sampling: with a lookahead in the pipeline
    // the draw is whole forward passes, and folding those into the output head
    // would report the graph's cheapest phase as its most expensive one.
    graph_stats.output_head_nanoseconds += elapsed_nanoseconds(head_started);
    if (!validation.ok()) {
        result.errors = std::move(validation.errors);
        return result;
    }
    FutureEntropyEvaluator lookahead;
    if (active_sampling.future_entropy_candidates != 0U) {
        lookahead = [this, position](std::span<const std::uint32_t> candidates,
                                     std::uint32_t top_n,
                                     std::span<double> normalized_entropy) {
            return future_entropy(candidates, top_n, position,
                                  normalized_entropy);
        };
    }
    const auto& active_logits = prepared_logits == nullptr
        ? logits : *prepared_logits;
    last_sample = sample_logits(
        active_logits, active_sampling,
        SamplingHistory{sampled_token_counts, sampled_token_ids}, sampler,
        lookahead);
    if (!last_sample.ok()) {
        result.errors = last_sample.errors;
        return result;
    }
    result.value = last_sample.token;
    ++sampled_token_counts[result.value];
    sampled_token_ids.push_back(result.value);
    if (config.enable_logit_trace) {
        record_logits(position, token, result.value, active_logits);
    }
    ++graph_stats.forward_tokens;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::flush_deferred_routes() {
    ValidationResult result;
    std::stable_sort(
        deferred_route_events.begin(), deferred_route_events.end(),
        [](const RouteEvent& left, const RouteEvent& right) {
            if (left.token_position != right.token_position) {
                return left.token_position < right.token_position;
            }
            return left.layer < right.layer;
        });
    for (const auto& event : deferred_route_events) {
        if (config.expert_prefetch_predictions != 0U) {
            route_predictor.observe(event);
        }
        if (route_trace.is_open()) {
            auto written = route_trace.write(event);
            if (!written.ok()) {
                append_errors(result, std::move(written.errors));
                break;
            }
        }
    }
    deferred_route_events.clear();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::flush_prefill_observability() {
    ValidationResult result;
    const auto position_layer_less = [](const auto& left, const auto& right) {
        if (left.position != right.position) return left.position < right.position;
        return left.layer < right.layer;
    };
    if (config.enable_layer_hash_trace) {
        std::stable_sort(diagnostics.layer_hashes.begin(),
                         diagnostics.layer_hashes.end(), position_layer_less);
        diagnostics.layer_hash_trace_hash = diagnostic_hash_u32(
            kDiagnosticFnvOffset, kLayers);
        for (const auto& record : diagnostics.layer_hashes) {
            auto aggregate = diagnostics.layer_hash_trace_hash;
            aggregate = diagnostic_hash_u32(aggregate, record.position);
            aggregate = diagnostic_hash_u32(aggregate, record.input_token);
            aggregate = diagnostic_hash_u32(aggregate, record.layer);
            diagnostics.layer_hash_trace_hash = diagnostic_hash_u64(
                aggregate, record.bf16_hash);
        }
        std::stable_sort(diagnostics.operation_hashes.begin(),
                         diagnostics.operation_hashes.end(),
                         position_layer_less);
    }
    return flush_deferred_routes();
}

ParseResult<std::uint32_t> DeepSeekV4Runtime::Impl::forward_prefill(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base) {
    ParseResult<std::uint32_t> result;
    if (tokens.empty() || tokens.size() > config.maximum_context_tokens ||
        position_base > config.maximum_context_tokens - tokens.size()) {
        result.errors.emplace_back(
            "DeepSeek prefill requires a non-empty in-context token range");
        return result;
    }
    const auto hidden_stride = static_cast<std::size_t>(kMhc) * kHidden;
    const bool physical =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    if (physical && config.prefill_page_tokens > 1U && tokens.size() > 1U) {
        // Page-major over every prompt row but the last. The last row keeps
        // the token-major path because it is the only one that may need
        // logits, and the fused output head is defined only there.
        defer_prefill_observability = true;
        const auto paged_tokens = tokens.size() - 1U;
        const auto fail = [&](ValidationResult&& status) {
            defer_prefill_observability = false;
            result.errors = std::move(status.errors);
            return result;
        };
        for (std::size_t page_begin = 0U; page_begin < paged_tokens;
             page_begin += config.prefill_page_tokens) {
            const auto page_rows = std::min<std::size_t>(
                config.prefill_page_tokens, paged_tokens - page_begin);
            std::vector<float> hidden(page_rows * hidden_stride);
            for (std::size_t row = 0U; row < page_rows; ++row) {
                const auto embedding_started = std::chrono::steady_clock::now();
                auto embedded = embed(
                    tokens[page_begin + row],
                    std::span<float>(hidden).subspan(
                        row * hidden_stride, hidden_stride));
                graph_stats.embedding_nanoseconds +=
                    elapsed_nanoseconds(embedding_started);
                if (!embedded.ok()) return fail(std::move(embedded));
            }
            ++graph_stats.prefill_pages;
            graph_stats.prefill_max_page_tokens = std::max<std::uint64_t>(
                graph_stats.prefill_max_page_tokens, page_rows);
            graph_stats.prefill_max_workspace_bytes =
                std::max<std::uint64_t>(
                    graph_stats.prefill_max_workspace_bytes,
                    static_cast<std::uint64_t>(page_rows) * hidden_stride *
                        sizeof(float));
            auto executed = device_mhc_forward_prefill_page(
                tokens.subspan(page_begin, page_rows),
                position_base + static_cast<std::uint32_t>(page_begin),
                hidden);
            if (!executed.ok()) return fail(std::move(executed));
        }
        defer_prefill_observability = false;
        auto flushed = flush_prefill_observability();
        if (!flushed.ok()) {
            result.errors = std::move(flushed.errors);
            return result;
        }
        ++graph_stats.prefill_pages;
        graph_stats.prefill_max_page_tokens = std::max<std::uint64_t>(
            graph_stats.prefill_max_page_tokens, 1U);
        return forward_token(
            tokens.back(),
            position_base + static_cast<std::uint32_t>(tokens.size() - 1U),
            true);
    }
    if (config.prefill_page_tokens == 1U || physical) {
        for (std::size_t position = 0U; position < tokens.size(); ++position) {
            ++graph_stats.prefill_pages;
            graph_stats.prefill_max_page_tokens = 1U;
            graph_stats.prefill_max_workspace_bytes = std::max<std::uint64_t>(
                graph_stats.prefill_max_workspace_bytes,
                static_cast<std::uint64_t>(hidden_stride) * sizeof(float));
            result = forward_token(
                tokens[position],
                position_base + static_cast<std::uint32_t>(position),
                position + 1U == tokens.size());
            if (!result.ok()) return result;
        }
        return result;
    }
    defer_prefill_observability = true;
    // Expert residency, not activation memory, is what bounds prefill. One
    // layer's 256 routed experts are 3.4 GB and fit the VRAM cache; all 43
    // layers together are 147 GB and do not. Sweeping every layer inside the
    // page loop therefore evicts the cache once per page: a 3,565-token
    // prefill moved 3,367 GB of demand H2D, 22.9x the 147 GB it must move,
    // with 745,172 evictions and 59% of the phase spent in demand wait.
    // Visiting layers outermost over a tile of pages touches each layer's
    // experts once per tile instead of once per page. A tile equal to
    // prefill_page_tokens reproduces the page-major nest exactly.
    const auto tile_tokens = config.prefill_layer_tile_tokens == 0U
        ? tokens.size()
        : std::min<std::size_t>(config.prefill_layer_tile_tokens, tokens.size());
    for (std::size_t tile_begin = 0U; tile_begin < tokens.size();
         tile_begin += tile_tokens) {
        const auto tile_rows =
            std::min<std::size_t>(tile_tokens, tokens.size() - tile_begin);
        std::vector<float> hidden(tile_rows * hidden_stride);

        for (std::size_t row = 0U; row < tile_rows; ++row) {
            const auto embedding_started = std::chrono::steady_clock::now();
            auto status = embed(tokens[tile_begin + row],
                                std::span<float>(hidden).subspan(
                                    row * hidden_stride, hidden_stride));
            graph_stats.embedding_nanoseconds +=
                elapsed_nanoseconds(embedding_started);
            if (!status.ok()) {
                defer_prefill_observability = false;
                result.errors = std::move(status.errors);
                return result;
            }
        }

        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            for (std::size_t page_begin = 0U; page_begin < tile_rows;
                 page_begin += config.prefill_page_tokens) {
                const auto page_rows = static_cast<std::uint32_t>(
                    std::min<std::size_t>(config.prefill_page_tokens,
                                          tile_rows - page_begin));
                if (layer == 0U) {
                    ++graph_stats.prefill_pages;
                    graph_stats.prefill_max_page_tokens =
                        std::max<std::uint64_t>(
                            graph_stats.prefill_max_page_tokens, page_rows);
                    graph_stats.prefill_max_workspace_bytes =
                        std::max<std::uint64_t>(
                            graph_stats.prefill_max_workspace_bytes,
                            static_cast<std::uint64_t>(tile_rows) *
                                    hidden_stride * sizeof(float) +
                                static_cast<std::uint64_t>(page_rows) *
                                    (2U * hidden_stride + 2U * kHidden +
                                     kQueryRank +
                                     static_cast<std::size_t>(kHeads) *
                                         kHeadDim +
                                     kHeadDim) *
                                    sizeof(float));
                }
                const auto absolute_page_begin =
                    position_base +
                    static_cast<std::uint32_t>(tile_begin + page_begin);
                auto status = block_page(
                    layer, tokens.subspan(tile_begin + page_begin, page_rows),
                    std::span<float>(hidden).subspan(
                        page_begin * hidden_stride, page_rows * hidden_stride),
                    absolute_page_begin);
                if (!status.ok()) {
                    defer_prefill_observability = false;
                    result.errors = std::move(status.errors);
                    return result;
                }
                for (std::uint32_t row = 0U; row < page_rows; ++row) {
                    const auto token_index = tile_begin + page_begin + row;
                    if (config.enable_layer_hash_trace) {
                        record_layer_hash(
                            position_base +
                                static_cast<std::uint32_t>(token_index),
                            tokens[token_index], layer,
                            std::span<const float>(hidden).subspan(
                                (page_begin + row) * hidden_stride,
                                hidden_stride));
                    }
                }
            }
        }
        auto routes_flushed = flush_deferred_routes();
        if (!routes_flushed.ok()) {
            defer_prefill_observability = false;
            result.errors = std::move(routes_flushed.errors);
            return result;
        }
        graph_stats.forward_tokens += tile_rows;
        if (tile_begin + tile_rows == tokens.size()) {
            --graph_stats.forward_tokens;
            const auto last_row = std::span<const float>(hidden).last(hidden_stride);
            result = sample_hidden(tokens.back(),
                                   position_base + static_cast<std::uint32_t>(
                                       tokens.size() - 1U),
                                   last_row);
            if (!result.ok()) {
                defer_prefill_observability = false;
                return result;
            }
        }
    }
    defer_prefill_observability = false;
    auto flushed = flush_prefill_observability();
    if (!flushed.ok()) result.errors = std::move(flushed.errors);
    return result;
}

