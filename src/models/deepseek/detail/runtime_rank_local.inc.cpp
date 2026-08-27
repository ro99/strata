ValidationResult DeepSeekV4Runtime::Impl::forward_hidden(
    std::uint32_t token, std::uint32_t position, std::span<float> hidden,
    std::vector<float>* fused_logits) {
    const auto embedding_started = std::chrono::steady_clock::now();
    auto result = embed(token, hidden);
    graph_stats.embedding_nanoseconds += elapsed_nanoseconds(embedding_started);
    if (!result.ok()) return result;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        if (rank_local_active && position >= active_prompt_tokens) {
            // Decode, and rank-local was admitted. Prefill stays centralized:
            // the rank-local set is a decode-shaped ownership of the weights,
            // and admission accounts for both being resident.
            return rank_local_forward_hidden(
                token, position, hidden, fused_logits);
        }
        return device_mhc_forward_hidden(
            token, position, hidden, fused_logits);
    }
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        result = block(layer, token, hidden, position);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_layer_hash(position, token, layer, hidden);
        }
    }
    return result;
}

bool DeepSeekV4Runtime::Impl::device_head_callback(
    void* opaque, std::span<const std::uint16_t> encoded_hidden,
    std::span<float> reduced) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<DeviceHeadContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->execute_device_head_callback(
        context, encoded_hidden, reduced);
}

bool DeepSeekV4Runtime::Impl::execute_device_head_callback(
    DeviceHeadContext& context,
    std::span<const std::uint16_t> encoded_hidden,
    std::span<float> reduced) {
    context.invoked = true;
    context.result = {};
    const auto projection = host_tensors.find("hc_head_fn");
    const auto scale = host_tensors.find("hc_head_scale");
    const auto base = host_tensors.find("hc_head_base");
    const auto norm_weight = host_tensors.find("norm.weight");
    if (encoded_hidden.size() != context.hidden.size() ||
        reduced.size() != kHidden || projection == host_tensors.end() ||
        scale == host_tensors.end() || base == host_tensors.end() ||
        norm_weight == host_tensors.end()) {
        context.result.errors.emplace_back(
            "DeepSeek deferred output-head inputs are unavailable");
        return false;
    }
    for (std::size_t index = 0U; index < context.hidden.size(); ++index) {
        context.hidden[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
        if (!std::isfinite(context.hidden[index])) {
            context.result.errors.emplace_back(
                "DeepSeek deferred output-head hidden state is non-finite");
            return false;
        }
    }
    double square_sum = 0.0;
    for (const float value : context.hidden) {
        square_sum += static_cast<double>(value) * value;
    }
    const float reciprocal = 1.0F / std::sqrt(
        static_cast<float>(square_sum /
                           static_cast<double>(context.hidden.size())) +
        kRmsEpsilon);
    std::fill(reduced.begin(), reduced.end(), 0.0F);
    for (std::uint32_t copy = 0U; copy < kMhc; ++copy) {
        double projected = 0.0;
        const auto row = static_cast<std::size_t>(copy) *
                         context.hidden.size();
        for (std::size_t column = 0U; column < context.hidden.size(); ++column) {
            projected += static_cast<double>(
                projection->second[row + column]) * context.hidden[column];
        }
        const float coefficient = sigmoid_f32(
            static_cast<float>(projected) * reciprocal * scale->second[0] +
            base->second[copy]) + kRmsEpsilon;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            reduced[column] += coefficient * context.hidden[
                static_cast<std::size_t>(copy) * kHidden + column];
        }
    }
    round_bf16(reduced);
    context.result = rms_norm_f32(
        reduced, reduced, norm_weight->second, kRmsEpsilon);
    if (context.result.ok()) round_bf16(reduced);
    return context.result.ok();
}

ValidationResult DeepSeekV4Runtime::Impl::admit_rank_local() {
    ValidationResult result;
    rank_local_active = false;
    if (checkpoint == nullptr || weights == nullptr || kv_cache == nullptr) {
        result.errors.emplace_back(
            "rank-local decode requires a loaded checkpoint, weight arena and "
            "physical KV cache");
        return result;
    }
    std::array<int, kDsv4RankLocalWorld> rank_devices{};
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        rank_devices[rank] = devices[rank];
    }

    // Load first: admission needs the sharded set's measured size, and a
    // rejection after loading is still fail-closed because the store is
    // cleared before returning.
    auto store = std::make_unique<Dsv4RankLocalWeightStore>();
    const auto checkpoint_before = checkpoint->stats();
    const auto cuda_before = cuda.stats();
    result = store->load(*checkpoint, cuda, rank_devices, kLayers);
    if (!result.ok()) return result;
    if (config.verbose) {
        const auto stats = store->stats();
        const auto checkpoint_after = checkpoint->stats();
        const auto cuda_after = cuda.stats();
        std::uint64_t cuda_copy_nanoseconds = 0U;
        std::uint64_t cuda_wait_nanoseconds = 0U;
        std::uint64_t cuda_allocation_nanoseconds = 0U;
        for (std::size_t slot = 0U; slot < cuda_after.devices.size(); ++slot) {
            const auto& after = cuda_after.devices[slot];
            const auto before = slot < cuda_before.devices.size()
                ? cuda_before.devices[slot] : CudaBackendStats::Device{};
            cuda_copy_nanoseconds += after.weight_copy_nanoseconds -
                                     before.weight_copy_nanoseconds;
            cuda_wait_nanoseconds += after.upload_wait_nanoseconds -
                                     before.upload_wait_nanoseconds;
            cuda_allocation_nanoseconds += after.weight_allocation_nanoseconds -
                                           before.weight_allocation_nanoseconds;
        }
        const auto read_nanoseconds = checkpoint_after.nanoseconds -
                                      checkpoint_before.nanoseconds;
        const auto accounted_seconds = static_cast<double>(
            read_nanoseconds + cuda_copy_nanoseconds + cuda_wait_nanoseconds +
            cuda_allocation_nanoseconds) / 1.0e9;
        const auto cpu_other_seconds = std::max(
            0.0, stats.seconds - accounted_seconds);
        const auto read_gib_s = stats.seconds == 0.0
            ? 0.0
            : static_cast<double>(stats.checkpoint_read_bytes) /
                  stats.seconds / static_cast<double>(1ULL << 30U);
        std::cerr << "[deepseek-load] phase=rank_local_weights elapsed_ms="
                  << stats.seconds * 1000.0
                  << " checkpoint_read_calls=" << stats.checkpoint_read_calls
                  << " checkpoint_read_bytes=" << stats.checkpoint_read_bytes
                  << " checkpoint_read_gib_s=" << read_gib_s
                  << " checkpoint_read_ms="
                  << static_cast<double>(read_nanoseconds) / 1.0e6
                  << " cuda_copy_ms="
                  << static_cast<double>(cuda_copy_nanoseconds) / 1.0e6
                  << " cuda_wait_ms="
                  << static_cast<double>(cuda_wait_nanoseconds) / 1.0e6
                  << " cuda_allocation_ms="
                  << static_cast<double>(cuda_allocation_nanoseconds) / 1.0e6
                  << " cpu_other_ms=" << cpu_other_seconds * 1000.0
                  << " rank0_device_bytes=" << stats.device_weight_bytes[0]
                  << " rank1_device_bytes=" << stats.device_weight_bytes[1]
                  << '\n';
    }
    const auto sharded = store->device_bytes();

    Dsv4RankLocalAdmissionRequest request;
    request.devices.assign(rank_devices.begin(), rank_devices.end());
    request.kv_cache_mode = config.kv_cache_mode;
    request.supported_checkpoint = true;
    request.fp4_routed_experts = config.enable_device_moe;
    request.layer_count = kLayers;
    // Admission is a setup-time check, so it uses the configured maximum for
    // both: the prompt length is not known until generate() runs, and a plan
    // that only fits the current prompt is not a plan.
    request.active_context_tokens = config.maximum_context_tokens;
    request.maximum_context_tokens = config.maximum_context_tokens;
#if defined(STRATA_HAS_NCCL)
    request.nccl_available = true;
#else
    request.nccl_available = false;
#endif
    // Dsv4MemoryPlan's spine and expert-cache figures are aggregates across
    // every device -- admission compares their sum against the aggregate VRAM
    // budget -- while the rank-local ceiling is per device. Charging each rank
    // the aggregate would double-count both on a two-device topology and
    // reject a configuration that fits. The weight cache measures the real
    // per-slot split, so take it from there rather than dividing by the
    // device count: 43 layers over two slots is 22 and 21, not 21.5.
    const auto cache = weights->stats();
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        auto& device = request.device[rank];
        device.initial_device_usage_bytes =
            rank_local_initial_device_vram_bytes[rank];
        device.rank_local_weight_bytes = sharded[rank];
        device.centralized_spine_bytes =
            rank < cache.pinned_bytes.size() ? cache.pinned_bytes[rank] : 0U;
        if (rank == mhc_slot) {
            device.centralized_spine_bytes += memory.mhc_device_bytes;
        }
        device.workspace_bytes = kDeviceWorkspaceReserve;
        device.kv_capacity_bytes =
            rank < memory.per_device_kv_cache_bytes.size()
                ? memory.per_device_kv_cache_bytes[rank] : 0U;
        device.nccl_buffer_bytes = 64ULL << 20U;
        device.head_buffer_bytes = 16ULL << 20U;
        // The rank-local store and centralized cache share one fixed CUDA
        // weight arena. Admission must cap the cache against the suballocation
        // space left after the store, as well as against the overall program
        // ceiling; otherwise the logical cache capacity can promise bytes the
        // arena can never allocate.
        const auto cache_pinned =
            rank < cache.pinned_bytes.size() ? cache.pinned_bytes[rank] : 0U;
        if (rank >= capacities.size() ||
            sharded[rank] > capacities[rank] ||
            cache_pinned > capacities[rank] - sharded[rank]) {
            result.errors.emplace_back(
                "rank-local CUDA device " + std::to_string(devices[rank]) +
                " weight arena cannot retain the rank-local store beside "
                "the centralized prefill spine");
            continue;
        }
        // The routed-expert tier is permanent and suballocates from this same
        // arena, so its bytes are reserved before the prefill cache is sized.
        // Without this the cache is promised space the tier already holds and
        // fails an acquire mid-prefill rather than simply being smaller.
        const auto tier_reserved =
            config.static_expert_plan_path.empty()
                ? 0U : config.static_expert_tier_bytes;
        // The shared expert is acquired from this same cache once per layer on
        // every forward pass -- experiment 0163 measured it as the only
        // per-token CUDA dispatch decode makes -- so its whole set has to stay
        // stageable no matter how many routed experts the cache has admitted.
        // Routed entries must therefore not be allowed to claim it.
        //
        // Without this the cache's logical capacity equals the arena's free
        // space exactly, which is only survivable while something else is the
        // binding minimum. A routed-expert tier makes `arena_expert_bytes` the
        // binding term, the slack goes to zero, and the next shared-expert
        // re-stage fails: experiment 0178 measured `layers.19.ffn.
        // shared_experts.w2` wanting 8.0 MiB against 20.0 MiB free of
        // 20504.2 MiB, in three blocks whose largest was 6.8 MiB. That is
        // exhaustion, not fragmentation, and it bricks the server -- every
        // later request returns a sticky mHC ordering error.
        constexpr auto shared_expert_reserve = []() constexpr {
            const std::uint64_t gate_up =
                2ULL * static_cast<std::uint64_t>(kExpertIntermediate) * kHidden;
            const std::uint64_t down =
                static_cast<std::uint64_t>(kHidden) * kExpertIntermediate;
            // One E8M0 scale per 128-element block, as the checkpoint declares.
            const std::uint64_t payload = gate_up + down;
            return static_cast<std::uint64_t>(kLayers) *
                   (payload + payload / 128ULL);
        }();
        const auto reserved_total =
            sharded[rank] + cache_pinned + tier_reserved + shared_expert_reserve;
        const auto arena_after_tier =
            capacities[rank] > reserved_total
                ? capacities[rank] - reserved_total
                : 0U;
        const auto arena_expert_bytes =
            static_cast<std::uint64_t>(arena_after_tier);
        const auto cache_expert_bytes =
            rank < cache.capacity_bytes.size() &&
                    cache.capacity_bytes[rank] > cache_pinned
                ? cache.capacity_bytes[rank] - cache_pinned
                : 0U;
        device.expert_cache_bytes =
            std::min(arena_expert_bytes, cache_expert_bytes);
    }
    if (!result.ok()) {
        store->clear();
        return result;
    }
    request.host.routed_cpu_storage_bytes = memory.routed_expert_host_bytes;
    request.host.host_parameter_bytes = memory.host_parameter_bytes;
    request.host.kv_state_bytes = memory.kv_state_bytes;
    request.host.host_workspace_bytes = memory.host_workspace_bytes;

    // The cards' real capacities, so the ceiling is a fraction of this
    // machine rather than of the one the constant was measured on.
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto memory = CudaBackend::device_memory(devices[rank]);
        if (memory.ok()) {
            request.device[rank].device_total_bytes = memory.value.total_bytes;
        }
    }
    auto admitted = admit_dsv4_rank_local(request, NumaTopology::detect());
    if (!admitted.ok()) {
        store->clear();
        result.errors = std::move(admitted.errors);
        return result;
    }
    result = weights->cap_expert_capacity(
        admitted.expert_cache_capacity_bytes);
    if (!result.ok()) {
        store->clear();
        return result;
    }

#if defined(STRATA_HAS_NCCL)
    auto executor = std::make_unique<Dsv4RankLocalLayerExecutor>(cuda);
    Dsv4RankLocalLayerOptions options;
    options.devices = rank_devices;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        options.rank_cpus[rank] = admitted.rank_cpus[rank];
    }
    options.resident = &resident;

    // Routed-expert tiers on the rank devices themselves. The layer is already
    // executing there, so the experts cost 0.128 ms each against the host
    // path's 0.282, and nothing crosses a device boundary. Each rank's tier
    // takes a disjoint slice of one ranking, so the two cards split the hottest
    // experts rather than both holding the same ones.
    if (!config.static_expert_plan_path.empty()) {
        auto plan = Dsv4ExpertResidencyPlan::load(
            config.static_expert_plan_path, kLayers, kExperts);
        if (!plan.ok()) {
            store->clear();
            append_errors(result, std::move(plan.errors),
                          "expert residency plan");
            return result;
        }
        static_expert_tiers.clear();
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            auto tier = std::make_unique<Dsv4StaticExpertTier>();
            auto copy = plan.value;
            auto prepared = tier->initialize(
                rank_devices[rank], cuda, *checkpoint, std::move(copy),
                config.static_expert_tier_bytes, rank, kDsv4RankLocalWorld);
            if (!prepared.ok()) {
                store->clear();
                static_expert_tiers.clear();
                append_errors(result, std::move(prepared.errors),
                              "expert tier rank " + std::to_string(rank));
                return result;
            }
            options.static_expert_tiers[rank] = tier.get();
            static_expert_tiers.push_back(std::move(tier));
        }
    }
    result = executor->initialize(options);
    if (!result.ok()) {
        store->clear();
        return result;
    }
    rank_local_actual_device_vram_bytes =
        device_vram_used_bytes(devices);
    if (rank_local_actual_device_vram_bytes.size() !=
        kDsv4RankLocalWorld) {
        store->clear();
        result.errors.emplace_back(
            "rank-local actual VRAM ledger does not match devices");
        return result;
    }
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto ceiling = rank_local_vram_ceiling(devices[rank]);
        if (rank_local_actual_device_vram_bytes[rank] > ceiling) {
            store->clear();
            result.errors.emplace_back(
                "rank-local CUDA device " +
                std::to_string(devices[rank]) + " uses " +
                std::to_string(rank_local_actual_device_vram_bytes[rank]) +
                " B after setup, above the " +
                std::to_string(ceiling) + " B program ceiling");
        }
    }
    if (!result.ok()) return result;
    rank_local_executor = std::move(executor);
#else
    store->clear();
    result.errors.emplace_back(
        "rank-local decode requires an NCCL-enabled build");
    return result;
#endif

    rank_local_weights = std::move(store);
    rank_local_admission = std::move(admitted);
    rank_local_active = true;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_forward_hidden(
    std::uint32_t token, std::uint32_t position, std::span<float> hidden,
    std::vector<float>* fused_logits) {
    ValidationResult result;
    static_cast<void>(token);
    static_cast<void>(position);
    static_cast<void>(hidden);
    static_cast<void>(fused_logits);
#if defined(STRATA_HAS_NCCL)
    const bool session = rank_local_active &&
                         rank_local_executor != nullptr &&
                         rank_local_weights != nullptr;
#else
    const bool session = false;
#endif
    if (!session) {
        result.errors.emplace_back(
            "rank-local decode was entered without an admitted session");
        return result;
    }
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back(
            "rank-local decode hidden state has the wrong shape");
        return result;
    }
    if (kv_cache == nullptr || devices.size() < kDsv4RankLocalWorld) {
        result.errors.emplace_back(
            "rank-local decode requires a physical KV cache on two devices");
        return result;
    }
#if defined(STRATA_HAS_NCCL)
    // The token is a transaction over the sequence: every layer's KV rows are
    // reserved and encoded here, and none of them is accounted until the
    // terminal head has produced output on both ranks. The destructor aborts,
    // so an early return truncates the sequence back to `position`.
    Dsv4RankLocalKvTransaction transaction(
        *kv_cache, active_sequence, {0U, 1U}, position);

    // Seed the mHC state on both ranks before any layer runs. The executor
    // takes dsv4_mhc_device_view per rank and refuses to execute against a
    // closed state, so this is a precondition, not an optimization.
    //
    // Three properties, all taken from seed_m3_layer0 at
    // a31ac58:apps/strata_dsv4_rank_local_layer.cu:2875 rather than inferred:
    //   - once per rank, each on its own device
    //   - always layer 0's *attention* mHC; later layers arrive through the
    //     per-layer call's next_attention_mhc, and the terminal layer passes
    //     nullptr for it
    //   - mHC weights are replicated per rank, never sharded, because
    //     dsv4_mhc_begin rejects a weight whose device is not the target
    //
    // It is once per token rather than once per session: the terminal
    // finish_chain closes the state machine, and any layer failure aborts the
    // branch, so each token re-seeds.
    const auto seed = rank_local_weights->layer_view(0U, token);
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto* attention_mhc = seed.rank[rank].attention_mhc;
        if (attention_mhc == nullptr) {
            result.errors.emplace_back(
                "rank-local layer 0 attention mHC weights are unavailable for "
                "rank " + std::to_string(rank));
            return result;
        }
        result = cuda.dsv4_mhc_begin_device(
            devices[rank], *attention_mhc, hidden);
        if (!result.ok()) return result;
    }
    // From here every exit must close the mHC branches it opened and roll the
    // token back. The transaction's destructor truncates the sequence, so an
    // early return can never leave a half-appended position visible.
    const auto close_branches = [&] {
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            static_cast<void>(cuda.dsv4_mhc_abort_branch(devices[rank]));
        }
    };
    rank_local_attention_input.clear();

    // An indexed layer used to force the whole token onto the sequential
    // driver, because selection needed the query rank on the host and the only
    // host node available is inside a CUDA callback, where a CUDA call is not
    // permitted. With projection, scoring, selection and candidate resolution
    // all enqueued in the layer's own command sequence, that is no longer true
    // of an indexed layer that actually selects.
    //
    // One narrow band remains sequential: an admitted indexer whose compressed
    // history has not yet passed its own top-k, where every compressed row is
    // attended and no selection runs at all. It is reachable only between an
    // admitted context above 2,048 tokens and a decode position below 2,052,
    // and it is left on the path that has been measured rather than moved onto
    // one that has not.
    const bool queued = std::none_of(
        attention_state.begin(), attention_state.end(),
        [position](const AttentionState& state) {
            const auto ratio = state.indexer_compressor.ratio;
            return ratio != 0U && (position + 1U) / ratio <= kIndexTopK;
        });
    if (queued) {
        // Finish every host-side reservation and cache lookup before a CUDA host
        // node can start. Besides keeping all borrowed views stable, this avoids
        // concurrent access to the host-tensor map from submission and callback
        // threads.
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            result = rank_local_prepare_layer(
                layer, token, position, transaction,
                rank_local_scratch[layer], rank_local_calls[layer]);
            if (!result.ok()) {
                close_branches();
                return result;
            }
        }
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            const auto enqueue_started = std::chrono::steady_clock::now();
            auto queued = rank_local_executor->enqueue_chain_layer(
                rank_local_calls[layer]);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(enqueue_started);
            if (!queued.ok()) {
                append_errors(result, std::move(queued.errors));
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
        }
    }
    if (!queued) {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            auto& scratch = rank_local_scratch[layer];
            auto& call = rank_local_calls[layer];
            result = rank_local_prepare_layer(
                layer, token, position, transaction, scratch, call);
            if (!result.ok()) {
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
            if (layer + 1U < kLayers) {
                Dsv4RankLocalLayerResult layer_result;
                const auto layer_started = std::chrono::steady_clock::now();
                auto ran = rank_local_executor->run(
                    call, Dsv4RankLocalFailure::None, layer_result);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(layer_started);
            graph_stats.rank_local_device_nanoseconds +=
                static_cast<std::uint64_t>(layer_result.timing.total_ms * 1.0e6);
            // total_ms is wall around the whole layer and already contains the
            // boundary, so this is a component of it, not a separate term.
            graph_stats.rank_local_boundary_nanoseconds +=
                static_cast<std::uint64_t>(
                    layer_result.timing.diagnostic_boundary_ms * 1.0e6);
            graph_stats.rank_local_collective_nanoseconds +=
                static_cast<std::uint64_t>(
                    (layer_result.timing.attention_collective_ms +
                     layer_result.timing.attention_publication_ms +
                     layer_result.timing.moe_publication_ms) * 1.0e6);
            graph_stats.rank_local_transition_nanoseconds +=
                static_cast<std::uint64_t>(
                    (layer_result.timing.transition_router_ms +
                     layer_result.timing.final_transition_ms) * 1.0e6);
            graph_stats.rank_local_shared_nanoseconds +=
                static_cast<std::uint64_t>(
                    std::max(layer_result.timing.shared_gpu_rank0_ms,
                             layer_result.timing.shared_gpu_rank1_ms) * 1.0e6);
            if (!ran.ok() || !layer_result.success ||
                layer_result.global_attention_status != 0U ||
                layer_result.global_moe_status != 0U) {
                append_errors(result, std::move(
                    ran.errors.empty() ? layer_result.errors : ran.errors));
                result.errors.emplace_back(
                    "rank-local decode failed at layer " +
                    std::to_string(layer));
                close_branches();
                return result;
            }
            // Both ranks publish the same reduction. A divergence here is a
            // replica fault, not a rounding difference, and the next layer's
            // selection would silently use one rank's state for both.
            if (!std::equal(layer_result.next_attention_input[0].begin(),
                            layer_result.next_attention_input[0].end(),
                            layer_result.next_attention_input[1].begin())) {
                result.errors.emplace_back(
                    "rank-local attention input diverged between ranks at "
                    "layer " + std::to_string(layer));
                close_branches();
                return result;
            }
            rank_local_attention_input.assign(
                layer_result.next_attention_input[0].begin(),
                layer_result.next_attention_input[0].end());
            graph_stats.attention_nanoseconds += static_cast<std::uint64_t>(
                layer_result.timing.attention_ms * 1.0e6);
            graph_stats.moe_nanoseconds += static_cast<std::uint64_t>(
                (std::max(layer_result.timing.cpu_routed_rank0_ms,
                          layer_result.timing.cpu_routed_rank1_ms) +
                 layer_result.timing.moe_collective_ms) * 1.0e6);
                continue;
            }
            // Terminal layer. The executor refuses to run() it: the output
            // head consumes the mHC state, so the last layer is queued and
            // drained by finish_chain, exactly as run_m3_sequential does at
            // a31ac58:apps/strata_dsv4_rank_local_layer.cu:3038.
            const auto terminal_started = std::chrono::steady_clock::now();
            auto queued = rank_local_executor->enqueue_chain_layer(call);
            graph_stats.rank_local_layer_nanoseconds +=
                elapsed_nanoseconds(terminal_started);
            if (!queued.ok()) {
                append_errors(result, std::move(queued.errors));
                static_cast<void>(rank_local_executor->abort_chain());
                close_branches();
                return result;
            }
        }
    }

    Dsv4RankLocalHeadRequest head_request;
    const bool fuse_head = fused_logits != nullptr;
    if (fuse_head) {
        static_assert(kDsv4RankLocalVocabulary == kVocabulary,
                      "rank-local head vocabulary must match the contract");
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            rank_local_head[rank] = {};
            rank_local_head[rank].owner = this;
            rank_local_local_logits[rank].assign(
                kDsv4RankLocalVocabularyShard, 0.0F);
            rank_local_published_logits[rank].assign(
                kDsv4RankLocalVocabulary, 0U);
            head_request.heads[rank] =
                &rank_local_weights->head().weights[rank];
            head_request.callbacks[rank] = device_head_callback;
            head_request.callback_contexts[rank] = &rank_local_head[rank];
            head_request.local_logits[rank] = rank_local_local_logits[rank];
            head_request.published_logits[rank] =
                rank_local_published_logits[rank];
        }
    }
    Dsv4RankLocalLayerChainResult chain;
    const auto finish_started = std::chrono::steady_clock::now();
    auto finished = rank_local_executor->finish_chain(
        &chain, fuse_head ? &head_request : nullptr);
    graph_stats.rank_local_layer_nanoseconds +=
        elapsed_nanoseconds(finish_started);
    const auto expected_chain_count = queued
        ? static_cast<std::size_t>(kLayers) : 1U;
    if (!finished.ok() || chain.chain_count != expected_chain_count ||
        !chain.terminal) {
        append_errors(result, std::move(finished.errors));
        if (queued) {
            for (const auto& contexts : rank_local_page_contexts) {
                for (const auto& context : contexts) {
                    append_errors(result, context.result.errors,
                                  "rank-local page callback layer " +
                                      std::to_string(context.layer) +
                                      " rank " +
                                      std::to_string(context.rank));
                }
            }
        }
        if (result.ok()) {
            result.errors.emplace_back(
                "rank-local terminal layer did not complete");
        }
        close_branches();
        return result;
    }
    if (queued) {
        for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
            for (std::size_t rank = 0U; rank < 1U; ++rank) {
                const auto& context = rank_local_page_contexts[layer][rank];
                if (!context.invoked || !context.result.ok()) {
                    append_errors(result, context.result.errors,
                                  "rank-local page callback layer " +
                                      std::to_string(context.layer) +
                                      " rank " +
                                      std::to_string(context.rank));
                    if (context.result.ok()) {
                        result.errors.emplace_back(
                            "rank-local page callback was not invoked at layer " +
                            std::to_string(layer) + " rank " +
                            std::to_string(rank));
                    }
                }
            }
        }
        if (!result.ok()) {
            close_branches();
            return result;
        }
        std::uint64_t page_callback_nanoseconds = 0U;
        for (const auto& contexts : rank_local_page_contexts) {
            page_callback_nanoseconds += contexts[0].elapsed_nanoseconds;
        }
        graph_stats.rank_local_kv_nanoseconds += page_callback_nanoseconds;
    }
    {
        // Attribute the phases on the same rank the MoE term is taken from,
        // so the phase sum and the total describe one rank's critical path
        // rather than a mixture of both.
        const auto slower =
            chain.cpu_moe_phases[0].total_nanoseconds >=
                    chain.cpu_moe_phases[1].total_nanoseconds
                ? 0U : 1U;
        const auto& phases = chain.cpu_moe_phases[slower];
        graph_stats.moe_nanoseconds += phases.total_nanoseconds;
        graph_stats.rank_local_moe_gate_up_nanoseconds +=
            phases.gate_up_nanoseconds;
        graph_stats.rank_local_moe_down_nanoseconds += phases.down_nanoseconds;
        graph_stats.rank_local_moe_reduce_nanoseconds +=
            phases.reduce_nanoseconds;
    }
    if (fuse_head && (!rank_local_head[0].invoked ||
                      !rank_local_head[1].invoked)) {
        result.errors.emplace_back(
            "rank-local output-head host callback was not invoked on both "
            "ranks");
        return result;
    }
    if (!std::equal(chain.final_hidden[0].begin(), chain.final_hidden[0].end(),
                    chain.final_hidden[1].begin())) {
        result.errors.emplace_back(
            "rank-local terminal hidden state diverged between ranks");
        return result;
    }
    std::copy(chain.final_hidden[0].begin(), chain.final_hidden[0].end(),
              hidden.begin());
    if (fuse_head) {
        fused_logits->assign(kVocabulary, 0.0F);
        for (std::size_t index = 0U; index < kVocabulary; ++index) {
            (*fused_logits)[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(
                    rank_local_published_logits[0][index]) << 16U);
        }
        graph_stats.output_head_nanoseconds += static_cast<std::uint64_t>(
            chain.terminal_head_ms * 1.0e6);
    }
    // The queued chain has drained, so nothing still reads the index
    // projections. They are re-acquired next token as cache hits; holding them
    // across the whole generation would leave the lease account open at its
    // end, which the caller checks.
    for (auto& scratch : rank_local_scratch) {
        for (auto& lease : scratch.index_query_projection) lease = {};
        for (auto& lease : scratch.index_weight_projection) lease = {};
    }
    // Every layer ran, both ranks agreed, and the head produced output. Only
    // now is the token's KV allowed to become visible.
    result = transaction.commit();
    if (config.enable_layer_hash_trace) {
        record_layer_hash(position, token, kLayers - 1U, hidden);
    }
    return result;
#else
    result.errors.emplace_back(
        "rank-local decode requires an NCCL-enabled build");
    return result;
#endif
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_patch_pages(
    const RankLocalLayerScratch& scratch) {
    ValidationResult result;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        const auto bytes = std::span<const std::byte>(scratch.patches[rank]);
        std::size_t cursor = 0U;
        for (const auto& write : scratch.page_writes[rank]) {
            const auto extent = static_cast<std::size_t>(write.data_bytes) +
                                write.scale_bytes;
            if (write.buffer == nullptr || cursor + extent > bytes.size()) {
                result.errors.emplace_back(
                    "rank-local page patch is truncated for rank " +
                    std::to_string(rank));
                return result;
            }
            const std::array<CudaBufferPatch, 2U> patches{{
                {write.data_offset, bytes.subspan(cursor, write.data_bytes)},
                {write.scale_offset,
                 bytes.subspan(cursor + write.data_bytes, write.scale_bytes)},
            }};
            result = cuda.update_buffer(*write.buffer, patches);
            if (!result.ok()) return result;
            cursor += extent;
        }
        if (cursor != bytes.size()) {
            result.errors.emplace_back(
                "rank-local page patch has unused bytes for rank " +
                std::to_string(rank));
            return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_candidates(
    std::uint32_t layer, std::uint32_t position,
    std::span<const std::uint32_t> indexed_positions,
    RankLocalLayerScratch& scratch, bool in_chain) {
    ValidationResult result;
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

    // Leases are released only after the executor has consumed the pages, so
    // they are cleared here rather than at the end of the previous layer.
    scratch.leases.clear();
    for (auto& pages : scratch.pages) pages.clear();
    scratch.compressed_block_leased.clear();
    std::unordered_map<std::uint64_t, std::uint32_t> page_indices;
    // One logical row order, two device page lists. Both ranks index the same
    // candidate array, so a page must occupy the same index in both lists.
    const auto locate = [&](Dsv4KvBlockKind kind,
                            const std::vector<Dsv4KvBlockInfo>& table,
                            std::uint32_t logical_row,
                            CudaDsv4AttentionCandidate& candidate) {
        const auto located = locate_physical_kv_block(table, logical_row);
        if (located == table.size()) {
            result.errors.emplace_back(
                "rank-local attention candidate page is unavailable");
            return;
        }
        const auto found = table.begin() + static_cast<std::ptrdiff_t>(located);
        const auto begin = found->logical_begin / found->compression_ratio;
        // A compressed block owns the page slot matching its block-table index;
        // a sliding block is appended after that reserved region, keeping the
        // lazy numbering it has always had.
        const bool positional =
            located < scratch.compressed_block_leased.size() &&
            kind != Dsv4KvBlockKind::Sliding;
        std::uint32_t page_index = 0U;
        if (positional) {
            page_index = static_cast<std::uint32_t>(located);
            if (scratch.compressed_block_leased[located] == 0U) {
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, kind, layer, logical_row, rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank][page_index] = {
                        scratch.leases.back().buffer(), found->capacity_rows};
                }
                scratch.compressed_block_leased[located] = 1U;
            }
        } else {
            auto page = page_indices.find(found->id);
            if (page == page_indices.end()) {
                const auto index =
                    static_cast<std::uint32_t>(scratch.pages[0].size());
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, kind, layer, logical_row, rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank].push_back(
                        {scratch.leases.back().buffer(),
                         found->capacity_rows});
                }
                page = page_indices.emplace(found->id, index).first;
            }
            page_index = page->second;
        }
        candidate.page = page_index;
        candidate.row = static_cast<std::uint32_t>(logical_row - begin);
        candidate.valid = true;
    };

    const auto compressed_count = ratio == 0U ? 0U : (position + 1U) / ratio;
    const bool sparse = ratio == 4U &&
        attention_state[layer].indexer_compressor.ratio == 4U;
    if (sparse) {
        // Positional page indexing for the compressed stream: a block's page
        // index is its block-table index, fixed before any candidate is
        // examined.
        //
        // The lazy first-touch numbering below cannot survive device-side
        // selection, which does not know which blocks a candidate set will
        // touch, let alone in what order. Making the mapping positional is what
        // lets a device kernel resolve a selected row to a page without the
        // host having seen the selection.
        //
        // Only the *numbering* is positional. The lease is still taken on first
        // touch, so an unselected block is never leased -- 512 of 4,096 blocks
        // at the declared context rather than all of them -- and the page slot
        // of an untouched block stays empty because no candidate can reference
        // it.
        std::uint32_t attendable_blocks = 0U;
        for (const auto& block : compressed) {
            const auto first_row = block.compression_ratio == 0U
                ? 0U : block.logical_begin / block.compression_ratio;
            if (first_row >= compressed_count) break;
            ++attendable_blocks;
        }
        for (auto& pages : scratch.pages) {
            pages.assign(attendable_blocks, CudaDsv4PhysicalPage{});
        }
        scratch.compressed_block_leased.assign(attendable_blocks, 0U);
        // In-chain selection has no first touch to lease on, so every
        // attendable block must carry a live page before the layer is queued.
        // Its block descriptor travels with it: the device resolves a selected
        // row against this table, and the page it names is this table's index.
        if (in_chain) {
            scratch.blocks.clear();
            for (std::uint32_t index = 0U; index < attendable_blocks; ++index) {
                const auto& block = compressed[index];
                const auto logical_row =
                    block.logical_begin / block.compression_ratio;
                for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld;
                     ++rank) {
                    auto lease = kv_cache->acquire_device(
                        active_sequence, attention_state[layer].compressor.kind,
                        layer, static_cast<std::uint32_t>(logical_row), rank);
                    if (!lease.ok()) {
                        append_errors(result, std::move(lease.errors));
                        return result;
                    }
                    scratch.leases.push_back(std::move(lease.value));
                    scratch.pages[rank][index] = {
                        scratch.leases.back().buffer(), block.capacity_rows};
                }
                scratch.compressed_block_leased[index] = 1U;
                scratch.blocks.push_back(CudaDsv4KvBlockDescriptor{
                    block.logical_begin, block.used_rows,
                    block.compression_ratio});
            }
        }
    }
    const auto compressed_width = ratio == 0U ? 0U : ratio == 4U
        ? kIndexTopK
        : ((std::max(1U, compressed_count) + 127U) / 128U) * 128U;
    constexpr std::uint32_t sliding_width = kWindow;
    scratch.candidates.assign(
        static_cast<std::size_t>(compressed_width) + sliding_width, {});
    const auto attended_compressed = sparse
        ? static_cast<std::uint32_t>(indexed_positions.size())
        : compressed_count;
    if (attended_compressed > compressed_width) {
        result.errors.emplace_back(
            "rank-local attention compressed candidates exceed their fixed "
            "region");
        return result;
    }
    // The in-chain compressed region is filled by the resolution kernel from a
    // selection this function never sees.
    if (!in_chain) {
        for (std::uint32_t item = 0U; item < attended_compressed; ++item) {
            const auto logical_row = sparse ? indexed_positions[item] : item;
            locate(attention_state[layer].compressor.kind, compressed,
                   logical_row, scratch.candidates[item]);
            if (!result.ok()) return result;
        }
    }
    const auto window_count = std::min(position + 1U, kWindow);
    for (std::uint32_t item = 0U; item < window_count; ++item) {
        const auto logical_row = position + 1U - window_count + item;
        locate(Dsv4KvBlockKind::Sliding, sliding, logical_row,
               scratch.candidates[
                   static_cast<std::size_t>(compressed_width) + item]);
        if (!result.ok()) return result;
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_warm_index_projections() {
    ValidationResult result;
    if (!rank_local_active || weights == nullptr) return result;
    auto cuda_demand = weights->demand();
    for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
        if (attention_state[layer].indexer_compressor.ratio == 0U) continue;
        const auto prefix = layer_prefix(layer) + "attn.indexer.";
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            // Acquired and released: this only has to put the pair in the
            // cache on this rank's device. The per-token acquire that follows
            // is then a hit, and the lease account stays balanced.
            Dsv4WeightCache::Lease query_projection;
            Dsv4WeightCache::Lease weight_projection;
            result = weights->acquire(rank, prefix + "wq_b",
                                      kIndexHeads * kIndexHeadDim, kQueryRank,
                                      query_projection);
            if (!result.ok()) return result;
            result = weights->acquire(rank, prefix + "weights_proj",
                                      kIndexHeads, kHidden, weight_projection);
            if (!result.ok()) return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_index_selection(
    std::uint32_t layer, std::uint32_t position,
    RankLocalLayerScratch& scratch) {
    ValidationResult result;
    const auto& state = attention_state[layer].indexer_compressor;
    const auto prefix = layer_prefix(layer) + "attn.indexer.";
    const auto compressed_count = (position + 1U) / state.ratio;
    auto& blocks = physical_index_blocks;
    result = kv_cache->block_table_into(
        active_sequence, Dsv4KvBlockKind::LearnedIndex, layer, blocks);
    if (!result.ok()) return result;

    // Both ranks score the same replicated history on their own device, so
    // each needs its own leases and its own page list over the same rows.
    auto cuda_demand = weights->demand();
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        result = weights->acquire(rank, prefix + "wq_b",
                                  kIndexHeads * kIndexHeadDim, kQueryRank,
                                  scratch.index_query_projection[rank]);
        if (!result.ok()) return result;
        result = weights->acquire(rank, prefix + "weights_proj", kIndexHeads,
                                  kHidden,
                                  scratch.index_weight_projection[rank]);
        if (!result.ok()) return result;
        std::uint32_t remaining = compressed_count;
        for (const auto& block : blocks) {
            if (remaining == 0U) break;
            const auto logical_row = block.logical_begin /
                                     block.compression_ratio;
            auto lease = kv_cache->acquire_device(
                active_sequence, Dsv4KvBlockKind::LearnedIndex, layer,
                static_cast<std::uint32_t>(logical_row), rank);
            if (!lease.ok()) {
                append_errors(result, std::move(lease.errors));
                return result;
            }
            const auto rows = std::min(remaining, block.used_rows);
            scratch.index_leases.push_back(std::move(lease.value));
            // A physical device lease holds the block-major payload alone;
            // acquire_device strips the header on upload.
            scratch.index_pages[rank].push_back(CudaDsv4PhysicalIndexPage{
                scratch.index_leases.back().buffer(), 0U, block.capacity_rows,
                rows});
            remaining -= rows;
        }
        if (remaining != 0U) {
            result.errors.emplace_back(
                "rank-local in-chain index history is incomplete at layer " +
                std::to_string(layer));
            return result;
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::rank_local_prepare_layer(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position,
    Dsv4RankLocalKvTransaction& transaction,
    RankLocalLayerScratch& scratch, Dsv4RankLocalLayerCall& call) {
    ValidationResult result;
    auto& state = attention_state[layer];
    const auto ratio = state.compressor.ratio;
    const auto index_ratio = state.indexer_compressor.ratio;
    const auto slot = layer_device(layer);
    const auto prefix = layer_prefix(layer) + "attn.";

    // Every queued call has completed before the next token enters this
    // function. Release that layer's previous-token candidate leases before
    // reserving its next row; otherwise the cache correctly refuses to mutate
    // a block that still appears in flight.
    scratch.leases.clear();
    scratch.index_leases.clear();
    for (auto& pages : scratch.pages) pages.clear();
    for (auto& pages : scratch.index_pages) pages.clear();
    result = transaction.reserve_layer(
        layer, position, ratio, state.compressor.kind, index_ratio);
    if (!result.ok()) return result;

    // An indexed layer only stays in the chain once its selection runs on the
    // device. Below the threshold there is nothing to select: every compressed
    // row is attended, so the candidate list is determined by position alone.
    const auto index_compressed_count = index_ratio == 0U
        ? 0U : (position + 1U) / index_ratio;
    const bool in_chain_selection =
        index_ratio != 0U && index_compressed_count > kIndexTopK;

    // Every candidate is determined entirely by position and the reserved
    // block table, or resolved on the device from a selection queued in the
    // same command sequence. Build the complete borrowed call now; the live
    // Q/KV and compressor rows are produced later by the executor's
    // stream-ordered callback, so all 43 layers can be submitted before the one
    // completion boundary.
    if (index_ratio == 0U || in_chain_selection) {
        auto sink = host_tensor(prefix + "attn_sink", kHeads);
        if (!sink.ok()) {
            append_errors(result, std::move(sink.errors));
            return result;
        }
        // The compressor state advance runs inside the page callback, on a
        // CUDA host node where a missing tensor could not be reported cleanly.
        // Fault it into the host-tensor map here instead.
        const auto residency = [&](const CompressorState& compressor,
                                   const std::string& compressor_prefix) {
            if (compressor.ratio == 0U) return true;
            const auto dimensions = static_cast<std::size_t>(
                compressor.coefficient) * compressor.head_dim;
            auto ape = host_tensor(
                compressor_prefix + "ape",
                static_cast<std::uint64_t>(compressor.ratio) * dimensions);
            auto norm_weight = host_tensor(compressor_prefix + "norm.weight",
                                           compressor.head_dim);
            if (!ape.ok() || !norm_weight.ok()) {
                append_errors(result, ape.ok() ? std::move(norm_weight.errors)
                                               : std::move(ape.errors));
                return false;
            }
            return true;
        };
        if (!residency(state.compressor, prefix + "compressor.") ||
            !residency(state.indexer_compressor,
                       prefix + "indexer.compressor.")) {
            return result;
        }
        for (std::size_t index = 0U; index < scratch.cosines.size(); ++index) {
            const float angle = static_cast<float>(position) *
                                state.frequencies[index];
            scratch.cosines[index] = std::cos(angle);
            scratch.inverse_sines[index] = -std::sin(angle);
            // The index query rotates with the same angles but keeps the
            // forward sine, which is the sign convention index_select() uses.
            scratch.index_cosines[index] = scratch.cosines[index];
            scratch.index_sines[index] = -scratch.inverse_sines[index];
        }

        scratch.patches[0].clear();
        scratch.patches[1].clear();
        // The executor only exists in an NCCL build. Rank-local decode already
        // fails closed at initialization without it, so this branch is
        // unreachable there; it still reports rather than falling through, and
        // the guard is what keeps a default STRATA_ENABLE_NCCL=OFF build
        // compiling.
#if defined(STRATA_HAS_NCCL)
        result = rank_local_executor->replica_page_patch_staging(
            layer, static_cast<std::size_t>(transaction.patch_bytes(layer)),
            scratch.replica_patch);
#else
        result.errors.emplace_back(
            "rank-local replica page staging requires an NCCL build");
#endif
        if (!result.ok()) return result;
        std::fill(scratch.replica_patch.begin(), scratch.replica_patch.end(),
                  std::byte{});
        scratch.compressed_row.clear();
        scratch.index_row.clear();
        scratch.indexed_positions.clear();
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            result = transaction.page_writes(
                layer, rank, scratch.page_writes[rank]);
            if (!result.ok()) return result;
        }
        const auto candidate_started = std::chrono::steady_clock::now();
        result = rank_local_candidates(layer, position, {}, scratch,
                                       in_chain_selection);
        graph_stats.rank_local_candidate_nanoseconds +=
            elapsed_nanoseconds(candidate_started);
        if (!result.ok()) return result;
        if (in_chain_selection) {
            result = rank_local_index_selection(layer, position, scratch);
            if (!result.ok()) return result;
        }

        call = {};
        call.layer = layer;
        call.position = position;
        call.weights = rank_local_weights->layer_view(layer, token);
        if (!in_chain_selection) {
            // The sparse-index compressor exists in the resident store for the
            // 1M operating point but is not part of this request's active
            // state.
            for (auto& rank_weights : call.weights.rank) {
                rank_weights.index_compressor_value = nullptr;
                rank_weights.index_compressor_gate = nullptr;
                rank_weights.index_compressor_elements = 0U;
            }
        }
        call.head_sinks = *sink.value;
        for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
            call.pages[rank] = scratch.pages[rank];
            auto& context = rank_local_page_contexts[layer][rank];
            context.owner = this;
            context.transaction = &transaction;
            context.scratch = &scratch;
            context.layer = layer;
            context.position = position;
            context.rank = rank;
            context.result = {};
            context.invoked = false;
            context.elapsed_nanoseconds = 0U;
            call.page_patches[rank].callback = rank == 0U
                ? rank_local_page_patch_callback : nullptr;
            call.page_patches[rank].context = rank == 0U ? &context : nullptr;
            call.page_patches[rank].ready_patch = rank == 1U
                ? std::span<const std::byte>(scratch.replica_patch)
                : std::span<const std::byte>{};
            call.page_patches[rank].writes = scratch.page_writes[rank];
        }
        call.candidates = scratch.candidates;
        call.inverse_rope_cosines = scratch.cosines;
        call.inverse_rope_sines = scratch.inverse_sines;
        if (in_chain_selection) {
            call.selection.active = true;
            for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
                call.selection.query_projection[rank] =
                    &scratch.index_query_projection[rank].weight();
                call.selection.weight_projection[rank] =
                    &scratch.index_weight_projection[rank].weight();
                call.selection.index_pages[rank] = scratch.index_pages[rank];
            }
            call.selection.blocks = scratch.blocks;
            call.selection.rope_cosines = scratch.index_cosines;
            call.selection.rope_sines = scratch.index_sines;
            call.selection.heads = kIndexHeads;
            call.selection.head_dim = kIndexHeadDim;
            call.selection.rope_dim = kRopeDim;
            call.selection.top_k = kIndexTopK;
            call.selection.compressed_width = kIndexTopK;
            call.selection.weight_scale = kIndexQueryScale;
            ++graph_stats.attention_index_queries;
            graph_stats.attention_index_candidates += index_compressed_count;
            graph_stats.attention_index_selected += kIndexTopK;
            graph_stats.attention_index_cuda_dispatches +=
                kDsv4RankLocalWorld;
        }
        call.ordered_page_patches = true;
        call.terminal = layer + 1U == kLayers;
        return result;
    }

    // One host-visible preparation per layer, on the slot that owns this
    // layer's centralized compressor weights. The executor's own preparation
    // is device-only and computes no compressor projection, so the pooled
    // rows and the index query have to come from here. It is one extra
    // projection pass per layer, not per rank: the result is replicated.
    //
    // The leases are scope-local because this preparation is synchronous:
    // with a host-visible query and no page-patch callback the backend copies
    // its diagnostics and synchronizes before returning, so nothing queued
    // still reads the weights after this function exits.
    Dsv4WeightCache::Lease query_a;
    Dsv4WeightCache::Lease query_b;
    Dsv4WeightCache::Lease key_value_weight;
    Dsv4WeightCache::Lease compressor_value;
    Dsv4WeightCache::Lease compressor_gate;
    Dsv4WeightCache::Lease index_value;
    Dsv4WeightCache::Lease index_gate;
    result = weights->acquire(
        slot, prefix + "wq_a", kQueryRank, kHidden, query_a);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wq_b", kHeads * kHeadDim, kQueryRank, query_b);
    if (!result.ok()) return result;
    result = weights->acquire(
        slot, prefix + "wkv", kHeadDim, kHidden, key_value_weight);
    if (!result.ok()) return result;
    scratch.compressor_values.clear();
    scratch.compressor_scores.clear();
    if (ratio != 0U) {
        const auto dimensions =
            static_cast<std::size_t>(state.compressor.coefficient) *
            state.compressor.head_dim;
        result = weights->acquire(slot, prefix + "compressor.wkv", dimensions,
                                  kHidden, compressor_value);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "compressor.wgate", dimensions,
                                  kHidden, compressor_gate);
        if (!result.ok()) return result;
        scratch.compressor_values.assign(dimensions, 0.0F);
        scratch.compressor_scores.assign(dimensions, 0.0F);
    }
    scratch.index_compressor_values.clear();
    scratch.index_compressor_scores.clear();
    if (index_ratio != 0U) {
        const auto dimensions =
            static_cast<std::size_t>(state.indexer_compressor.coefficient) *
            state.indexer_compressor.head_dim;
        result = weights->acquire(slot, prefix + "indexer.compressor.wkv",
                                  dimensions, kHidden, index_value);
        if (!result.ok()) return result;
        result = weights->acquire(slot, prefix + "indexer.compressor.wgate",
                                  dimensions, kHidden, index_gate);
        if (!result.ok()) return result;
        scratch.index_compressor_values.assign(dimensions, 0.0F);
        scratch.index_compressor_scores.assign(dimensions, 0.0F);
    }
    auto query_norm = host_tensor(prefix + "q_norm.weight", kQueryRank);
    if (!query_norm.ok()) {
        append_errors(result, std::move(query_norm.errors));
        return result;
    }
    auto key_value_norm = host_tensor(prefix + "kv_norm.weight", kHeadDim);
    if (!key_value_norm.ok()) {
        append_errors(result, std::move(key_value_norm.errors));
        return result;
    }
    auto sink = host_tensor(prefix + "attn_sink", kHeads);
    if (!sink.ok()) {
        append_errors(result, std::move(sink.errors));
        return result;
    }

    for (std::size_t index = 0U; index < scratch.cosines.size(); ++index) {
        const float angle = static_cast<float>(position) *
                            state.frequencies[index];
        scratch.cosines[index] = std::cos(angle);
        // The executor recovers the forward sine as its negation, so only the
        // inverse pair travels in the call.
        scratch.inverse_sines[index] = -std::sin(angle);
    }

    CudaDsv4AttentionPrepareRequest request;
    request.query_a = &query_a.weight();
    request.query_b = &query_b.weight();
    request.key_value = &key_value_weight.weight();
    if (!scratch.compressor_values.empty()) {
        request.compressor_value = &compressor_value.weight();
        request.compressor_gate = &compressor_gate.weight();
    }
    if (!scratch.index_compressor_values.empty()) {
        request.index_compressor_value = &index_value.weight();
        request.index_compressor_gate = &index_gate.weight();
    }
    request.query_norm = *query_norm.value;
    request.key_value_norm = *key_value_norm.value;
    // Forward RoPE for the key/value row; the query carries the same angles.
    std::array<float, kRopeDim / 2U> sines{};
    for (std::size_t index = 0U; index < sines.size(); ++index) {
        sines[index] = -scratch.inverse_sines[index];
    }
    request.rope_cosines = scratch.cosines;
    request.rope_sines = sines;
    request.mhc_device = devices[slot];
    request.maximum_workspace_bytes = 1ULL << 20U;
    // This preparation exists to produce host-visible projections, not to
    // stage a command. The executor prepares again per rank to stage the one
    // its attention consumes, and a published query left here would make that
    // second preparation out of order.
    request.host_only = true;
    scratch.query_rank.assign(kQueryRank, 0.0F);
    scratch.key_value.assign(kHeadDim, 0.0F);
    auto prepare_started = std::chrono::steady_clock::now();
    result = cuda.dsv4_prepare_attention(
        devices[slot], request, scratch.query_rank, scratch.key_value,
        scratch.compressor_values, scratch.compressor_scores,
        scratch.index_compressor_values, scratch.index_compressor_scores);
    graph_stats.attention_query_nanoseconds +=
        elapsed_nanoseconds(prepare_started);
    if (!result.ok()) return result;

    // Pool the compressor rows without publishing them: one logical row has to
    // reach two devices' pages, so the transaction owns the encode.
    result = compress_state(layer, state.compressor, prefix + "compressor.",
                            {}, position, state.frequencies,
                            scratch.compressor_values,
                            scratch.compressor_scores, nullptr, {},
                            &scratch.compressed_row);
    if (!result.ok()) return result;
    result = compress_state(layer, state.indexer_compressor,
                            prefix + "indexer.compressor.", {}, position,
                            state.frequencies,
                            scratch.index_compressor_values,
                            scratch.index_compressor_scores, nullptr, {},
                            &scratch.index_row);
    if (!result.ok()) return result;

    const auto kv_started = std::chrono::steady_clock::now();
    const auto patch_bytes =
        static_cast<std::size_t>(transaction.patch_bytes(layer));
    std::array<std::span<std::byte>, kDsv4RankLocalWorld> patches{};
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        scratch.patches[rank].assign(patch_bytes, std::byte{});
        patches[rank] = scratch.patches[rank];
        result = transaction.page_writes(layer, rank,
                                         scratch.page_writes[rank]);
        if (!result.ok()) return result;
    }
    result = transaction.commit_layer(layer, scratch.key_value,
                                      scratch.compressed_row, patches,
                                      scratch.index_row);
    if (!result.ok()) return result;
    result = rank_local_patch_pages(scratch);
    graph_stats.rank_local_kv_nanoseconds += elapsed_nanoseconds(kv_started);
    if (!result.ok()) return result;

    scratch.indexed_positions.clear();
    if (index_ratio != 0U) {
        const auto compressed_count = (position + 1U) / index_ratio;
        if (compressed_count > kIndexTopK &&
            rank_local_attention_input.size() != kHidden) {
            result.errors.emplace_back(
                "rank-local sparse selection has no attention input at layer " +
                std::to_string(layer));
            return result;
        }
        auto select_started = std::chrono::steady_clock::now();
        // The host-visible preparation above is this device's most recent, so
        // the index projections read its activations rather than sending the
        // query rank and layer input back across the bus.
        result = index_select(layer, rank_local_attention_input,
                              scratch.query_rank, position,
                              scratch.indexed_positions, true);
        graph_stats.attention_index_nanoseconds +=
            elapsed_nanoseconds(select_started);
        if (!result.ok()) return result;
    }
    const auto candidate_started = std::chrono::steady_clock::now();
    result = rank_local_candidates(layer, position, scratch.indexed_positions,
                                   scratch);
    graph_stats.rank_local_candidate_nanoseconds +=
        elapsed_nanoseconds(candidate_started);
    if (!result.ok()) return result;

    call = {};
    call.layer = layer;
    call.position = position;
    call.weights = rank_local_weights->layer_view(layer, token);
    // Indexed-context preparation remains the explicit sequential arm until
    // Step 4 moves selection inside the device command. Its separate
    // host-visible preparation already computed these projections.
    for (auto& rank_weights : call.weights.rank) {
        rank_weights.compressor_value = nullptr;
        rank_weights.compressor_gate = nullptr;
        rank_weights.index_compressor_value = nullptr;
        rank_weights.index_compressor_gate = nullptr;
        rank_weights.compressor_elements = 0U;
        rank_weights.index_compressor_elements = 0U;
    }
    call.head_sinks = *sink.value;
    for (std::size_t rank = 0U; rank < kDsv4RankLocalWorld; ++rank) {
        call.pages[rank] = scratch.pages[rank];
    }
    call.candidates = scratch.candidates;
    call.inverse_rope_cosines = scratch.cosines;
    call.inverse_rope_sines = scratch.inverse_sines;
    // No page patch: both ranks' pages already hold this position's rows, so
    // the executor's preparation stays device-only and costs no host sync.
    call.terminal = layer + 1U == kLayers;
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_hidden(
    std::uint32_t token, std::uint32_t position,
    std::span<float> hidden, std::vector<float>* fused_logits) {
    ValidationResult result;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back(
            "DeepSeek device mHC hidden state has the wrong shape");
        return result;
    }
    const auto device = devices[mhc_slot];
    std::vector<float> weighted(
        config.enable_layer_hash_trace ? kHidden : 0U);
    std::vector<float> layer_input(kHidden);
    Dsv4WeightCache::Lease head_lease;
    const bool fuse_head = fused_logits != nullptr &&
        !config.enable_layer_hash_trace &&
        layer_device(kLayers - 1U) == mhc_slot;
    if (fused_logits != nullptr) fused_logits->clear();
    if (fuse_head) {
        result = weights->acquire(
            layer_device(kLayers - 1U), "head", kVocabulary, kHidden,
            head_lease);
        if (!result.ok()) return result;
        fused_logits->assign(kVocabulary, 0.0F);
    }
    auto phase_started = std::chrono::steady_clock::now();
    const bool device_only_begin = !config.enable_layer_hash_trace;
    if (!device_only_begin) {
        result = cuda.dsv4_mhc_begin(
            device, device_mhc_weights[0U][0U], hidden, weighted,
            layer_input);
    } else {
        result = cuda.dsv4_mhc_begin_device(
            device, device_mhc_weights[0U][0U], hidden);
    }
    graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
    if (!result.ok()) return result;
    pending_mhc_attention_transition = false;
    completed_attention_mhc_transition = false;
    completed_router_projection = false;
    deferred_attention_moe_input = false;

    constexpr std::uint32_t branch_count = 2U * kLayers;
    for (std::uint32_t flat = 0U; flat < branch_count; ++flat) {
        const auto layer = flat / 2U;
        const auto branch_index = flat % 2U;
        const std::string branch = branch_index == 0U ? "attn" : "ffn";
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_mhc_pre", weighted);
            record_operation_hash(position, token, layer,
                                  branch + "_norm", layer_input);
        }

        std::vector<float> branch_output(
            config.enable_layer_hash_trace ? kHidden : 0U);
        phase_started = std::chrono::steady_clock::now();
        if (branch_index == 0U) {
            result = attention(
                layer, layer_input, position, branch_output);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
        } else {
            result = moe(
                layer, token, layer_input, branch_output, position);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
        }
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_output", branch_output);
        }

        phase_started = std::chrono::steady_clock::now();
        if (completed_attention_mhc_transition) {
            if (branch_index != 0U || config.enable_layer_hash_trace) {
                result.errors.emplace_back(
                    "DeepSeek combined attention transition is out of order");
            } else {
                if (!deferred_attention_moe_input) {
                    std::copy(combined_attention_mhc_input.begin(),
                              combined_attention_mhc_input.end(),
                              layer_input.begin());
                }
                completed_attention_mhc_transition = false;
            }
        } else if (flat + 1U < branch_count) {
            const auto next = flat + 1U;
            const auto next_layer = next / 2U;
            const auto next_branch = next % 2U;
            const bool combine_with_attention =
                branch_index == 1U && next_branch == 0U &&
                !config.enable_layer_hash_trace &&
                attention_state[next_layer].indexer_compressor.ratio == 0U;
            if (combine_with_attention) {
                if (pending_mhc_attention_transition) {
                    result.errors.emplace_back(
                        "DeepSeek mHC attention transition is already pending");
                } else {
                    pending_mhc_attention_transition = true;
                }
            } else {
                auto post_output = config.enable_layer_hash_trace
                    ? hidden : std::span<float>{};
                result = cuda.dsv4_mhc_transition_device(
                    device, device_mhc_weights[next_layer][next_branch],
                    weighted, layer_input, post_output);
            }
        } else if (fuse_head) {
            device_head_context = {};
            device_head_context.owner = this;
            result = cuda.enqueue_dsv4_mhc_finish_head_device(
                device, head_lease.weight(), device_head_callback,
                &device_head_context);
        } else {
            result = cuda.dsv4_mhc_finish_device(device, hidden);
        }
        graph_stats.mhc_post_nanoseconds +=
            elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer,
                                  branch + "_mhc_post", hidden);
            if (branch_index == 1U) {
                record_layer_hash(position, token, layer, hidden);
            }
        }
    }
    auto moe_collected = collect_host_routed_moe_chain();
    if (!moe_collected.ok()) {
        append_errors(result, std::move(moe_collected.errors));
    }
    if (fuse_head && result.ok()) {
        auto completed = cuda.complete_dsv4_mhc_head_device(
            device, *fused_logits);
        if (!completed.ok()) {
            append_errors(result, std::move(completed.errors));
        }
        if (!device_head_context.result.ok()) {
            append_errors(result,
                          std::move(device_head_context.result.errors));
        }
        if (!device_head_context.invoked) {
            result.errors.emplace_back(
                "DeepSeek output-head host callback was not invoked");
        }
    }
    if (pending_mhc_attention_transition) {
        result.errors.emplace_back(
            "DeepSeek mHC attention transition remained pending");
    }
    if (result.ok() && fused_logits != nullptr && !fuse_head) {
        result = head_logits(hidden, *fused_logits);
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_prefill_page(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base,
    std::span<float> hidden) {
    auto result = device_mhc_forward_prefill_page_impl(
        tokens, position_base, hidden);
    // A failure can leave the chain index set; the token-major path must
    // find it empty.
    host_moe_chain_row.reset();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_mhc_forward_prefill_page_impl(
    std::span<const std::uint32_t> tokens, std::uint32_t position_base,
    std::span<float> hidden) {
    ValidationResult result;
    const auto rows = tokens.size();
    const auto stride = static_cast<std::size_t>(kMhc) * kHidden;
    if (rows == 0U || rows > config.prefill_page_tokens ||
        hidden.size() != rows * stride ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek device mHC prefill page has incompatible dimensions");
        return result;
    }
    const auto device = devices[mhc_slot];
    result = cuda.dsv4_mhc_reserve_slots(
        device, static_cast<std::uint32_t>(rows));
    if (!result.ok()) return result;

    const bool trace = config.enable_layer_hash_trace;
    // A non-empty attention output selects the unfused physical attention
    // command. The fused one cannot be used here: it defers the MoE input and
    // holds the layer's KV device leases until the collect, and a block
    // refuses to be appended to while any lease is outstanding, so the second
    // row of any page would fail. The branch itself still reaches the mHC
    // workspace on the device either way.
    std::vector<float> layer_inputs(rows * kHidden);
    std::vector<float> branch_outputs(rows * kHidden);
    std::vector<std::uint32_t> row_slots(rows);
    for (std::size_t row = 0U; row < rows; ++row) {
        row_slots[row] = static_cast<std::uint32_t>(row);
    }
    std::vector<float> router_logits(rows * kExperts);
    std::vector<Dsv4Route> routes(rows);
    // Prefill puts the routed experts on the GPU once a page is wide enough to
    // pay for uploading each distinct expert: one upload then serves every row
    // that chose it, instead of every row reading the weights out of host DRAM
    // again. Decode never reaches here and keeps the CPU shards.
    const bool page_moe_on_device =
        config.prefill_device_moe_minimum_rows != 0U &&
        rows >= config.prefill_device_moe_minimum_rows;
    std::vector<float> moe_outputs(page_moe_on_device ? rows * kHidden : 0U);
    std::vector<std::vector<float>> weighted(
        rows, std::vector<float>(trace ? kHidden : 0U));
    const auto layer_input_row = [&](std::size_t row) {
        return std::span<float>(layer_inputs).subspan(row * kHidden, kHidden);
    };

    const auto select = [&](std::size_t row) {
        return cuda.dsv4_mhc_select_slot(
            device, static_cast<std::uint32_t>(row));
    };
    // Neither branch here is the fused command that carries state across a
    // row's attention or MoE call, so none of those flags may be set. They
    // would otherwise leak from one row into the next.
    const auto fused_state_is_clear = [this]() {
        return !pending_mhc_attention_transition &&
               !completed_attention_mhc_transition &&
               !completed_router_projection && !deferred_attention_moe_input;
    };

    for (std::size_t row = 0U; row < rows; ++row) {
        result = select(row);
        if (!result.ok()) return result;
        const auto hidden_row = hidden.subspan(row * stride, stride);
        const auto phase_started = std::chrono::steady_clock::now();
        result = cuda.dsv4_mhc_begin(
            device, device_mhc_weights[0U][0U], hidden_row, weighted[row],
            layer_input_row(row));
        graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
    }

    constexpr std::uint32_t branch_count = 2U * kLayers;
    for (std::uint32_t flat = 0U; flat < branch_count; ++flat) {
        const auto layer = flat / 2U;
        const auto branch_index = flat % 2U;
        const std::string branch = branch_index == 0U ? "attn" : "ffn";
        if (branch_index == 0U) {
            // One call attends the whole page: the query, key/value and output
            // projections become three row-batched matmuls instead of three
            // per row, and each row still appends and attends in position
            // order behind them.
            const auto phase_started = std::chrono::steady_clock::now();
            result = attention_page(layer, layer_inputs, position_base,
                                    branch_outputs, row_slots);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        } else {
            // Route the page in one batched projection, then decode each
            // selected expert's weight tile once for every row that chose it.
            // The whole page must be routed before any of it is executed, so
            // each row's mHC-pre and norm records are emitted here to keep a
            // row's operation order the same as the token-major path's.
            // The router stays per row. Its logits are not rounded to BF16,
            // and the row-batched projection reassociates the accumulation:
            // the selection is unchanged but the coefficients move by a ULP,
            // which is a different model.
            const auto router_started = std::chrono::steady_clock::now();
            for (std::size_t row = 0U; row < rows; ++row) {
                const auto position =
                    position_base + static_cast<std::uint32_t>(row);
                auto logits = std::span<float>(router_logits)
                    .subspan(row * kExperts, kExperts);
                result = linear(layer_device(layer),
                                layer_prefix(layer) + "ffn.gate", kExperts,
                                kHidden, layer_input_row(row), logits, false);
                if (!result.ok()) return result;
                if (trace) {
                    record_operation_hash(position, tokens[row], layer,
                                          "ffn_mhc_pre", weighted[row]);
                    record_operation_hash(position, tokens[row], layer,
                                          "ffn_norm", layer_input_row(row));
                }
                result = route_moe(
                    layer, tokens[row],
                    std::span<const float>(router_logits)
                        .subspan(row * kExperts, kExperts),
                    position, routes[row]);
                if (!result.ok()) return result;
            }
            graph_stats.moe_router_nanoseconds +=
                elapsed_nanoseconds(router_started);
            if (page_moe_on_device) {
                // Prefill places the routed experts on the GPU: each distinct
                // expert of the page is uploaded once and applied to all its
                // rows as a matmul. Decode keeps them in the NUMA-local CPU
                // shards, where the weights already live and a step has six
                // experts and one row.
                const auto phase_started = std::chrono::steady_clock::now();
                result = execute_moe_page(layer, routes, layer_inputs,
                                          moe_outputs);
                graph_stats.moe_nanoseconds +=
                    elapsed_nanoseconds(phase_started);
                if (!result.ok()) return result;
            }
        }
        for (std::size_t row = 0U; row < rows; ++row) {
            result = select(row);
            if (!result.ok()) return result;
            // Both pending-callback tables are addressed by row for as long as
            // one layer is being swept across the page.
            host_moe_chain_row = static_cast<std::uint32_t>(row);
            const auto position =
                position_base + static_cast<std::uint32_t>(row);
            const auto hidden_row = hidden.subspan(row * stride, stride);
            if (trace && branch_index == 0U) {
                record_operation_hash(position, tokens[row], layer,
                                      "attn_mhc_pre", weighted[row]);
                record_operation_hash(position, tokens[row], layer,
                                      "attn_norm", layer_input_row(row));
            }

            // The attention branch left its result in this row's device
            // workspace. The routed experts, when they run on the GPU, produce
            // a host row instead, which the transition below uploads.
            const auto attended = std::span<const float>(branch_outputs)
                .subspan(row * kHidden, kHidden);
            const auto page_moe_row = std::span<const float>(moe_outputs)
                .subspan(row * kHidden, kHidden);
            std::vector<float> branch_output(
                trace && branch_index == 1U && !page_moe_on_device ? kHidden
                                                                   : 0U);
            auto phase_started = std::chrono::steady_clock::now();
            if (branch_index == 1U && !page_moe_on_device) {
                // The page was routed above, so this joins the row's shared
                // expert with the routed partial the precompute produced.
                result = execute_moe(layer, routes[row], layer_input_row(row),
                                     branch_output);
                graph_stats.moe_nanoseconds +=
                    elapsed_nanoseconds(phase_started);
                if (!result.ok()) return result;
            }
            if (trace) {
                record_operation_hash(
                    position, tokens[row], layer, branch + "_output",
                    branch_index == 0U ? attended
                    : page_moe_on_device
                        ? page_moe_row
                        : std::span<const float>(branch_output));
            }

            phase_started = std::chrono::steady_clock::now();
            const auto post_output = trace ? hidden_row : std::span<float>{};
            const bool host_branch = branch_index == 1U && page_moe_on_device;
            if (flat + 1U < branch_count) {
                const auto next = flat + 1U;
                const auto& next_weights =
                    device_mhc_weights[next / 2U][next % 2U];
                result = host_branch
                    ? cuda.dsv4_mhc_transition(
                          device, next_weights, page_moe_row, weighted[row],
                          layer_input_row(row), post_output)
                    : cuda.dsv4_mhc_transition_device(
                          device, next_weights, weighted[row],
                          layer_input_row(row), post_output);
            } else {
                result = host_branch
                    ? cuda.dsv4_mhc_finish(device, page_moe_row, hidden_row)
                    : cuda.dsv4_mhc_finish_device(device, hidden_row);
            }
            graph_stats.mhc_post_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            if (trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_post", hidden_row);
                if (branch_index == 1U) {
                    record_layer_hash(position, tokens[row], layer, hidden_row);
                }
            }
            if (!fused_state_is_clear()) {
                result.errors.emplace_back(
                    "DeepSeek page-major branch left fused state behind");
                return result;
            }
        }
        host_moe_chain_row.reset();
        if (branch_index == 1U) {
            // One collect per layer drains the page's routed-MoE callbacks.
            auto collected = collect_host_routed_moe_chain();
            if (!collected.ok()) {
                append_errors(result, std::move(collected.errors));
                return result;
            }
        }
    }

    result = select(0U);
    if (!result.ok()) return result;
    graph_stats.forward_tokens += rows;
    return result;
}

