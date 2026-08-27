ValidationResult DeepSeekV4Runtime::Impl::expert(
    std::uint32_t layer, std::uint32_t expert_id,
    float routed_coefficient,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (expert_id >= kExperts || input.size() != kHidden || output.size() != kHidden) {
        result.errors.emplace_back("DeepSeek expert id or span shape is invalid");
        return result;
    }
    const auto slot = expert_device(expert_id);
    const auto prefix = layer_prefix(layer) + "ffn.experts." +
                        std::to_string(expert_id) + ".";
    std::vector<float> gate(kExpertIntermediate);
    std::vector<float> up(kExpertIntermediate);
    std::vector<float> activated(kExpertIntermediate);
    result = linear(slot, prefix + "w1", kExpertIntermediate, kHidden, input, gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "w3", kExpertIntermediate, kHidden, input, up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(activated, gate, up, 10.0F);
    if (!result.ok()) return result;
    for (auto& value : activated) {
        value *= routed_coefficient;
    }
    round_bf16(activated);
    return linear(slot, prefix + "w2", kHidden, kExpertIntermediate,
                  activated, output);
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    return host_routed_moe_impl(layer, &route, input, output, 0U, 0U);
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe_from_device_input(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position,
    std::span<float> output) {
    if (!output.empty()) {
        return {{"DeepSeek deferred CPU-MoE output must remain device-owned"}};
    }
    return enqueue_host_routed_moe(layer, token, position);
}

ValidationResult DeepSeekV4Runtime::Impl::enqueue_host_routed_moe(
    std::uint32_t layer, std::uint32_t token, std::uint32_t position) {
    ValidationResult result;
    const auto chain = host_moe_chain_row.value_or(layer);
    if (layer >= kLayers || host_moe_pending != chain ||
        expert_workers == nullptr) {
        result.errors.emplace_back(
            "DeepSeek fixed CPU-MoE command order is invalid");
        return result;
    }
    if (host_moe_pending == 0U) {
        host_moe_routed_cpu_before =
            device_moe_stats.routed_cpu_nanoseconds;
    }
    auto& context = host_moe_context(chain);
    context.owner = this;
    context.layer = layer;
    context.token = token;
    context.position = position;
    context.result = {};
    context.invoked = false;
    context.accepted = false;
    context.callback_finished = {};
    context.execution_started = std::chrono::steady_clock::now();
    if (context.input.size() != kHidden) context.input.resize(kHidden);

    if (context.shared.w1 == nullptr) {
        const auto shared_slot = mhc_slot;
        context.shared.coefficient = 1.0F;
        const auto prefix = layer_prefix(layer) + "ffn.shared_experts.";
        const std::array<std::string, 3U> names{
            prefix + "w1", prefix + "w3", prefix + "w2"};
        constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 3U>
            shapes{{{kExpertIntermediate, kHidden},
                    {kExpertIntermediate, kHidden},
                    {kHidden, kExpertIntermediate}}};
        const CudaWeight** outputs[]{
            &context.shared.w1, &context.shared.w3, &context.shared.w2};
        for (std::size_t index = 0U; index < names.size(); ++index) {
            auto acquired = weights->acquire(
                shared_slot, names[index], shapes[index].first,
                shapes[index].second, context.shared_leases[index]);
            if (!acquired.ok()) {
                append_errors(result, std::move(acquired.errors),
                              names[index]);
                return result;
            }
            *outputs[index] = &context.shared_leases[index].weight();
        }
    }
    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(
        context.execution_started);
    auto enqueued = cuda.enqueue_dsv4_host_moe_from_device_input(
        devices[mhc_slot], context.shared,
        kDeepSeekV4ExecutionContract.swiglu_limit,
        host_routed_moe_callback, &context);
    if (!enqueued.ok()) {
        append_errors(result, std::move(enqueued.errors),
                      "DeepSeek fixed CPU/shared MoE enqueue");
        return result;
    }
    ++host_moe_pending;
    return result;
}

bool DeepSeekV4Runtime::Impl::host_routed_moe_callback(
    void* opaque, std::span<const std::uint16_t> encoded_hidden,
    std::span<const float> router_logits,
    std::span<float> rank_partials) {
    if (opaque == nullptr) return false;
    auto& context = *static_cast<HostMoeContext*>(opaque);
    if (context.owner == nullptr) return false;
    return context.owner->execute_host_routed_moe_callback(
        context, encoded_hidden, router_logits, rank_partials);
}

bool DeepSeekV4Runtime::Impl::execute_host_routed_moe_callback(
    HostMoeContext& context,
    std::span<const std::uint16_t> encoded_hidden,
    std::span<const float> router_logits,
    std::span<float> rank_partials) {
    constexpr std::size_t shards = 2U;
    constexpr std::size_t shard_intermediate =
        kExpertIntermediate / shards;
    context.invoked = true;
    context.result = {};
    const auto routed_started = std::chrono::steady_clock::now();
    if (encoded_hidden.size() != kHidden ||
        router_logits.size() != kExperts ||
        rank_partials.size() != shards * kHidden) {
        context.result.errors.emplace_back(
            "DeepSeek fixed CPU-MoE callback shape is invalid");
        context.callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    for (std::size_t index = 0U; index < encoded_hidden.size(); ++index) {
        context.input[index] = std::bit_cast<float>(
            static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
        if (!std::isfinite(context.input[index])) {
            context.result.errors.emplace_back(
                "DeepSeek fixed CPU-MoE hidden row is non-finite");
            context.callback_finished = std::chrono::steady_clock::now();
            return false;
        }
    }
    const auto route_started = std::chrono::steady_clock::now();
    context.result = route_moe(
        context.layer, context.token, router_logits, context.position,
        context.route);
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(route_started);
    if (!context.result.ok()) {
        context.callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    for (std::size_t shard = 0U; shard < shards; ++shard) {
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            auto viewed = dsv4_tiled_expert_weights(
                resident.find_tiled_expert(
                    context.layer, context.route.experts[rank],
                    static_cast<std::uint32_t>(shard)),
                kHidden, kExpertIntermediate, shards);
            if (!viewed.ok()) {
                append_errors(context.result, std::move(viewed.errors));
                context.callback_finished = std::chrono::steady_clock::now();
                return false;
            }
            context.tiled[shard][rank] = viewed.value;
        }
    }


    constexpr std::size_t block = 32U;
    constexpr auto intermediate_blocks = shard_intermediate / block;
    auto phase_started = std::chrono::steady_clock::now();
    context.result = run_expert_ranges(
        kTopK * intermediate_blocks,
        [&](std::size_t shard, std::uint64_t task) {
            const auto rank = static_cast<std::size_t>(
                task / intermediate_blocks);
            const auto offset = (task % intermediate_blocks) * block;
            std::array<float, block> gate{};
            std::array<float, block> up{};
            for (std::size_t half = 0U; half < 2U; ++half) {
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(
                        gate.data() + half * 16U, 16U),
                    context.input,
                    context.tiled[shard][rank].w13_packed,
                    context.tiled[shard][rank].w13_scales,
                    2U * shard_intermediate,
                    offset + half * 16U);
                dsv4_tiled_expert_matvec16(
                    std::span<float, 16U>(
                        up.data() + half * 16U, 16U),
                    context.input,
                    context.tiled[shard][rank].w13_packed,
                    context.tiled[shard][rank].w13_scales,
                    2U * shard_intermediate,
                    shard_intermediate + offset + half * 16U);
            }
            auto* destination = tiled_activation.data() +
                (shard * kTopK + rank) * shard_intermediate + offset;
            for (std::size_t index = 0U; index < block; ++index) {
                destination[index] = gate[index] /
                    (1.0F + std::exp(-gate[index])) * up[index];
            }
        }, true);
    device_moe_stats.routed_gate_up_nanoseconds +=
        elapsed_nanoseconds(phase_started);

    phase_started = std::chrono::steady_clock::now();
    if (context.result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        context.result = run_expert_ranges(
            kTopK * hidden_blocks,
            [&](std::size_t shard, std::uint64_t task) {
                const auto rank = static_cast<std::size_t>(
                    task / hidden_blocks);
                const auto offset = (task % hidden_blocks) * block;
                const auto source = std::span<const float>(tiled_activation)
                    .subspan((shard * kTopK + rank) * shard_intermediate,
                             shard_intermediate);
                for (std::size_t half = 0U; half < 2U; ++half) {
                    auto destination = std::span<float, 16U>(
                        tiled_routed.data() +
                            (shard * kTopK + rank) * kHidden +
                            offset + half * 16U,
                        16U);
                    dsv4_tiled_expert_matvec16(
                        destination, source,
                        context.tiled[shard][rank].w2_packed,
                        context.tiled[shard][rank].w2_scales,
                        kHidden, offset + half * 16U);
                }
            }, true);
    }
    device_moe_stats.routed_down_nanoseconds +=
        elapsed_nanoseconds(phase_started);

    phase_started = std::chrono::steady_clock::now();
    if (context.result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        context.result = run_expert_ranges(
            hidden_blocks,
            [&](std::size_t shard, std::uint64_t task) {
                const auto offset = task * block;
                auto* destination =
                    rank_partials.data() + shard * kHidden + offset;
                for (std::size_t index = 0U; index < block; ++index) {
                    destination[index] = tiled_routed[
                        (shard * kTopK) * kHidden + offset + index] *
                        context.route.weights[0];
                }
                for (std::size_t rank = 1U; rank < kTopK; ++rank) {
                    const auto* source = tiled_routed.data() +
                        (shard * kTopK + rank) * kHidden + offset;
                    for (std::size_t index = 0U; index < block; ++index) {
                        destination[index] = std::fma(
                            source[index], context.route.weights[rank],
                            destination[index]);
                    }
                }
            }, false);
    }
    device_moe_stats.routed_reduce_nanoseconds +=
        elapsed_nanoseconds(phase_started);
    device_moe_stats.routed_cpu_nanoseconds +=
        elapsed_nanoseconds(routed_started);
    context.callback_finished = std::chrono::steady_clock::now();
    context.accepted = context.result.ok();
    return context.accepted;
}

DeepSeekV4Runtime::Impl::PhysicalAttentionContext&
DeepSeekV4Runtime::Impl::physical_attention_context(std::size_t index) {
    while (physical_attention_contexts.size() <= index) {
        physical_attention_contexts.push_back(
            std::make_unique<PhysicalAttentionContext>());
    }
    return *physical_attention_contexts[index];
}

DeepSeekV4Runtime::Impl::HostMoeContext&
DeepSeekV4Runtime::Impl::host_moe_context(std::size_t index) {
    while (host_moe_contexts.size() <= index) {
        host_moe_contexts.push_back(std::make_unique<HostMoeContext>());
    }
    return *host_moe_contexts[index];
}

ValidationResult DeepSeekV4Runtime::Impl::collect_host_routed_moe_chain() {
    ValidationResult result;
    if (host_moe_pending == 0U) return result;
    const auto collect_started = std::chrono::steady_clock::now();
    auto collected = cuda.collect_deepseek_moe(
        devices[mhc_slot], {}, {});
    if (!collected.ok()) {
        append_errors(result, std::move(collected.errors),
                      "DeepSeek fixed CPU/shared MoE collect");
    }
    for (std::uint32_t pending = 0U; pending < host_moe_pending; ++pending) {
        auto& context = host_moe_context(pending);
        if (context.invoked) ++device_moe_stats.host_callback_batches;
        if (!context.invoked || !context.accepted) {
            ++device_moe_stats.host_callback_failures;
        }
        if (!context.result.ok()) {
            append_errors(result, std::move(context.result.errors));
        }
        ++device_moe_stats.batches;
        ++device_moe_stats.device_join_batches;
        ++device_moe_stats.device_commands;
        device_moe_stats.routed_experts += kTopK;
        ++device_moe_stats.shared_experts;
    }
    device_moe_stats.shared_collect_nanoseconds +=
        elapsed_nanoseconds(collect_started);
    device_moe_stats.nanoseconds += elapsed_nanoseconds(collect_started) +
        (device_moe_stats.routed_cpu_nanoseconds -
         host_moe_routed_cpu_before);
    host_moe_pending = 0U;
    host_moe_routed_cpu_before = 0U;
    pending_attention_leases.clear();
    pending_attention_weights.clear();
    for (auto& entry : host_moe_contexts) {
        auto& context = *entry;
        context.shared = {};
        for (auto& lease : context.shared_leases) {
            lease = Dsv4WeightCache::Lease{};
        }
    }
    for (auto& entry : physical_attention_contexts) {
        auto& context = *entry;
        if (!context.result.ok()) {
            append_errors(
                result, std::move(context.result.errors),
                "DeepSeek physical attention callback layer " +
                    std::to_string(context.layer));
        }
        if ((context.sliding_append.has_value() ||
             context.compressed_append.has_value() ||
             context.index_append.has_value()) &&
            !context.invoked) {
            result.errors.emplace_back(
                "DeepSeek physical attention callback was not invoked for layer " +
                std::to_string(context.layer));
        }
        const auto account = [&](auto& append) {
            if (!append.has_value()) return;
            auto accounted = append->account();
            if (!accounted.ok()) {
                append_errors(
                    result, std::move(accounted.errors),
                    "DeepSeek physical attention account layer " +
                        std::to_string(context.layer));
            }
        };
        account(context.sliding_append);
        account(context.compressed_append);
        account(context.index_append);
        context.sliding_append.reset();
        context.compressed_append.reset();
        context.index_append.reset();
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::host_routed_moe_impl(
    std::uint32_t layer, const Dsv4Route* route,
    std::span<const float> input, std::span<float> output,
    std::uint32_t token, std::uint32_t token_position) {
    ValidationResult result;
    constexpr std::size_t shards = 2U;
    constexpr std::size_t shard_intermediate = kExpertIntermediate / shards;
    const bool persistent_device_branch =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    const bool device_input = route == nullptr;
    if (layer >= kLayers ||
        (device_input ? !input.empty() : input.size() != kHidden) ||
        (output.size() != kHidden &&
         !(persistent_device_branch && output.empty())) ||
        (!device_input &&
         (route->experts.size() != kTopK ||
          route->weights.size() != kTopK)) ||
        (device_input && !persistent_device_branch) ||
        expert_workers == nullptr || expert_lane_nodes.size() != expert_workers->size()) {
        result.errors.emplace_back("DeepSeek host-routed MoE state is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    std::array<std::array<Dsv4TiledExpertWeights, kTopK>, shards> tiled{};
    const auto resolve_tiled = [&](const Dsv4Route& active_route) {
        for (std::size_t shard = 0U; shard < shards; ++shard) {
            for (std::size_t rank = 0U; rank < kTopK; ++rank) {
                if (active_route.experts[rank] >= kExperts ||
                    !std::isfinite(active_route.weights[rank])) {
                    result.errors.emplace_back(
                        "DeepSeek host-routed MoE route is invalid");
                    return false;
                }
                auto viewed = dsv4_tiled_expert_weights(
                    resident.find_tiled_expert(
                        layer, active_route.experts[rank],
                        static_cast<std::uint32_t>(shard)),
                    kHidden, kExpertIntermediate, shards);
                if (!viewed.ok()) {
                    append_errors(result, std::move(viewed.errors));
                    return false;
                }
                tiled[shard][rank] = viewed.value;
            }
        }
        return true;
    };
    if (!device_input && !resolve_tiled(*route)) {
        if (result.ok()) {
                result.errors.emplace_back(
                    "DeepSeek host-routed MoE route is invalid");
        }
        return result;
    }

    const auto shared_slot = persistent_device_branch
        ? mhc_slot : layer_device(layer);
    std::array<Dsv4WeightCache::Lease, 3U> shared_leases;
    CudaDeepSeekMoeExpert shared;
    shared.coefficient = 1.0F;
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    const std::array<std::string, 3U> names{
        shared_prefix + "w1", shared_prefix + "w3", shared_prefix + "w2"};
    constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 3U> shapes{{
        {kExpertIntermediate, kHidden}, {kExpertIntermediate, kHidden},
        {kHidden, kExpertIntermediate}}};
    const CudaWeight** shared_weights[]{&shared.w1, &shared.w3, &shared.w2};
    for (std::size_t index = 0U; index < shared_leases.size(); ++index) {
        auto acquired = weights->acquire(shared_slot, names[index],
                                         shapes[index].first,
                                         shapes[index].second,
                                         shared_leases[index]);
        if (!acquired.ok()) {
            append_errors(result, std::move(acquired.errors), names[index]);
            return result;
        }
        *shared_weights[index] = &shared_leases[index].weight();
    }
    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point callback_finished{};
    bool callback_invoked = false;
    bool callback_accepted = false;
    std::vector<float> staged_input(device_input ? kHidden : 0U);
    Dsv4Route staged_route;
    auto active_input = input;
    const Dsv4Route* active_route = route;
    auto routed_callback = [&](std::span<float> rank_partials) {
    callback_invoked = true;
    const auto routed_started = std::chrono::steady_clock::now();
    if (rank_partials.size() != shards * kHidden) {
        result.errors.emplace_back(
            "DeepSeek CPU-MoE callback rank-partial shape is invalid");
        callback_finished = std::chrono::steady_clock::now();
        return false;
    }
    constexpr std::size_t block = 32U;
    constexpr auto intermediate_blocks = shard_intermediate / block;
    auto routed_phase_started = std::chrono::steady_clock::now();
    result = run_expert_ranges(kTopK * intermediate_blocks,
                        [&](std::size_t shard, std::uint64_t task) {
        const auto rank = static_cast<std::size_t>(task / intermediate_blocks);
        const auto offset = (task % intermediate_blocks) * block;
        std::array<float, block> gate{};
        std::array<float, block> up{};
        for (std::size_t half = 0U; half < 2U; ++half) {
            dsv4_tiled_expert_matvec16(
                std::span<float, 16U>(gate.data() + half * 16U, 16U),
                active_input,
                tiled[shard][rank].w13_packed,
                tiled[shard][rank].w13_scales, 2U * shard_intermediate,
                offset + half * 16U);
            dsv4_tiled_expert_matvec16(
                std::span<float, 16U>(up.data() + half * 16U, 16U),
                active_input,
                tiled[shard][rank].w13_packed,
                tiled[shard][rank].w13_scales, 2U * shard_intermediate,
                shard_intermediate + offset + half * 16U);
        }
        auto* destination = tiled_activation.data() +
            (shard * kTopK + rank) * shard_intermediate + offset;
        for (std::size_t index = 0U; index < block; ++index) {
            destination[index] = gate[index] /
                (1.0F + std::exp(-gate[index])) * up[index];
        }
    }, true);
    device_moe_stats.routed_gate_up_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    routed_phase_started = std::chrono::steady_clock::now();
    if (result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        result = run_expert_ranges(kTopK * hidden_blocks,
                            [&](std::size_t shard, std::uint64_t task) {
            const auto rank = static_cast<std::size_t>(task / hidden_blocks);
            const auto offset = (task % hidden_blocks) * block;
            const auto source = std::span<const float>(tiled_activation)
                .subspan((shard * kTopK + rank) * shard_intermediate,
                         shard_intermediate);
            for (std::size_t half = 0U; half < 2U; ++half) {
                auto destination = std::span<float, 16U>(
                    tiled_routed.data() + (shard * kTopK + rank) * kHidden +
                        offset + half * 16U,
                    16U);
                dsv4_tiled_expert_matvec16(
                    destination, source, tiled[shard][rank].w2_packed,
                    tiled[shard][rank].w2_scales, kHidden,
                    offset + half * 16U);
            }
        }, true);
    }
    device_moe_stats.routed_down_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    routed_phase_started = std::chrono::steady_clock::now();
    if (result.ok()) {
        constexpr auto hidden_blocks = kHidden / block;
        result = run_expert_ranges(hidden_blocks,
                            [&](std::size_t shard, std::uint64_t task) {
            const auto offset = task * block;
            auto* destination = rank_partials.data() + shard * kHidden + offset;
            for (std::size_t index = 0U; index < block; ++index) {
                destination[index] = tiled_routed[
                    (shard * kTopK) * kHidden + offset + index] *
                    active_route->weights[0];
            }
            for (std::size_t rank = 1U; rank < kTopK; ++rank) {
                const auto* source = tiled_routed.data() +
                    (shard * kTopK + rank) * kHidden + offset;
                for (std::size_t index = 0U; index < block; ++index) {
                    destination[index] = std::fma(
                        source[index], active_route->weights[rank],
                        destination[index]);
                }
            }
        }, false);
    }
    device_moe_stats.routed_reduce_nanoseconds +=
        elapsed_nanoseconds(routed_phase_started);
    device_moe_stats.routed_cpu_nanoseconds +=
        elapsed_nanoseconds(routed_started);
    callback_finished = std::chrono::steady_clock::now();
    callback_accepted = result.ok();
    return callback_accepted;
    };

    using RoutedCallback = decltype(routed_callback);
    const auto invoke_routed = +[](void* opaque,
                                   std::span<float> rank_partials) {
        return (*static_cast<RoutedCallback*>(opaque))(rank_partials);
    };
    auto device_input_callback = [&nobreak = callback_invoked,
                                  &accepted = callback_accepted,
                                  &finished = callback_finished,
                                  &result, &staged_input, &staged_route,
                                  &active_input, &active_route, &resolve_tiled,
                                  &routed_callback, this, layer, token,
                                  token_position](
        std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials) {
        nobreak = true;
        if (encoded_hidden.size() != kHidden ||
            router_logits.size() != kExperts) {
            result.errors.emplace_back(
                "DeepSeek device-input CPU-MoE callback shape is invalid");
            finished = std::chrono::steady_clock::now();
            accepted = false;
            return false;
        }
        for (std::size_t index = 0U; index < encoded_hidden.size(); ++index) {
            staged_input[index] = std::bit_cast<float>(
                static_cast<std::uint32_t>(encoded_hidden[index]) << 16U);
            if (!std::isfinite(staged_input[index])) {
                result.errors.emplace_back(
                    "DeepSeek device-input CPU-MoE hidden row is non-finite");
                finished = std::chrono::steady_clock::now();
                accepted = false;
                return false;
            }
        }
        const auto route_started = std::chrono::steady_clock::now();
        result = route_moe(
            layer, token, router_logits, token_position, staged_route);
        graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(route_started);
        if (!result.ok() || !resolve_tiled(staged_route)) {
            finished = std::chrono::steady_clock::now();
            accepted = false;
            return false;
        }
        active_input = staged_input;
        active_route = &staged_route;
        return routed_callback(rank_partials);
    };
    using DeviceInputCallback = decltype(device_input_callback);
    const auto invoke_device_input = +[](
        void* opaque, std::span<const std::uint16_t> encoded_hidden,
        std::span<const float> router_logits,
        std::span<float> rank_partials) {
        return (*static_cast<DeviceInputCallback*>(opaque))(
            encoded_hidden, router_logits, rank_partials);
    };
    ValidationResult enqueued;
    if (device_input) {
        enqueued = cuda.enqueue_dsv4_host_moe_from_device_input(
            devices[shared_slot], shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_device_input, &device_input_callback);
    } else if (persistent_device_branch) {
        enqueued = cuda.enqueue_dsv4_host_moe_from_mhc(
            devices[shared_slot], shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_routed, &routed_callback);
    } else {
        enqueued = cuda.enqueue_dsv4_host_moe(
            devices[shared_slot], input, shared,
            kDeepSeekV4ExecutionContract.swiglu_limit,
            invoke_routed, &routed_callback);
    }
    if (!enqueued.ok()) {
        append_errors(result, std::move(enqueued.errors),
                      "DeepSeek CPU/shared MoE enqueue");
        return result;
    }

    std::vector<float> ignored_routed;
    const auto shared_started = std::chrono::steady_clock::now();
    auto device_output = persistent_device_branch &&
                         !config.enable_layer_hash_trace
        ? std::span<float>{} : output;
    auto collected = cuda.collect_deepseek_moe(
        devices[shared_slot], ignored_routed, device_output);
    const auto collected_finished = std::chrono::steady_clock::now();
    if (callback_finished != std::chrono::steady_clock::time_point{} &&
        collected_finished >= callback_finished) {
        device_moe_stats.shared_collect_nanoseconds +=
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    collected_finished - callback_finished).count());
    } else {
        device_moe_stats.shared_collect_nanoseconds +=
            elapsed_nanoseconds(shared_started);
    }
    if (callback_invoked) ++device_moe_stats.host_callback_batches;
    if (!callback_accepted) ++device_moe_stats.host_callback_failures;
    if (!collected.ok()) {
        append_errors(result, std::move(collected.errors),
                      "DeepSeek CPU/shared MoE collect");
    }
    if (!result.ok()) return result;

    ++device_moe_stats.batches;
    ++device_moe_stats.device_join_batches;
    ++device_moe_stats.device_commands;
    device_moe_stats.routed_experts += kTopK;
    ++device_moe_stats.shared_experts;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::device_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (layer >= kLayers || input.size() != kHidden || output.size() != kHidden ||
        route.experts.size() != kTopK || route.weights.size() != kTopK) {
        result.errors.emplace_back("DeepSeek device MoE input or route shape is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    struct RoutePlacement {
        std::size_t slot{};
        std::size_t local_rank{};
    };
    struct PendingDevice {
        std::vector<Dsv4WeightCache::Lease> leases;
        std::vector<CudaDeepSeekMoeExpert> routed;
        CudaDeepSeekMoeExpert shared;
        std::vector<float> routed_output;
        std::vector<float> shared_output;
        bool has_shared{};
        bool enqueued{};
    };

    std::vector<PendingDevice> pending(devices.size());
    for (auto& device : pending) {
        device.leases.reserve((kTopK + 1U) * 3U);
        device.routed.reserve(kTopK);
    }
    std::array<RoutePlacement, kTopK> placements{};

    const auto acquire_triplet = [this, &result](
        std::size_t slot, std::string_view prefix, float coefficient,
        PendingDevice& pending_device, CudaDeepSeekMoeExpert& descriptor) {
        descriptor.coefficient = coefficient;
        const auto acquire = [this, &result, slot, &pending_device](
            std::string name, std::uint64_t rows, std::uint64_t columns,
            const CudaWeight*& weight) {
            pending_device.leases.emplace_back();
            auto loaded = weights->acquire(slot, name, rows, columns,
                                           pending_device.leases.back());
            if (!loaded.ok()) {
                append_errors(result, std::move(loaded.errors), name);
                pending_device.leases.pop_back();
                return false;
            }
            weight = &pending_device.leases.back().weight();
            return true;
        };
        return acquire(std::string(prefix) + "w1", kExpertIntermediate,
                       kHidden, descriptor.w1) &&
               acquire(std::string(prefix) + "w3", kExpertIntermediate,
                       kHidden, descriptor.w3) &&
               acquire(std::string(prefix) + "w2", kHidden,
                       kExpertIntermediate, descriptor.w2);
    };

    // Every acquire below is a candidate demand transfer, and this layer's
    // experts are spread over all three devices. Batching lets those copies
    // run on their links concurrently instead of one at a time; the batch is
    // closed before the first MoE command is enqueued.
    auto upload_batch = config.serial_expert_upload
        ? Dsv4WeightCache::UploadBatch{}
        : weights->begin_upload_batch();

    const auto routed_prefix = layer_prefix(layer) + "ffn.experts.";
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto expert_id = route.experts[rank];
        if (expert_id >= kExperts || !std::isfinite(route.weights[rank])) {
            result.errors.emplace_back(
                "DeepSeek device MoE expert id or coefficient is invalid");
            return result;
        }
        const auto slot = expert_device(expert_id);
        auto& pending_device = pending[slot];
        placements[rank] = {slot, pending_device.routed.size()};
        CudaDeepSeekMoeExpert descriptor;
        const auto prefix = routed_prefix + std::to_string(expert_id) + ".";
        if (!acquire_triplet(slot, prefix, route.weights[rank],
                             pending_device, descriptor)) {
            return result;
        }
        pending_device.routed.push_back(descriptor);
    }

    const auto shared_slot = layer_device(layer);
    auto& shared_device = pending[shared_slot];
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    if (!acquire_triplet(shared_slot, shared_prefix, 1.0F,
                         shared_device, shared_device.shared)) {
        return result;
    }
    shared_device.has_shared = true;

    // Waits out the deferred copies, so every weight below is on its device
    // before any command reads it, and the wait is inside moe_prepare where
    // the serial version paid it.
    if (auto closed = upload_batch.close(); !closed.ok()) {
        append_errors(result, std::move(closed.errors),
                      "DeepSeek routed expert upload");
        return result;
    }

    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    const auto device_commands = static_cast<std::uint64_t>(std::count_if(
        pending.begin(), pending.end(), [](const auto& pending_device) {
            return !pending_device.routed.empty() || pending_device.has_shared;
        }));

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (pending_device.routed.empty() && !pending_device.has_shared) continue;
        pending_device.routed_output.resize(
            pending_device.routed.size() * kHidden);
        if (pending_device.has_shared) {
            pending_device.shared_output.resize(kHidden);
        }
        auto enqueued = cuda.enqueue_deepseek_moe(
            devices[slot], input, pending_device.routed,
            pending_device.has_shared ? &pending_device.shared : nullptr, 10.0F);
        if (!enqueued.ok()) {
            append_errors(result, std::move(enqueued.errors),
                          "DeepSeek device MoE enqueue");
            break;
        }
        pending_device.enqueued = true;
    }

    // Every accepted command must be observed before its cache leases leave
    // scope, including commands submitted before a later-device enqueue error.
    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (!pending_device.enqueued) continue;
        auto collected = cuda.collect_deepseek_moe(
            devices[slot], pending_device.routed_output,
            pending_device.shared_output);
        pending_device.enqueued = false;
        if (!collected.ok()) {
            append_errors(result, std::move(collected.errors),
                          "DeepSeek device MoE collect");
        }
    }
    if (!result.ok()) return result;

    for (auto& pending_device : pending) {
        round_bf16(pending_device.routed_output);
        round_bf16(pending_device.shared_output);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::size_t rank = 0U; rank < kTopK; ++rank) {
        const auto placement = placements[rank];
        const auto routed = std::span<const float>(
            pending[placement.slot].routed_output)
            .subspan(placement.local_rank * kHidden, kHidden);
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(
            output[column] + shared_device.shared_output[column]);
    }
    ++device_moe_stats.batches;
    device_moe_stats.device_commands += device_commands;
    device_moe_stats.routed_experts += kTopK;
    ++device_moe_stats.shared_experts;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::route_moe(
    std::uint32_t layer, std::uint32_t token, std::span<const float> logits,
    std::uint32_t position, Dsv4Route& output) {
    ValidationResult result;
    const auto prefix = layer_prefix(layer) + "ffn.";
    const auto& router = deepseek_v4_flash_0731_spec().router;
    Dsv4RouteResult route;
    if (layer < 3U) {
        const auto name = prefix + "gate.tid2eid";
        const auto found = host_raw.find(name);
        if (found == host_raw.end()) {
            result.errors.emplace_back("DeepSeek resident hash-routing table is absent");
            return result;
        }
        const auto row_bytes = static_cast<std::size_t>(kTopK) * sizeof(std::int64_t);
        const auto offset = static_cast<std::size_t>(token) * row_bytes;
        if (token >= kVocabulary || found->second.size() < offset + row_bytes) {
            result.errors.emplace_back("DeepSeek hash-routing token row is out of range");
            return result;
        }
        std::array<std::uint32_t, kTopK> selected{};
        for (std::uint32_t rank = 0U; rank < kTopK; ++rank) {
            std::int64_t encoded = 0;
            std::memcpy(&encoded,
                        found->second.data() + offset + rank * sizeof(encoded),
                        sizeof(encoded));
            if (encoded < 0 || encoded >= static_cast<std::int64_t>(kExperts)) {
                result.errors.emplace_back("DeepSeek hash-routing expert is invalid");
                return result;
            }
            selected[rank] = static_cast<std::uint32_t>(encoded);
        }
        route = dsv4_route_hash_sqrtsoftplus_f32(logits, selected, router);
    } else {
        auto bias = host_tensor(prefix + "gate.bias", kExperts);
        if (!bias.ok()) {
            append_errors(result, std::move(bias.errors));
            return result;
        }
        route = dsv4_route_sqrtsoftplus_f32(logits, *bias.value, router);
    }
    if (!route.ok()) {
        append_errors(result, std::move(route.errors));
        return result;
    }
    if (config.enable_layer_hash_trace) {
        record_operation_hash(position, token, layer, "ffn_router_weights", route.value.weights);
    }
    const bool prefetch_enabled = config.expert_prefetch_predictions != 0U;
    // A speculative pass is rolled back, so its routes are not part of the
    // sequence. Recording them would put tokens that were never emitted into
    // the trace and teach the predictor a history that did not happen.
    if (!speculative_pass && (route_trace.is_open() || prefetch_enabled)) {
        RouteEvent event;
        event.request = active_request_id;
        event.token_position = position;
        event.layer = layer;
        event.experts = route.value.experts;
        event.coefficients = route.value.weights;
        event.phase = position < active_prompt_tokens
                          ? RoutePhase::Prefill : RoutePhase::Decode;
        if (defer_prefill_observability && event.phase == RoutePhase::Prefill) {
            deferred_route_events.push_back(std::move(event));
        } else {
            if (prefetch_enabled) {
                route_predictor.observe(event);
                if (event.phase == RoutePhase::Decode) {
                    pending_prefetch_predictions = route_predictor.predict(
                        event, config.expert_prefetch_predictions,
                        config.expert_prefetch_minimum_confidence);
                }
            }
            if (route_trace.is_open()) {
                auto written = route_trace.write(event);
                if (!written.ok()) return written;
            }
        }
    }
    output = std::move(route.value);
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::execute_moe(
    std::uint32_t layer, const Dsv4Route& route,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    if (config.enable_host_routed_moe) {
        return host_routed_moe(layer, route, input, output);
    }
    const auto prefix = layer_prefix(layer) + "ffn.";
    std::vector<ExpertKey> demand_keys;
    demand_keys.reserve(route.experts.size());
    for (const auto expert_id : route.experts) {
        demand_keys.push_back(ExpertKey{layer, expert_id});
    }
    auto demand_guard = weights->demand(demand_keys);
    const auto schedule_prefetch = [this] {
        for (const auto& prediction : pending_prefetch_predictions) {
            weights->request_prefetch(
                prediction.key, expert_device(prediction.key.expert));
        }
        pending_prefetch_predictions.clear();
    };

    if (config.enable_device_moe) {
        result = device_moe(layer, route, input, output);
        if (result.ok()) schedule_prefetch();
        return result;
    }

    std::fill(output.begin(), output.end(), 0.0F);
    std::vector<float> routed(kHidden);
    for (std::size_t rank = 0U; rank < route.experts.size(); ++rank) {
        result = expert(layer, route.experts[rank], route.weights[rank], input,
                        routed);
        if (!result.ok()) return result;
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output[column] += routed[column];
        }
    }

    const auto slot = layer_device(layer);
    std::vector<float> shared_gate(kExpertIntermediate);
    std::vector<float> shared_up(kExpertIntermediate);
    std::vector<float> shared_activated(kExpertIntermediate);
    std::vector<float> shared_output(kHidden);
    result = linear(slot, prefix + "shared_experts.w1", kExpertIntermediate,
                    kHidden, input, shared_gate);
    if (!result.ok()) return result;
    result = linear(slot, prefix + "shared_experts.w3", kExpertIntermediate,
                    kHidden, input, shared_up);
    if (!result.ok()) return result;
    result = dsv4_swiglu_f32(shared_activated, shared_gate, shared_up, 10.0F);
    if (!result.ok()) return result;
    round_bf16(shared_activated);
    result = linear(slot, prefix + "shared_experts.w2", kHidden,
                    kExpertIntermediate, shared_activated, shared_output);
    if (!result.ok()) return result;
    for (std::uint32_t column = 0U; column < kHidden; ++column) {
        output[column] = round_bf16(output[column] + shared_output[column]);
    }
    schedule_prefetch();
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::moe(
    std::uint32_t layer, std::uint32_t token, std::span<const float> input,
    std::span<float> output, std::uint32_t position) {
    ValidationResult result;
    const auto router_started = std::chrono::steady_clock::now();
    if (completed_router_projection && deferred_attention_moe_input) {
        completed_router_projection = false;
        deferred_attention_moe_input = false;
        return host_routed_moe_from_device_input(
            layer, token, position, output);
    }
    std::vector<float> logits(kExperts);
    if (completed_router_projection) {
        std::copy(combined_router_logits.begin(), combined_router_logits.end(),
                  logits.begin());
        completed_router_projection = false;
    } else {
        result = linear(layer_device(layer), layer_prefix(layer) + "ffn.gate",
                        kExperts, kHidden, input, logits, false);
        if (!result.ok()) return result;
    }
    Dsv4Route route;
    result = route_moe(layer, token, logits, position, route);
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(router_started);
    if (!result.ok()) return result;
    return execute_moe(layer, route, input, output);
}

ValidationResult DeepSeekV4Runtime::Impl::moe_page(
    std::uint32_t layer, std::span<const std::uint32_t> tokens,
    std::span<const float> input, std::span<float> output,
    std::uint32_t position_base) {
    ValidationResult result;
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    if (rows == 0U || input.size() != static_cast<std::size_t>(rows) * kHidden ||
        output.size() != input.size()) {
        result.errors.emplace_back("DeepSeek MoE page has incompatible dimensions");
        return result;
    }
    const auto router_started = std::chrono::steady_clock::now();
    std::vector<float> logits(static_cast<std::size_t>(rows) * kExperts);
    result = linear_rows(layer_device(layer), layer_prefix(layer) + "ffn.gate",
                         kExperts, kHidden, input, rows, logits, false,
                         nullptr, config.enable_dsv4_fp8_tensor_page);
    if (!result.ok()) return result;
    std::vector<Dsv4Route> routes(rows);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        result = route_moe(
            layer, tokens[row],
            std::span<const float>(logits).subspan(
                static_cast<std::size_t>(row) * kExperts, kExperts),
            position_base + row, routes[row]);
        if (!result.ok()) return result;
    }
    graph_stats.moe_router_nanoseconds += elapsed_nanoseconds(router_started);
    if (rows == 1U || config.row_major_moe_page) {
        for (std::uint32_t row = 0U; row < rows; ++row) {
            result = execute_moe(
                layer, routes[row],
                input.subspan(static_cast<std::size_t>(row) * kHidden, kHidden),
                output.subspan(static_cast<std::size_t>(row) * kHidden, kHidden));
            if (!result.ok()) return result;
        }
        return result;
    }
    return execute_moe_page(layer, routes, input, output);
}

// Row-grouped MoE for a prefill page. The single-row path reads a 13.37 MB
// expert triplet from HBM to serve one row, so a page of R rows reads a hot
// expert once for every row that chose it. Here each distinct expert is
// acquired and read once and applied to all its rows at once, which is the
// only structural difference between this engine's prefill and a CPU-hybrid
// stack that batches the same page 30x faster (experiment 0055).
//
// Per-row arithmetic is unchanged: every row still visits its own six experts
// with its own coefficients, and the final accumulation still runs in rank
// order, so output is bit-identical to looping execute_moe over the rows.
ValidationResult DeepSeekV4Runtime::Impl::execute_moe_page(
    std::uint32_t layer, std::span<const Dsv4Route> routes,
    std::span<const float> input, std::span<float> output) {
    ValidationResult result;
    const auto rows = static_cast<std::uint32_t>(routes.size());
    if (rows == 0U || input.size() != static_cast<std::size_t>(rows) * kHidden ||
        output.size() != input.size()) {
        result.errors.emplace_back("DeepSeek MoE page shape is invalid");
        return result;
    }

    const auto prepare_started = std::chrono::steady_clock::now();
    struct PendingDevice {
        std::vector<Dsv4WeightCache::Lease> leases;
        std::vector<CudaDeepSeekMoeRowGroup> groups;
        std::deque<std::vector<std::uint32_t>> group_rows;
        std::deque<std::vector<float>> group_coefficients;
        std::vector<std::uint32_t> group_offsets;
        std::unordered_map<std::uint32_t, std::size_t> expert_slot;
        CudaDeepSeekMoeExpert shared;
        std::vector<std::uint32_t> shared_rows;
        std::vector<float> routed_output;
        std::vector<float> shared_output;
        std::uint32_t work_count{};
        bool has_shared{};
        bool enqueued{};
    };
    struct Placement {
        std::uint32_t slot{};
        std::uint32_t group{};
        std::uint32_t position{};
    };

    std::vector<PendingDevice> pending(devices.size());
    std::vector<Placement> placements(
        static_cast<std::size_t>(rows) * kTopK);

    const auto acquire_triplet = [this, &result](
        std::size_t slot, std::string_view prefix,
        PendingDevice& pending_device, CudaDeepSeekMoeExpert& descriptor) {
        descriptor.coefficient = 1.0F;
        const auto acquire = [this, &result, slot, &pending_device](
            std::string name, std::uint64_t weight_rows,
            std::uint64_t weight_columns, const CudaWeight*& weight) {
            pending_device.leases.emplace_back();
            auto loaded = weights->acquire(slot, name, weight_rows,
                                           weight_columns,
                                           pending_device.leases.back());
            if (!loaded.ok()) {
                append_errors(result, std::move(loaded.errors), name);
                pending_device.leases.pop_back();
                return false;
            }
            weight = &pending_device.leases.back().weight();
            return true;
        };
        return acquire(std::string(prefix) + "w1", kExpertIntermediate,
                       kHidden, descriptor.w1) &&
               acquire(std::string(prefix) + "w3", kExpertIntermediate,
                       kHidden, descriptor.w3) &&
               acquire(std::string(prefix) + "w2", kHidden,
                       kExpertIntermediate, descriptor.w2);
    };

    auto upload_batch = config.serial_expert_upload
        ? Dsv4WeightCache::UploadBatch{}
        : weights->begin_upload_batch();

    const auto routed_prefix = layer_prefix(layer) + "ffn.experts.";
    for (std::uint32_t row = 0U; row < rows; ++row) {
        const auto& route = routes[row];
        if (route.experts.size() != kTopK || route.weights.size() != kTopK) {
            result.errors.emplace_back("DeepSeek MoE page route shape is invalid");
            return result;
        }
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto expert_id = route.experts[rank];
            if (expert_id >= kExperts || !std::isfinite(route.weights[rank])) {
                result.errors.emplace_back(
                    "DeepSeek MoE page expert id or coefficient is invalid");
                return result;
            }
            const auto slot = expert_device(expert_id);
            auto& pending_device = pending[slot];
            auto found = pending_device.expert_slot.find(expert_id);
            if (found == pending_device.expert_slot.end()) {
                CudaDeepSeekMoeRowGroup group;
                if (resident.tiled_experts()) {
                    // The transformed shards are the resident copy, so this
                    // uploads them as they stand. Reading the canonical
                    // triplet instead means re-reading the checkpoint for
                    // weights that are already in host memory.
                    for (std::uint32_t shard = 0U;
                         shard < kResidentExpertShards; ++shard) {
                        pending_device.leases.emplace_back();
                        auto acquired = weights->acquire_tiled_expert(
                            slot, layer, expert_id, shard,
                            pending_device.leases.back());
                        if (!acquired.ok()) {
                            append_errors(result, std::move(acquired.errors),
                                          "DeepSeek transformed expert shard");
                            pending_device.leases.pop_back();
                            return result;
                        }
                        group.tiled_shards[shard] =
                            &pending_device.leases.back().weight();
                    }
                } else {
                    CudaDeepSeekMoeExpert descriptor;
                    const auto prefix =
                        routed_prefix + std::to_string(expert_id) + ".";
                    if (!acquire_triplet(slot, prefix, pending_device,
                                         descriptor)) {
                        return result;
                    }
                    group.w1 = descriptor.w1;
                    group.w3 = descriptor.w3;
                    group.w2 = descriptor.w2;
                }
                const auto group_index = pending_device.groups.size();
                pending_device.group_rows.emplace_back();
                pending_device.group_coefficients.emplace_back();
                pending_device.groups.push_back(group);
                found = pending_device.expert_slot
                            .emplace(expert_id, group_index).first;
            }
            auto& group_rows = pending_device.group_rows[found->second];
            auto& group_coefficients =
                pending_device.group_coefficients[found->second];
            placements[static_cast<std::size_t>(row) * kTopK + rank] = {
                static_cast<std::uint32_t>(slot),
                static_cast<std::uint32_t>(found->second),
                static_cast<std::uint32_t>(group_rows.size())};
            group_rows.push_back(row);
            group_coefficients.push_back(route.weights[rank]);
        }
    }

    const auto shared_slot = layer_device(layer);
    auto& shared_device = pending[shared_slot];
    const auto shared_prefix = layer_prefix(layer) + "ffn.shared_experts.";
    if (!acquire_triplet(shared_slot, shared_prefix, shared_device,
                         shared_device.shared)) {
        return result;
    }
    shared_device.has_shared = true;
    shared_device.shared_rows.resize(rows);
    std::iota(shared_device.shared_rows.begin(),
              shared_device.shared_rows.end(), 0U);

    if (auto closed = upload_batch.close(); !closed.ok()) {
        append_errors(result, std::move(closed.errors),
                      "DeepSeek routed expert page upload");
        return result;
    }

    graph_stats.moe_prepare_nanoseconds += elapsed_nanoseconds(prepare_started);
    const auto execution_started = std::chrono::steady_clock::now();
    std::uint64_t device_commands = 0U;
    for (auto& pending_device : pending) {
        if (pending_device.groups.empty() && !pending_device.has_shared) continue;
        std::uint32_t offset = 0U;
        pending_device.group_offsets.reserve(pending_device.groups.size());
        for (std::size_t index = 0U; index < pending_device.groups.size();
             ++index) {
            pending_device.group_offsets.push_back(offset);
            pending_device.groups[index].rows =
                pending_device.group_rows[index];
            pending_device.groups[index].coefficients =
                pending_device.group_coefficients[index];
            offset += static_cast<std::uint32_t>(
                pending_device.group_rows[index].size());
        }
        pending_device.work_count = offset;
        pending_device.routed_output.resize(
            static_cast<std::size_t>(offset) * kHidden);
        if (pending_device.has_shared) {
            pending_device.shared_output.resize(
                static_cast<std::size_t>(rows) * kHidden);
        }
        ++device_commands;
    }

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (pending_device.groups.empty() && !pending_device.has_shared) continue;
        auto enqueued = cuda.enqueue_deepseek_moe_rows(
            devices[slot], input, rows, pending_device.groups,
            pending_device.has_shared ? &pending_device.shared : nullptr,
            pending_device.shared_rows, 10.0F);
        if (!enqueued.ok()) {
            append_errors(result, std::move(enqueued.errors),
                          "DeepSeek device MoE page enqueue");
            break;
        }
        pending_device.enqueued = true;
    }

    for (std::size_t slot = 0U; slot < pending.size(); ++slot) {
        auto& pending_device = pending[slot];
        if (!pending_device.enqueued) continue;
        auto collected = cuda.collect_deepseek_moe_rows(
            devices[slot], pending_device.routed_output,
            pending_device.shared_output);
        pending_device.enqueued = false;
        if (!collected.ok()) {
            append_errors(result, std::move(collected.errors),
                          "DeepSeek device MoE page collect");
        }
    }
    if (!result.ok()) return result;

    for (auto& pending_device : pending) {
        round_bf16(pending_device.routed_output);
        round_bf16(pending_device.shared_output);
    }
    std::fill(output.begin(), output.end(), 0.0F);
    for (std::uint32_t row = 0U; row < rows; ++row) {
        auto output_row = output.subspan(
            static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::size_t rank = 0U; rank < kTopK; ++rank) {
            const auto placement =
                placements[static_cast<std::size_t>(row) * kTopK + rank];
            const auto& pending_device = pending[placement.slot];
            const auto work_index =
                pending_device.group_offsets[placement.group] +
                placement.position;
            const auto routed = std::span<const float>(
                pending_device.routed_output)
                .subspan(static_cast<std::size_t>(work_index) * kHidden, kHidden);
            for (std::uint32_t column = 0U; column < kHidden; ++column) {
                output_row[column] += routed[column];
            }
        }
        const auto shared = std::span<const float>(shared_device.shared_output)
            .subspan(static_cast<std::size_t>(row) * kHidden, kHidden);
        for (std::uint32_t column = 0U; column < kHidden; ++column) {
            output_row[column] = round_bf16(output_row[column] + shared[column]);
        }
    }

    ++device_moe_stats.batches;
    device_moe_stats.device_commands += device_commands;
    device_moe_stats.routed_experts +=
        static_cast<std::uint64_t>(rows) * kTopK;
    device_moe_stats.shared_experts += rows;
    device_moe_stats.nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - execution_started).count());
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::block(
    std::uint32_t layer, std::uint32_t token, std::span<float> hidden,
    std::uint32_t position) {
    ValidationResult result;
    if (hidden.size() != static_cast<std::size_t>(kMhc) * kHidden) {
        result.errors.emplace_back("DeepSeek mHC hidden state has the wrong shape");
        return result;
    }
    const auto prefix = layer_prefix(layer);
    for (const auto* branch_name : {"attn", "ffn"}) {
        const std::string branch(branch_name);
        const auto projection_name = prefix + "hc_" + branch + "_fn";
        auto projection = host_tensor(projection_name,
                                      kMix * kMhc * kHidden);
        auto scale = host_tensor(prefix + "hc_" + branch + "_scale", 3U);
        auto base = host_tensor(prefix + "hc_" + branch + "_base", kMix);
        if (!projection.ok()) append_errors(result, std::move(projection.errors));
        if (!scale.ok()) append_errors(result, std::move(scale.errors));
        if (!base.ok()) append_errors(result, std::move(base.errors));
        if (!result.ok()) return result;
        const std::vector<float> residual(hidden.begin(), hidden.end());
        std::vector<float> reduced(kHidden);
        Dsv4MhcMix mix;
        auto phase_started = std::chrono::steady_clock::now();
        result = mhc_pre(reduced, mix, residual, projection_name,
                         *projection.value, *scale.value, *base.value, true);
        graph_stats.mhc_pre_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        round_bf16(reduced);
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_mhc_pre", reduced);
        }
        phase_started = std::chrono::steady_clock::now();
        result = norm(reduced, reduced, prefix + branch + "_norm.weight");
        graph_stats.branch_norm_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_norm", reduced);
        }
        std::vector<float> branch_output(kHidden);
        phase_started = std::chrono::steady_clock::now();
        if (branch == "attn") {
            result = attention(layer, reduced, position, branch_output);
            graph_stats.attention_nanoseconds += elapsed_nanoseconds(phase_started);
        } else {
            result = moe(layer, token, reduced, branch_output, position);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
        }
        if (!result.ok()) return result;
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_output", branch_output);
        }
        phase_started = std::chrono::steady_clock::now();
        result = dsv4_mhc_post_f32(hidden, branch_output, residual, mix);
        graph_stats.mhc_post_nanoseconds += elapsed_nanoseconds(phase_started);
        if (!result.ok()) return result;
        round_bf16(hidden);
        if (config.enable_layer_hash_trace) {
            record_operation_hash(position, token, layer, branch + "_mhc_post", hidden);
        }
    }
    return result;
}

ValidationResult DeepSeekV4Runtime::Impl::block_page(
    std::uint32_t layer, std::span<const std::uint32_t> tokens,
    std::span<float> hidden, std::uint32_t position_base) {
    ValidationResult result;
    const auto rows = tokens.size();
    const auto hidden_stride = static_cast<std::size_t>(kMhc) * kHidden;
    if (rows == 0U || rows > config.maximum_context_tokens ||
        hidden.size() != rows * hidden_stride ||
        position_base > config.maximum_context_tokens - rows) {
        result.errors.emplace_back(
            "DeepSeek prefill page has incompatible dimensions");
        return result;
    }
    const auto prefix = layer_prefix(layer);
    for (const auto* branch_name : {"attn", "ffn"}) {
        const std::string branch(branch_name);
        const auto projection_name = prefix + "hc_" + branch + "_fn";
        auto projection = host_tensor(projection_name,
                                      kMix * kMhc * kHidden);
        auto scale = host_tensor(prefix + "hc_" + branch + "_scale", 3U);
        auto base = host_tensor(prefix + "hc_" + branch + "_base", kMix);
        if (!projection.ok()) append_errors(result, std::move(projection.errors));
        if (!scale.ok()) append_errors(result, std::move(scale.errors));
        if (!base.ok()) append_errors(result, std::move(base.errors));
        if (!result.ok()) return result;

        const std::vector<float> residual(hidden.begin(), hidden.end());
        std::vector<float> reduced(rows * kHidden);
        std::vector<Dsv4MhcMix> mixes(rows);
        for (std::size_t row = 0U; row < rows; ++row) {
            const auto position = position_base + static_cast<std::uint32_t>(row);
            auto reduced_row = std::span<float>(reduced).subspan(row * kHidden,
                                                                 kHidden);
            const auto residual_row = std::span<const float>(residual).subspan(
                row * hidden_stride, hidden_stride);
            auto phase_started = std::chrono::steady_clock::now();
            result = mhc_pre(reduced_row, mixes[row], residual_row,
                             projection_name, *projection.value,
                             *scale.value, *base.value);
            graph_stats.mhc_pre_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            round_bf16(reduced_row);
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_pre", reduced_row);
            }
            phase_started = std::chrono::steady_clock::now();
            result = norm(reduced_row, reduced_row,
                          prefix + branch + "_norm.weight");
            graph_stats.branch_norm_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_norm", reduced_row);
            }
        }

        std::vector<float> branch_output(rows * kHidden);
        if (branch == "attn") {
            const auto phase_started = std::chrono::steady_clock::now();
            result = attention_page(layer, reduced, position_base,
                                    branch_output);
            graph_stats.attention_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        } else {
            const auto phase_started = std::chrono::steady_clock::now();
            result = moe_page(layer, tokens, reduced, branch_output,
                              position_base);
            graph_stats.moe_nanoseconds += elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
        }
        if (config.enable_layer_hash_trace) {
            for (std::size_t row = 0U; row < rows; ++row) {
                const auto position = position_base +
                                      static_cast<std::uint32_t>(row);
                const auto output_row = std::span<const float>(branch_output)
                    .subspan(row * kHidden, kHidden);
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_output", output_row);
            }
        }

        for (std::size_t row = 0U; row < rows; ++row) {
            const auto position = position_base + static_cast<std::uint32_t>(row);
            auto hidden_row = hidden.subspan(row * hidden_stride, hidden_stride);
            const auto output_row = std::span<const float>(branch_output).subspan(
                row * kHidden, kHidden);
            const auto residual_row = std::span<const float>(residual).subspan(
                row * hidden_stride, hidden_stride);
            const auto phase_started = std::chrono::steady_clock::now();
            result = dsv4_mhc_post_f32(hidden_row, output_row, residual_row,
                                       mixes[row]);
            graph_stats.mhc_post_nanoseconds +=
                elapsed_nanoseconds(phase_started);
            if (!result.ok()) return result;
            round_bf16(hidden_row);
            if (config.enable_layer_hash_trace) {
                record_operation_hash(position, tokens[row], layer,
                                      branch + "_mhc_post", hidden_row);
            }
        }
    }
    return result;
}

