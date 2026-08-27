DeepSeekV4Runtime::DeepSeekV4Runtime() : impl_(std::make_unique<Impl>()) {}
DeepSeekV4Runtime::~DeepSeekV4Runtime() = default;
DeepSeekV4Runtime::DeepSeekV4Runtime(DeepSeekV4Runtime&&) noexcept = default;
DeepSeekV4Runtime& DeepSeekV4Runtime::operator=(DeepSeekV4Runtime&&) noexcept = default;

ValidationResult DeepSeekV4Runtime::initialize(
    const std::string& model_directory, const Dsv4RuntimeConfig& caller_config) {
    // Every zero-means-probe field is filled in here, once, so nothing below
    // this line has to know which numbers came from the caller and which from
    // the machine.
    Dsv4RuntimeConfig config(caller_config);
    resolve_hardware_defaults(config);
    ValidationResult result;
    const auto initialization_started = std::chrono::steady_clock::now();
    if (impl_->initialized) {
        result.errors.emplace_back("DeepSeek runtime is already initialized");
        return result;
    }
    // Pure request validation must not depend on whether this build or host
    // exposes CUDA. This also lets admission-only and CPU CI report the same
    // model-contract error as a CUDA workstation.
    const auto model_context =
        deepseek_v4_flash_0731_spec().max_context_tokens;
    if (config.maximum_context_tokens == 0U ||
        config.maximum_context_tokens > model_context) {
        result.errors.emplace_back(
            "DeepSeek runtime context must be within the model limit [1, " +
            std::to_string(model_context) + "] tokens");
        return result;
    }
    result = validate_common_runtime_config(
        config.devices, config.vram_cache_fraction,
        config.sampling_temperature, "DeepSeek");
    if (!result.ok()) return result;
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        // Fail closed before the checkpoint is opened or any weight is
        // resident. Conditions that need the manifest are re-checked at
        // admission; these are the ones knowable from the build and the
        // request alone, and they must not cost a model load to discover.
#if !defined(STRATA_HAS_NCCL)
        result.errors.emplace_back(
            "rank-local decode was requested but this build has no NCCL "
            "support; rebuild with -DSTRATA_ENABLE_NCCL=ON");
#endif
        if (config.devices.size() != kDsv4RankLocalWorld) {
            result.errors.emplace_back(
                "rank-local decode requires exactly " +
                std::to_string(kDsv4RankLocalWorld) + " CUDA devices, got " +
                std::to_string(config.devices.size()));
        }
        if (config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice) {
            result.errors.emplace_back(
                "rank-local decode requires the physical-device DSV4 KV mode");
        }
        std::array<std::vector<int>, kDsv4RankLocalWorld> rank_cpus;
        auto cpu_plan = plan_dsv4_rank_local_cpus(
            NumaTopology::detect(), kDsv4RankLocalMinimumCpusPerRank,
            rank_cpus);
        if (!cpu_plan.ok()) {
            result.errors.insert(result.errors.end(), cpu_plan.errors.begin(),
                                 cpu_plan.errors.end());
        }
        if (!result.ok()) return result;
    }
    if (config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle &&
        config.kv_block_rows == 0U) {
        result.errors.emplace_back(
            "DeepSeek KV block row count must be positive");
        return result;
    }
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice &&
        config.kv_block_rows != kDsv4PhysicalKvBlockRows) {
        result.errors.emplace_back(
            "DeepSeek physical KV requires 256-source-token blocks");
        return result;
    }
    if (config.enable_gpu_lightning_indexer &&
        config.kv_cache_mode != Dsv4KvCacheMode::Block) {
        result.errors.emplace_back(
            "GPU Lightning Indexer requires the exact compact block KV cache");
        return result;
    }
    if (!config.device_kv_cache_bytes.empty() &&
        config.device_kv_cache_bytes.size() != config.devices.size()) {
        result.errors.emplace_back(
            "DeepSeek KV device budget count must match the device count");
        return result;
    }
    if (config.prefill_page_tokens == 0U ||
        config.prefill_page_tokens > kMaximumPrefillPageTokens) {
        result.errors.emplace_back(
            "DeepSeek prefill page must be within [1, 8192] tokens");
        return result;
    }
    if (config.prefill_layer_tile_tokens != 0U &&
        (config.prefill_layer_tile_tokens < config.prefill_page_tokens ||
         config.prefill_layer_tile_tokens > config.maximum_context_tokens)) {
        result.errors.emplace_back(
            "DeepSeek prefill layer tile must be zero or within "
            "[prefill page tokens, maximum context tokens]");
        return result;
    }
    if (config.enable_logit_trace &&
        (config.logit_trace_top_k == 0U ||
         config.logit_trace_top_k > kVocabulary)) {
        result.errors.emplace_back(
            "DeepSeek logit trace top-K must be within [1, 129280]");
        return result;
    }
    if (config.host_attention_threads > kHeads) {
        result.errors.emplace_back(
            "DeepSeek host attention worker count must not exceed 64");
        return result;
    }
    if (config.resident_read_workers == 0U ||
        config.resident_read_workers > 64U) {
        result.errors.emplace_back(
            "DeepSeek resident read worker count must be within [1, 64]");
        return result;
    }
    if (config.spine_warmup_workers == 0U ||
        config.spine_warmup_workers > 64U) {
        result.errors.emplace_back(
            "DeepSeek spine warmup worker count must be within [1, 64]");
        return result;
    }
    if (config.expert_prefetch_predictions > kExperts ||
        !std::isfinite(config.expert_prefetch_minimum_confidence) ||
        config.expert_prefetch_minimum_confidence < 0.0 ||
        config.expert_prefetch_minimum_confidence > 1.0 ||
        (config.expert_prefetch_predictions != 0U &&
         (config.expert_prefetch_queue_depth == 0U ||
          config.expert_prefetch_queue_depth > 1024U ||
          config.expert_prefetch_byte_budget == 0U ||
          config.expert_prefetch_lease_ticks == 0U))) {
        result.errors.emplace_back(
            "DeepSeek expert prefetch requires bounded predictions, bytes, queue, "
            "lease, and confidence");
        return result;
    }
    if (config.enable_host_routed_moe &&
        config.expert_prefetch_predictions != 0U) {
        result.errors.emplace_back(
            "DeepSeek host-routed MoE replaces routed GPU prefetch");
        return result;
    }
    if (config.enable_dspark) {
        result.errors.emplace_back(
            "DSpark tensors are verified, but speculative execution is not enabled in "
            "the base-model executor; refusing a silent approximation");
        return result;
    }
    if (config.prepack_mhc_projection &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice &&
        !dsv4_mhc_prepacked_supported()) {
        result.errors.emplace_back(
            "DeepSeek prepacked mHC requires x86 AVX2");
        return result;
    }
    impl_ = std::make_unique<Impl>();
    auto checkpoint = Dsv4CheckpointReader::open(model_directory);
    if (!checkpoint.ok()) {
        result.errors = std::move(checkpoint.errors);
        return result;
    }
    auto tokenizer = ModelTokenizer::load(
        (std::filesystem::path(model_directory) / "tokenizer.json").string());
    if (!tokenizer.ok()) {
        result.errors = std::move(tokenizer.errors);
        return result;
    }
    // The static expert tier lives on a device outside the execution set, so
    // it needs its own context and weight arena. It is included here rather
    // than in config.devices because it must not join the layer schedule, the
    // KV placement or the rank pair -- it only holds weights and computes the
    // experts it holds.
    std::vector<int> context_devices(config.devices.begin(), config.devices.end());
    const bool tier_requested = !config.static_expert_plan_path.empty() &&
                                config.static_expert_tier_device >= 0;
    if (tier_requested) {
        const auto tier_device = config.static_expert_tier_device;
        if (std::find(context_devices.begin(), context_devices.end(),
                      tier_device) != context_devices.end()) {
            result.errors.emplace_back(
                "static expert tier device " + std::to_string(tier_device) +
                " is already an execution device; the tier must be separate");
            return result;
        }
        context_devices.push_back(tier_device);
    }
    result = impl_->cuda.initialize(context_devices, config.detailed_timing);
    if (!result.ok()) return result;
    if (tier_requested && config.static_expert_tier_bytes != 0U) {
        // Reserve up front: the tier allocates thousands of small weights and
        // the arena refuses a per-weight fallback once enabled.
        result = impl_->cuda.reserve_weight_arena(
            config.static_expert_tier_device,
            config.static_expert_tier_bytes);
        if (!result.ok()) return result;
    }
    if (config.enable_flash_attention) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_flash_attention_device(device);
            if (!result.ok()) return result;
        }
    }
    if (config.enable_gpu_lightning_indexer) {
        for (const int device : config.devices) {
            result = impl_->cuda.validate_lightning_index_device(device);
            if (!result.ok()) return result;
        }
    }

    auto effective_explicit_vram_budget = config.explicit_vram_budget_bytes;
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        impl_->rank_local_initial_device_vram_bytes =
            device_vram_used_bytes(config.devices);
        if (impl_->rank_local_initial_device_vram_bytes.size() !=
            config.devices.size()) {
            result.errors.emplace_back(
                "rank-local initial VRAM ledger does not match devices");
            return result;
        }
        for (std::size_t slot = 0U;
             slot < impl_->rank_local_initial_device_vram_bytes.size();
             ++slot) {
            const auto initial =
                impl_->rank_local_initial_device_vram_bytes[slot];
            const auto ceiling = rank_local_vram_ceiling(config.devices[slot]);
            if (initial >= ceiling) {
                result.errors.emplace_back(
                    "rank-local CUDA device " +
                    std::to_string(config.devices[slot]) +
                    " already uses " + std::to_string(initial) +
                    " B, which leaves no room below the " +
                    std::to_string(ceiling) + " B program ceiling");
                return result;
            }
            const auto available = ceiling - initial;
            effective_explicit_vram_budget =
                effective_explicit_vram_budget == 0U
                    ? available
                    : std::min(effective_explicit_vram_budget, available);
        }
    }
    auto device_plan = plan_runtime_devices(
        config.devices, config.vram_cache_fraction, kDeviceWorkspaceReserve,
        2ULL << 30U, "DeepSeek", effective_explicit_vram_budget);
    if (!device_plan.ok()) {
        result.errors = std::move(device_plan.errors);
        return result;
    }
    auto capacities = std::move(device_plan.value.budgets);
    auto weight_capacities = std::move(device_plan.value.weight_capacities);
    auto kv_device_capacities = config.device_kv_cache_bytes;
    if (kv_device_capacities.empty()) {
        kv_device_capacities.resize(config.devices.size());
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
                auto physical = dsv4_physical_kv_admission(
                    config.maximum_context_tokens);
                if (!physical.ok()) {
                    result.errors = std::move(physical.errors);
                    return result;
                }
                std::fill(kv_device_capacities.begin(),
                          kv_device_capacities.end(),
                          physical.value.payload_bytes);
            } else {
                const auto add_pages = [&](std::size_t slot,
                                           Dsv4KvBlockKind kind,
                                           std::uint32_t ratio,
                                           std::uint64_t rows) {
                    const auto capacity_rows = dsv4_kv_block_rows(
                        kind, ratio, true);
                    const auto format = dsv4_kv_format(kind, false, true);
                    const auto page_bytes = dsv4_kv_row_bytes(kind, format) *
                                            capacity_rows;
                    const auto pages = (rows + capacity_rows - 1U) /
                                       capacity_rows;
                    if (capacity_rows == 0U || page_bytes == 0U ||
                        pages > std::numeric_limits<std::uint64_t>::max() /
                                    page_bytes ||
                        pages * page_bytes >
                            std::numeric_limits<std::uint64_t>::max() -
                                kv_device_capacities[slot]) {
                        return false;
                    }
                    kv_device_capacities[slot] += pages * page_bytes;
                    return true;
                };
                const auto& ratios =
                    kDeepSeekV4ExecutionContract.compression_ratios;
                for (std::uint32_t layer = 0U; layer < kLayers; ++layer) {
                    const auto slot = device_plan.value.weighted_schedule[
                        layer % device_plan.value.weighted_schedule.size()];
                    const auto sliding_rows = std::min<std::uint64_t>(
                        config.maximum_context_tokens,
                        kWindow + kDsv4PhysicalKvBlockRows - 1U);
                    const auto ratio = ratios[layer];
                    bool admitted = add_pages(
                        slot, Dsv4KvBlockKind::Sliding, 1U, sliding_rows);
                    if (admitted && ratio != 0U) {
                        const auto compressed_rows =
                            (static_cast<std::uint64_t>(
                                 config.maximum_context_tokens) + ratio - 1U) /
                            ratio;
                        admitted = add_pages(
                            slot, ratio == 4U ? Dsv4KvBlockKind::Csa
                                             : Dsv4KvBlockKind::Hca,
                            ratio, compressed_rows);
                        if (admitted && ratio == 4U &&
                            config.maximum_context_tokens >
                                kIndexTopK * ratio) {
                            admitted = add_pages(
                                slot, Dsv4KvBlockKind::LearnedIndex, ratio,
                                compressed_rows);
                        }
                    }
                    if (!admitted) {
                        result.errors.emplace_back(
                            "DeepSeek physical KV capacity overflows");
                        return result;
                    }
                }
            }
        }
    }
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2 &&
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        auto physical = dsv4_physical_kv_admission(
            config.maximum_context_tokens);
        if (!physical.ok()) {
            result.errors = std::move(physical.errors);
            return result;
        }
        for (std::size_t slot = 0U; slot < kv_device_capacities.size();
             ++slot) {
            if (kv_device_capacities[slot] < physical.value.payload_bytes) {
                result.errors.emplace_back(
                    "rank-local CUDA device " +
                    std::to_string(config.devices[slot]) +
                    " KV capacity " +
                    std::to_string(kv_device_capacities[slot]) +
                    " B is below the replicated full-context requirement " +
                    std::to_string(physical.value.payload_bytes) + " B");
                return result;
            }
        }
    }
    for (std::size_t slot = 0U; slot < weight_capacities.size(); ++slot) {
        if (kv_device_capacities[slot] >= weight_capacities[slot]) {
            result.errors.emplace_back(
                "DeepSeek KV device budget leaves no weight-cache capacity");
            return result;
        }
        weight_capacities[slot] -= kv_device_capacities[slot];
    }
    auto arena_capacities = weight_capacities;
    auto cache_weight_capacities = arena_capacities;
    // The routed-expert tier suballocates from the same weight arena as the
    // centralized prefill cache, and the tier is permanent while the cache is
    // a prefill performance term. Reserve the tier's bytes out of the cache's
    // logical capacity so the cache stops short of the arena instead of
    // failing an acquire mid-prefill, which is what an unreserved tier caused.
    if (!config.static_expert_plan_path.empty() &&
        config.static_expert_tier_bytes != 0U) {
        for (auto& capacity : cache_weight_capacities) {
            capacity = capacity > config.static_expert_tier_bytes
                ? capacity - config.static_expert_tier_bytes : 0U;
        }
    }
    const auto mhc_slot = static_cast<std::size_t>(std::distance(
        arena_capacities.begin(),
        std::max_element(arena_capacities.begin(), arena_capacities.end())));
    std::uint64_t mhc_projection_bytes = 0U;
    std::uint64_t mhc_auxiliary_bytes = 0U;
    if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
        constexpr std::uint64_t projection_payload_bytes =
            static_cast<std::uint64_t>(kMix) * kMhc * kHidden *
            sizeof(float);
        const auto projection_storage_bytes =
            CudaBackend::weight_storage_bytes(projection_payload_bytes, 0U);
        constexpr std::uint64_t boundary_count = 2U * kLayers;
        constexpr std::uint64_t auxiliary_bytes_per_boundary =
            112U + static_cast<std::uint64_t>(kHidden) *
                       sizeof(std::uint16_t);
        if (projection_storage_bytes == 0U ||
            projection_storage_bytes >
                std::numeric_limits<std::uint64_t>::max() / boundary_count) {
            result.errors.emplace_back(
                "DeepSeek device mHC projection capacity overflows");
            return result;
        }
        mhc_projection_bytes = projection_storage_bytes * boundary_count;
        mhc_auxiliary_bytes =
            auxiliary_bytes_per_boundary * boundary_count;
        if (mhc_auxiliary_bytes >= arena_capacities[mhc_slot]) {
            result.errors.emplace_back(
                "DeepSeek device mHC auxiliaries exceed device capacity");
            return result;
        }
        arena_capacities[mhc_slot] -= mhc_auxiliary_bytes;
        cache_weight_capacities = arena_capacities;
        if (mhc_projection_bytes >= cache_weight_capacities[mhc_slot]) {
            result.errors.emplace_back(
                "DeepSeek device mHC projections leave no weight cache");
            return result;
        }
        cache_weight_capacities[mhc_slot] -= mhc_projection_bytes;
        result = impl_->cuda.validate_dsv4_mhc_device(
            config.devices[mhc_slot]);
        if (!result.ok()) return result;
    }
    const auto admission_started = std::chrono::steady_clock::now();
    Dsv4AdmissionConfig admission_config;
    admission_config.host_memory_ceiling_bytes = config.host_memory_limit_bytes;
    admission_config.vram_weight_budgets = capacities;
    admission_config.host_kv_cache_bytes = config.host_kv_cache_bytes;
    admission_config.device_kv_cache_bytes = kv_device_capacities;
    admission_config.maximum_context_tokens = config.maximum_context_tokens;
    admission_config.prefill_page_tokens = config.prefill_page_tokens;
    admission_config.enable_dspark = config.enable_dspark;
    admission_config.enable_mhc_prepack =
        config.prepack_mhc_projection &&
        config.kv_cache_mode != Dsv4KvCacheMode::PhysicalDevice;
    admission_config.host_routed_experts = config.enable_host_routed_moe;
    admission_config.compact_kv_cache =
        config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle;
    admission_config.physical_kv_cache =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    admission_config.device_resident_mhc =
        config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
    admission_config.require_zero_nvme_decode = config.require_zero_nvme_decode;
    auto admission = plan_dsv4_resident_topology(checkpoint.value->manifest(),
                                                  admission_config);
    const double admission_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - admission_started).count();
    if (!admission.ok()) {
        result.errors = std::move(admission.errors);
        return result;
    }
    admission.plan.fractional_vram_budget_bytes =
        device_plan.value.fractional_budgets;
    admission.plan.explicit_vram_budget_bytes =
        device_plan.value.explicit_budgets;
    admission.plan.applied_vram_budget_bytes = capacities;
    admission.plan.vram_budget_bound.reserve(capacities.size());
    for (std::size_t slot = 0U; slot < capacities.size(); ++slot) {
        admission.plan.vram_budget_bound.emplace_back(
            runtime_budget_bound_name(
                device_plan.value.fractional_budgets[slot],
                device_plan.value.explicit_budgets[slot]));
    }

    impl_->config = config;
    impl_->sampler.seed(config.sampling_seed);
    impl_->memory = admission.plan;
    impl_->checkpoint = std::move(checkpoint.value);
    impl_->tokenizer = std::move(tokenizer.value);
    impl_->devices = config.devices;
    impl_->capacities = cache_weight_capacities;
    impl_->schedule = std::move(device_plan.value.weighted_schedule);
    impl_->mhc_slot = mhc_slot;
    if (!config.route_trace_path.empty()) {
        result = impl_->route_trace.open(config.route_trace_path);
        if (!result.ok()) return result;
    }
    if (config.verbose) {
        for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
            std::cerr << "[hardware] cuda=" << impl_->devices[slot]
                      << " vram_budget_bytes=" << capacities[slot]
                      << " weight_cache_bytes="
                      << cache_weight_capacities[slot]
                      << " weight_arena_bytes=" << arena_capacities[slot]
                      << " kv_cache_bytes=" << kv_device_capacities[slot]
                      << " workspace_reserve_bytes=" << kDeviceWorkspaceReserve
                      << '\n';
        }
        if (config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice) {
            std::cerr << "[hardware] deepseek_device_mhc_cuda="
                      << impl_->devices[impl_->mhc_slot]
                      << " projection_bytes=" << mhc_projection_bytes
                      << " auxiliary_bytes=" << mhc_auxiliary_bytes
                      << " cross_layer_device_state=true\n";
        }
    }
    for (std::size_t slot = 0U; slot < impl_->devices.size(); ++slot) {
        result = impl_->cuda.reserve_weight_arena(
            impl_->devices[slot], arena_capacities[slot]);
        if (!result.ok()) return result;
    }
    impl_->weights = std::make_unique<Dsv4WeightCache>(
        *impl_->checkpoint, impl_->resident, impl_->cuda,
        impl_->devices, cache_weight_capacities,
        config.expert_prefetch_predictions == 0U
            ? 0U : config.expert_prefetch_byte_budget,
        config.expert_prefetch_queue_depth,
        config.expert_prefetch_lease_ticks);
    if (config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle) {
        Dsv4KvCacheConfig kv_config;
        kv_config.block_rows = config.kv_block_rows;
        kv_config.sliding_window_rows = kWindow;
        kv_config.host_capacity_bytes = impl_->memory.host_kv_cache_bytes;
        kv_config.devices = config.devices;
        kv_config.device_capacity_bytes = kv_device_capacities;
        kv_config.physical_layout =
            config.kv_cache_mode == Dsv4KvCacheMode::PhysicalDevice;
        impl_->kv_cache = std::make_unique<Dsv4KvCache>(
            std::move(kv_config), &impl_->cuda);
        result = impl_->kv_cache->validate();
        if (!result.ok()) return result;
        auto sequence = impl_->kv_cache->create_sequence();
        if (!sequence.ok()) {
            result.errors = std::move(sequence.errors);
            return result;
        }
        impl_->active_sequence = sequence.value;
    }
    if (config.host_attention_threads != 0U) {
        impl_->attention_workers = std::make_unique<HostWorkerPool>(
            config.host_attention_threads);
    } else {
        impl_->attention_workers.reset();
    }
    if (config.enable_host_routed_moe) {
        constexpr std::size_t workers = 48U;
        impl_->expert_workers = std::make_unique<HostWorkerPool>(
            workers, std::chrono::milliseconds(1));
        impl_->expert_lane_nodes.resize(workers);
        impl_->expert_lane_positions.resize(workers);
        const auto topology = NumaTopology::detect();
        result = impl_->expert_workers->parallel_for_addressed(
            workers, [&](std::size_t lane) {
                impl_->expert_lane_nodes[lane] =
                    topology.node_of_cpu(sched_getcpu());
            });
        if (!result.ok()) return result;
        for (std::size_t lane = 0U; lane < workers; ++lane) {
            const auto node = impl_->expert_lane_nodes[lane];
            if (node < 0 || node >= 2) {
                result.errors.emplace_back(
                    "DeepSeek host-routed MoE worker is outside its two NUMA nodes");
                return result;
            }
            auto& node_lanes = impl_->expert_node_lanes[
                static_cast<std::size_t>(node)];
            impl_->expert_lane_positions[lane] = node_lanes.size();
            node_lanes.push_back(lane);
        }
        if (impl_->expert_node_lanes[0].size() != workers / 2U ||
            impl_->expert_node_lanes[1].size() != workers / 2U) {
            result.errors.emplace_back(
                "DeepSeek host-routed MoE needs its workers split evenly across "
                "the NUMA nodes");
            return result;
        }
        constexpr std::size_t shards = 2U;
        constexpr std::size_t shard_intermediate =
            kExpertIntermediate / shards;
        impl_->tiled_activation.resize(
            shards * kTopK * shard_intermediate);
        impl_->tiled_routed.resize(shards * kTopK * kHidden);
    }

    ValidationResult staging_result;
    ValidationResult warmup_result;
    double staging_seconds = 0.0;
    double warmup_seconds = 0.0;
    std::atomic<bool> staging_finished{false};
    const auto stage_resident = [&] {
        const auto started = std::chrono::steady_clock::now();
        staging_result = impl_->resident.stage(*impl_->checkpoint,
                                               config.host_memory_limit_bytes,
                                               config.resident_read_workers,
                                               config.enable_dspark,
                                               config.enable_host_routed_moe,
                                               config.hugepage_expert_arena);
        staging_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        staging_finished.store(true, std::memory_order_release);
    };
    const auto warm_spine = [&] {
        const auto started = std::chrono::steady_clock::now();
        warmup_result = impl_->warmup();
        warmup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    };
    if (config.overlap_resident_warmup) {
        std::thread staging_thread(stage_resident);
        warm_spine();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=spine_warmup_complete elapsed_ms="
                      << warmup_seconds * 1000.0 << '\n';
            if (!staging_finished.load(std::memory_order_acquire)) {
                std::cerr << "[deepseek-load] phase=resident_stage_wait_start\n";
            }
        }
        staging_thread.join();
    } else {
        stage_resident();
        if (staging_result.ok()) warm_spine();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=spine_warmup_complete elapsed_ms="
                      << warmup_seconds * 1000.0 << '\n';
        }
    }
    if (config.verbose) {
        const auto stage_stats = impl_->resident.stats();
        std::cerr << "[deepseek-load] phase=resident_stage_complete elapsed_ms="
                  << staging_seconds * 1000.0
                  << " bytes=" << stage_stats.bytes
                  << " workers=" << stage_stats.workers << '\n';
    }
    if (!staging_result.ok()) {
        append_errors(result, std::move(staging_result.errors));
    }
    if (!warmup_result.ok()) {
        append_errors(result, std::move(warmup_result.errors));
    }
    if (!result.ok()) return result;
    // Page-lock after staging and warm-up have both finished, so registration
    // never races an upload reading out of the same mapping. This is a pure
    // transfer-rate optimization: if the kernel refuses to lock the pages the
    // run continues unpinned and says so, because no output byte depends on it.
    double pin_seconds = 0.0;
    if (config.pin_resident_arena) {
        const auto pin_started = std::chrono::steady_clock::now();
        auto pinned = impl_->resident.pin(impl_->cuda);
        pin_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pin_started).count();
        if (!pinned.ok()) {
            // Always say so. A silent failure here leaves the run at the
            // pageable transfer rate while every metric claims otherwise.
            std::cerr << "[deepseek-load] resident arena not pinned: "
                      << pinned.errors.front() << '\n';
        }
    }
    if (config.enable_device_moe) {
        // Each exact expert is three projections. Account conservatively for
        // the arena's per-projection pointer and block alignment padding.
        constexpr std::uint64_t kMaximumExpertArenaPadding =
            3U * (15U + 255U);
        if (impl_->memory.maximum_expert_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                kMaximumExpertArenaPadding ||
            impl_->memory.maximum_expert_bytes + kMaximumExpertArenaPadding >
                std::numeric_limits<std::uint64_t>::max() / kTopK) {
            result.errors.emplace_back(
                "DeepSeek exact top-k expert lease size overflows");
            return result;
        }
        result = impl_->weights->validate_atomic_expert_capacity(
            (impl_->memory.maximum_expert_bytes + kMaximumExpertArenaPadding) *
            kTopK);
        if (!result.ok()) return result;
    }
    // Rank-local decode is admitted last, after every centralized component
    // has reported its real size. Admission needs measured byte accounts, not
    // estimates, and those only exist once the arena and KV cache are built.
    if (config.decode_topology == Dsv4DecodeTopology::RankLocalTp2) {
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=rank_local_setup_start\n";
        }
        const auto rank_local_started = std::chrono::steady_clock::now();
        result = impl_->admit_rank_local();
        const auto rank_local_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - rank_local_started).count();
        if (config.verbose) {
            std::cerr << "[deepseek-load] phase=rank_local_setup elapsed_ms="
                      << rank_local_seconds * 1000.0 << '\n';
        }
        if (!result.ok()) return result;
    }
    const auto reset_started = std::chrono::steady_clock::now();
    result = impl_->reset_sequence(1U);
    const auto reset_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reset_started).count();
    if (config.verbose) {
        std::cerr << "[deepseek-load] phase=sequence_reset elapsed_ms="
                  << reset_seconds * 1000.0 << '\n';
    }
    if (!result.ok()) return result;
    impl_->initialized = true;
    impl_->initialization_metrics.initialization_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      initialization_started).count();
    impl_->initialization_metrics.admission_seconds = admission_seconds;
    impl_->initialization_metrics.resident_staging_seconds = staging_seconds;
    impl_->initialization_metrics.resident_warmup_seconds = warmup_seconds;
    impl_->initialization_metrics.memory = impl_->memory;
    impl_->initialization_metrics.resident_stage = impl_->resident.stats();
    impl_->initialization_metrics.cuda = impl_->cuda.stats();
    impl_->initialization_metrics.cache = impl_->weights->stats();
    if (impl_->rank_local_active && impl_->rank_local_weights != nullptr) {
        const auto rank_weights = impl_->rank_local_weights->device_bytes();
        impl_->initialization_metrics.rank_local_initial_device_vram_bytes =
            impl_->rank_local_initial_device_vram_bytes;
        impl_->initialization_metrics.rank_local_weight_bytes.assign(
            rank_weights.begin(), rank_weights.end());
        impl_->initialization_metrics
            .rank_local_expert_cache_capacity_bytes.assign(
                impl_->rank_local_admission.expert_cache_capacity_bytes.begin(),
                impl_->rank_local_admission.expert_cache_capacity_bytes.end());
        impl_->initialization_metrics.rank_local_admitted_device_bytes.assign(
            impl_->rank_local_admission.device_total_bytes.begin(),
            impl_->rank_local_admission.device_total_bytes.end());
        impl_->initialization_metrics.rank_local_admitted_host_bytes =
            impl_->rank_local_admission.host_total_bytes;
    }
    if (impl_->kv_cache != nullptr) {
        impl_->initialization_metrics.kv_cache = impl_->kv_cache->stats();
    }
    impl_->initialization_metrics.rss_bytes = process_resident_set_bytes();
    impl_->initialization_metrics.device_vram_used_bytes =
        device_vram_used_bytes(impl_->devices);
    impl_->initialization_metrics.detailed_timing = config.detailed_timing;
    impl_->initialization_metrics.dspark_enabled = false;
    impl_->initialization_metrics.device_moe_enabled = config.enable_device_moe;
    impl_->initialization_metrics.host_routed_moe_enabled =
        config.enable_host_routed_moe;
    impl_->initialization_metrics.resident_warmup_overlapped =
        config.overlap_resident_warmup;
    impl_->initialization_metrics.block_kv_cache_enabled =
        config.kv_cache_mode != Dsv4KvCacheMode::ScalarOracle;
    impl_->initialization_metrics.kv_block_rows = config.kv_block_rows;
    impl_->initialization_metrics.host_attention_threads =
        config.host_attention_threads;
    impl_->initialization_metrics.prefill_page_tokens =
        config.prefill_page_tokens;
    impl_->initialization_metrics.prefill_layer_tile_tokens =
        config.prefill_layer_tile_tokens;
    impl_->initialization_metrics.flash_attention_enabled =
        config.enable_flash_attention;
    impl_->initialization_metrics.gpu_lightning_indexer_enabled =
        config.enable_gpu_lightning_indexer;
    impl_->initialization_metrics.flash_attention_minimum_rows =
        config.flash_attention_minimum_rows;
    impl_->initialization_metrics.resident_read_workers =
        impl_->resident.stats().workers;
    impl_->initialization_metrics.resident_pin_seconds = pin_seconds;
    impl_->initialization_metrics.resident_arena_pinned = impl_->resident.pinned();
    impl_->initialization_metrics.spine_warmup_workers =
        static_cast<std::uint32_t>(std::min<std::size_t>(
            config.spine_warmup_workers, impl_->devices.size()));
    impl_->initialization_metrics.expert_prefetch_predictions =
        config.expert_prefetch_predictions;
    impl_->initialization_metrics.expert_prefetch_queue_depth =
        config.expert_prefetch_queue_depth;
    impl_->initialization_metrics.expert_prefetch_byte_budget =
        config.expert_prefetch_byte_budget;
    impl_->initialization_metrics.expert_prefetch_lease_ticks =
        config.expert_prefetch_lease_ticks;
    impl_->initialization_metrics.expert_prefetch_minimum_confidence =
        config.expert_prefetch_minimum_confidence;
    if (config.verbose) {
        const auto& metrics = impl_->initialization_metrics;
        const auto overlapped_seconds = std::max(
            metrics.resident_staging_seconds,
            metrics.resident_warmup_seconds);
        const auto serial_seconds = std::max(
            0.0, metrics.initialization_seconds - overlapped_seconds);
        const auto stage_gib_s = metrics.resident_staging_seconds == 0.0
            ? 0.0
            : static_cast<double>(metrics.resident_stage.bytes) /
                  metrics.resident_staging_seconds / static_cast<double>(1ULL << 30U);
        std::cerr << "[deepseek-load] phase=summary total_ms="
                  << metrics.initialization_seconds * 1000.0
                  << " admission_ms=" << metrics.admission_seconds * 1000.0
                  << " resident_stage_ms="
                  << metrics.resident_staging_seconds * 1000.0
                  << " resident_stage_bytes=" << metrics.resident_stage.bytes
                  << " resident_stage_gib_s=" << stage_gib_s
                  << " spine_warmup_ms="
                  << metrics.resident_warmup_seconds * 1000.0
                  << " weight_upload_bytes=" << metrics.cuda.weight_upload_bytes
                  << " weight_copy_ms="
                  << static_cast<double>(metrics.cuda.weight_copy_nanoseconds) /
                         1.0e6
                  << " weight_allocation_ms="
                  << static_cast<double>(
                         metrics.cuda.weight_allocation_nanoseconds) / 1.0e6
                  << " overlap_window_ms=" << overlapped_seconds * 1000.0
                  << " other_serial_ms=" << serial_seconds * 1000.0
                  << " rss_bytes=" << metrics.rss_bytes << '\n';
    }
    return result;
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_stream(
    std::string_view prompt, std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    const std::array messages{ChatMessage{ChatRole::User,
                                          std::string(prompt)}};
    return generate_chat_stream(messages, maximum_new_tokens, on_token);
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const TokenStreamCallback& on_token) {
    SamplingOptions sampling;
    sampling.temperature = impl_->config.sampling_temperature;
    sampling.seed = impl_->config.sampling_seed;
    return generate_chat_stream(messages, maximum_new_tokens, sampling, {}, on_token);
}

Dsv4GenerationResult DeepSeekV4Runtime::generate_chat_stream(
    std::span<const ChatMessage> messages,
    std::uint32_t maximum_new_tokens,
    const SamplingOptions& sampling,
    std::span<const std::string> stop,
    const TokenStreamCallback& on_token) {
    Dsv4GenerationResult result;
    if (!impl_->initialized) {
        result.errors.emplace_back("DeepSeek runtime is not initialized");
        return result;
    }
    if (maximum_new_tokens == 0U) {
        result.errors.emplace_back("maximum_new_tokens must be positive");
        return result;
    }
    std::string sampling_error;
    if (!validate_sampling_options(sampling, sampling_error)) {
        result.errors.emplace_back("invalid sampling option: " + sampling_error);
        return result;
    }
    impl_->active_sampling = sampling;
    impl_->sampled_token_counts.assign(kVocabulary, 0U);
    impl_->sampled_token_ids.clear();
    impl_->sampler.seed(sampling.seed);
    std::string validation_error;
    if (!validate_chat_messages(messages, validation_error)) {
        result.errors.push_back(std::move(validation_error));
        return result;
    }
    impl_->weights->finish_prefetch();
    ChatPromptRequest prompt_request;
    prompt_request.messages = messages;
    prompt_request.maximum_new_tokens = maximum_new_tokens;
    prompt_request.maximum_context_tokens = impl_->config.maximum_context_tokens;
    prompt_request.render = [](std::span<const ChatMessage> active) {
        return render_deepseek_v4_chat_prompt(active);
    };
    prompt_request.encode = [&](const std::string& text) {
        return impl_->tokenizer.encode(text);
    };
    auto prompt = prepare_chat_prompt(prompt_request);
    if (!prompt.ok()) {
        result.errors = std::move(prompt.errors);
        return result;
    }
    result.prompt_token_ids = std::move(prompt.token_ids);
    impl_->active_request_id = impl_->generated_requests++;
    impl_->active_prompt_tokens =
        static_cast<std::uint32_t>(result.prompt_token_ids.size());
    impl_->reset_diagnostics();
    const auto active_context_tokens = static_cast<std::uint32_t>(
        result.prompt_token_ids.size() + maximum_new_tokens);
    std::size_t prefill_offset = impl_->config.enable_incremental_kv_continuation &&
        impl_->reusable_sequence
        ? incremental_kv_prefix_tokens(impl_->cached_token_ids,
                                       result.prompt_token_ids)
        : 0U;
    if (prefill_offset != 0U &&
        !std::all_of(impl_->attention_state.begin(),
                     impl_->attention_state.end(),
                     [active_context_tokens](const AttentionState& state) {
                         return state.compressor.ratio != 4U ||
                             active_context_tokens <=
                                 kIndexTopK * state.compressor.ratio ||
                             state.indexer_compressor.ratio ==
                                 state.compressor.ratio;
                     })) {
        prefill_offset = 0U;
    }
    impl_->reusable_sequence = false;
    if (prefill_offset == 0U) {
        auto reset = impl_->reset_sequence(active_context_tokens);
        if (!reset.ok()) {
            result.errors = std::move(reset.errors);
            return result;
        }
    } else {
        // A continuation keeps the sequence, so reset_sequence does not run and
        // the leases the previous generation left open are still held. The
        // prefill below appends the new turn into the last block of that same
        // sequence, and an append refuses to mutate a leased block.
        impl_->release_retained_kv_leases();
    }
    const auto reads_before = impl_->checkpoint->stats();
    const auto cuda_before = impl_->cuda.stats();
    const auto cache_before = impl_->weights->stats();
    const auto kv_cache_before = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_before = impl_->device_moe_stats;
    const auto graph_before = impl_->graph_stats;
    const auto prefill_started = std::chrono::steady_clock::now();
    auto prefill_tokens = std::span<const std::uint32_t>(
        result.prompt_token_ids).subspan(prefill_offset);
    auto next = impl_->forward_prefill(
        prefill_tokens, static_cast<std::uint32_t>(prefill_offset));
    if (!next.ok()) {
        result.errors = std::move(next.errors);
        return result;
    }
    impl_->weights->drain_prefetch();
    // In-chain selection projects the index query on both ranks, so both
    // devices need the index weights; prefill leaves each layer's pair on one
    // device only. Faulting the other copy in on the first decoded token would
    // be a checkpoint read inside the window the zero-NVMe contract covers, so
    // it happens here, before the decode boundary is sampled.
    auto warmed = impl_->rank_local_warm_index_projections();
    if (!warmed.ok()) {
        result.errors = std::move(warmed.errors);
        return result;
    }
    impl_->cached_token_ids = result.prompt_token_ids;
    result.metrics.prefill_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prefill_started).count();
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill_tokens.size();
    result.metrics.reused_prompt_tokens = prefill_offset;
    result.metrics.incremental_kv_continuation = prefill_offset != 0U;
    const auto reads_after_prefill = impl_->checkpoint->stats();
    const auto cuda_after_prefill = impl_->cuda.stats();
    const auto cache_after_prefill = impl_->weights->stats();
    const auto kv_cache_after_prefill = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_after_prefill = impl_->device_moe_stats;
    const auto graph_after_prefill = impl_->graph_stats;
    constexpr std::uint32_t stop_token = 1U;
    StopSequenceBuffer output(stop);
    if (next.value != stop_token) {
        result.generated_token_ids.push_back(next.value);
        result.logprobs.push_back(impl_->last_sample);
        const auto piece = impl_->tokenizer.decode_token(next.value);
        if (!piece.ok()) {
            result.errors = std::move(piece.errors);
            return result;
        }
        output.append(next.value, piece.value, on_token);
    }
    std::uint32_t position = static_cast<std::uint32_t>(
        result.prompt_token_ids.size());
    std::uint64_t decode_steps = 0U;
    result.metrics.decode_step_seconds.reserve(maximum_new_tokens);
    const auto decode_started = std::chrono::steady_clock::now();
    while (next.value != stop_token && !output.stopped() && !output.cancelled() &&
           result.generated_token_ids.size() < maximum_new_tokens) {
        const auto input_token = next.value;
        const auto step_started = std::chrono::steady_clock::now();
        next = impl_->forward_token(input_token, position++, true);
        result.metrics.decode_step_seconds.push_back(
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - step_started).count());
        if (!next.ok()) {
            result.errors = std::move(next.errors);
            return result;
        }
        impl_->cached_token_ids.push_back(input_token);
        ++decode_steps;
        if (next.value != stop_token) {
            result.generated_token_ids.push_back(next.value);
            result.logprobs.push_back(impl_->last_sample);
            const auto piece = impl_->tokenizer.decode_token(next.value);
            if (!piece.ok()) {
                result.errors = std::move(piece.errors);
                return result;
            }
            output.append(next.value, piece.value, on_token);
        }
    }
    output.finish(on_token);
    result.stopped = output.stopped();
    impl_->weights->finish_prefetch();
    result.metrics.decode_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - decode_started).count();
    result.metrics.decode_tokens = decode_steps;
    result.text = output.text();
    const auto reads_after_decode = impl_->checkpoint->stats();
    const auto cuda_after_decode = impl_->cuda.stats();
    const auto cache_after_decode = impl_->weights->stats();
    const auto kv_cache_after_decode = impl_->kv_cache == nullptr
        ? Dsv4KvCacheStats{} : impl_->kv_cache->stats();
    const auto device_moe_after_decode = impl_->device_moe_stats;
    const auto graph_after_decode = impl_->graph_stats;
    const double prefill_seconds = result.metrics.prefill_seconds;
    const double decode_seconds = result.metrics.decode_seconds;
    auto decode_step_seconds = std::move(result.metrics.decode_step_seconds);
    result.metrics = impl_->initialization_metrics;
    result.metrics.prefill_seconds = prefill_seconds;
    result.metrics.decode_seconds = decode_seconds;
    result.metrics.decode_step_seconds = std::move(decode_step_seconds);
    result.metrics.prompt_tokens = result.prompt_token_ids.size();
    result.metrics.prefill_tokens = prefill_tokens.size();
    result.metrics.reused_prompt_tokens = prefill_offset;
    result.metrics.decode_tokens = decode_steps;
    result.metrics.incremental_kv_continuation = prefill_offset != 0U;
    result.metrics.rss_bytes = process_resident_set_bytes();
    result.metrics.device_vram_used_bytes =
        device_vram_used_bytes(impl_->devices);
    result.metrics.generation_checkpoint_reads = read_delta(reads_after_decode,
                                                             reads_before);
    result.metrics.decode_checkpoint_reads = read_delta(reads_after_decode,
                                                         reads_after_prefill);
    result.metrics.cuda = impl_->cuda.stats();
    result.metrics.cache = impl_->weights->stats();
    result.metrics.kv_cache = kv_cache_after_decode;
    result.metrics.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_before);
    result.metrics.graph = graph_delta(graph_after_decode, graph_before);
    result.metrics.prefill.checkpoint_reads = read_delta(reads_after_prefill,
                                                         reads_before);
    result.metrics.prefill.cuda = cuda_delta(cuda_after_prefill, cuda_before);
    result.metrics.prefill.cache = cache_delta(cache_after_prefill, cache_before);
    result.metrics.prefill.kv_cache = kv_cache_delta(
        kv_cache_after_prefill, kv_cache_before);
    result.metrics.prefill.device_moe = device_moe_delta(
        device_moe_after_prefill, device_moe_before);
    result.metrics.prefill.graph = graph_delta(graph_after_prefill, graph_before);
    result.metrics.decode.checkpoint_reads = read_delta(reads_after_decode,
                                                        reads_after_prefill);
    result.metrics.decode.cuda = cuda_delta(cuda_after_decode, cuda_after_prefill);
    result.metrics.decode.cache = cache_delta(cache_after_decode, cache_after_prefill);
    result.metrics.decode.kv_cache = kv_cache_delta(
        kv_cache_after_decode, kv_cache_after_prefill);
    result.metrics.decode.device_moe = device_moe_delta(
        device_moe_after_decode, device_moe_after_prefill);
    result.metrics.decode.graph = graph_delta(
        graph_after_decode, graph_after_prefill);
    result.diagnostics = std::move(impl_->diagnostics);
    if (result.metrics.cache.lease_acquires !=
            result.metrics.cache.lease_releases ||
        result.metrics.cache.prefetch_lease_acquires !=
            result.metrics.cache.prefetch_lease_releases ||
        result.metrics.cache.active_prefetch_leases != 0U ||
        std::any_of(result.metrics.cache.active_leases.begin(),
                    result.metrics.cache.active_leases.end(),
                    [](std::uint64_t count) { return count != 0U; })) {
        result.errors.emplace_back(
            "DeepSeek generation completed with outstanding CUDA weight leases");
    }
    if (impl_->config.require_zero_nvme_decode &&
        (result.metrics.decode_checkpoint_reads.calls != 0U ||
         result.metrics.decode_checkpoint_reads.bytes != 0U)) {
        result.errors.emplace_back(
            "DeepSeek zero-NVMe decode contract was violated by checkpoint reads");
    }
    if (impl_->route_trace.is_open()) {
        auto flushed = impl_->route_trace.flush();
        result.errors.insert(result.errors.end(),
                             std::make_move_iterator(flushed.errors.begin()),
                             std::make_move_iterator(flushed.errors.end()));
    }
    impl_->reusable_sequence =
        impl_->config.enable_incremental_kv_continuation && result.ok();
    return result;
}

Dsv4GenerationResult DeepSeekV4Runtime::generate(
    std::string_view prompt, std::uint32_t maximum_new_tokens) {
    return generate_stream(prompt, maximum_new_tokens, {});
}

const Dsv4MemoryPlan& DeepSeekV4Runtime::memory_plan() const noexcept {
    return impl_->memory;
}

}  // namespace strata
