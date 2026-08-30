
struct CudaWeight::Impl {
    void* weights{};
    void* scales{};
    CudaWeightDescriptor descriptor;
    // Set once, at load, by CudaBackend::prepack_fragment. The fragment order
    // REPLACES the canonical device layout -- one-copy residency -- so every
    // consumer of this weight must dispatch a register-fed kernel once this is
    // true. A consumer that reads it canonically would read a permutation.
    bool fragment_prepacked{};
    // Mutually exclusive with fragment_prepacked. This is Marlin's compact
    // K16/N64 code order plus permuted group-32 E8M0 scales.
    bool marlin_prepacked{};
    std::uint64_t bytes{};
    int device{-1};
    std::shared_ptr<WeightArena> arena;
    std::uint64_t arena_offset{};

    ~Impl() {
        if (arena != nullptr) {
            arena->release(arena_offset, bytes);
            return;
        }
        if (device >= 0) static_cast<void>(cudaSetDevice(device));
        if (weights != nullptr) static_cast<void>(cudaFree(weights));
        if (scales != nullptr) static_cast<void>(cudaFree(scales));
    }
};

struct CudaBuffer::Impl {
    void* data{};
    std::uint64_t bytes{};
    int device{-1};

    ~Impl() {
        if (device >= 0) static_cast<void>(cudaSetDevice(device));
        if (data != nullptr) static_cast<void>(cudaFree(data));
    }
};

struct CudaDsv4MhcWeights::Impl {
    CudaWeight projection;
    CudaBuffer auxiliary;
};

struct CudaBackend::Impl {
    struct DeviceState {
        cudaStream_t stream{};
        cudaStream_t moe_shared_stream{};
        std::array<cudaStream_t, 3U> dsv4_attention_aux_streams{};
        cublasHandle_t cublas{};
        cudaStream_t upload_stream{};
        cudaEvent_t upload_ready{};
        bool upload_ordered{};
        cudaEvent_t activation_start{};
        cudaEvent_t activation_uploaded{};
        cudaEvent_t mhc_transition_finished{};
        cudaEvent_t router_started{};
        cudaEvent_t kernel_finished{};
        cudaEvent_t activation_downloaded{};
        cudaEvent_t moe_start{};
        cudaEvent_t moe_hidden_uploaded{};
        cudaEvent_t moe_kernel_finished{};
        cudaEvent_t moe_download_started{};
        cudaEvent_t moe_completed{};
        cudaEvent_t moe_shared_input_finished{};
        cudaEvent_t moe_shared_gate_up_finished{};
        cudaEvent_t moe_shared_activation_finished{};
        cudaEvent_t moe_shared_finished{};
        cudaEvent_t dsv4_cross_device_ready{};
        cudaEvent_t dsv4_attention_input_ready{};
        std::array<cudaEvent_t, 3U> dsv4_attention_aux_finished{};
        float* input{};
        float* output{};
        std::uint64_t input_bytes{};
        std::uint64_t output_bytes{};
        // Pinned staging for matmul activations. A cudaMemcpyAsync whose host
        // side is pageable is not asynchronous: the driver stages it itself and
        // blocks, which decode pays on both legs of every one of its ~430
        // matmul round trips a step. FlashAttention and the MoE command already
        // stage through pinned host memory; this is the same for the generic
        // matmul.
        // GLM-5.3 shared-expert scratch. The tier is resident for the life
        // of the runtime and every layer runs the same three shapes, so these
        // are allocated once on first use and never grow again.
        float* glm53_shared_input{};
        float* glm53_shared_gate{};
        float* glm53_shared_up{};
        float* glm53_shared_activation{};
        float* glm53_shared_output{};
        std::uint32_t glm53_shared_hidden{};
        std::uint32_t glm53_shared_intermediate{};
        // Pinned host staging: a pageable cudaMemcpyAsync is not asynchronous,
        // and the whole point of this path is that the enqueue returns before
        // the host starts its eight routed experts.
        float* glm53_shared_staging{};
        std::uint32_t glm53_shared_staging_floats{};
        bool glm53_mla_scores_pending{};
        std::uint32_t glm53_shared_batch{};
        bool glm53_shared_gate_up_in_flight{};
        bool glm53_shared_down_in_flight{};
        std::byte* matmul_host_input{};
        std::byte* matmul_host_output{};
        std::uint64_t matmul_host_input_bytes{};
        std::uint64_t matmul_host_output_bytes{};
        // Register-fed matmul workspaces: the B-fragment activation, the
        // split-K partials, and one arrival counter per N-tile. All three are
        // grown geometrically and kept, so a decode step that repeats the same
        // shapes allocates nothing after the first call.
        // The shared expert dispatches on its own stream, concurrently with the
        // routed path, so it keeps workspaces separate from the generic
        // matmul's rather than sharing them.
        RegfedWorkspace moe_regfed{};
        // Scratch for the upload-path fragment prepack. Uploads for a device
        // are stream-ordered, so one buffer serves them; only its growth needs
        // the lock.
        void* upload_prepack_scratch{};
        std::uint64_t upload_prepack_scratch_bytes{};
        // Capacity-bounded pinned ring for streamed weight uploads.  Mapped
        // checkpoint pages are pageable, so cudaMemcpyAsync otherwise performs
        // its own blocking host staging for every cache miss.  The ring makes
        // that staging explicit, reusable, and large enough for many expert
        // projections so CPU copies overlap the copy engine.  It is sized from
        // the device weight arena rather than from a particular GPU model.
        std::byte* weight_host_staging{};
        std::uint64_t weight_host_staging_bytes{};
        std::uint64_t weight_host_staging_cursor{};
        // Register-fed fused MoE workspaces: split-K partials for gate, up and
        // down, plus the two B-fragment activation buffers. Grown geometrically
        // and kept, so a decode step that repeats the same shapes allocates
        // nothing after the first token.
        void* moe_regfed_gate_partials{};
        void* moe_regfed_up_partials{};
        void* moe_regfed_down_partials{};
        void* moe_regfed_hidden_fragment{};
        void* moe_regfed_activation_fragment{};
        void* moe_regfed_compact{};
        std::uint64_t moe_regfed_gate_partial_bytes{};
        std::uint64_t moe_regfed_up_partial_bytes{};
        std::uint64_t moe_regfed_down_partial_bytes{};
        std::uint64_t moe_regfed_hidden_fragment_bytes{};
        std::uint64_t moe_regfed_activation_fragment_bytes{};
        std::uint64_t moe_regfed_compact_bytes{};
        RegfedWorkspace gemma_regfed{};
        GemmaMarlinWorkspace gemma_marlin{};
        float* moe_regfed_gate{};
        float* moe_regfed_up{};
        std::uint64_t moe_regfed_gate_bytes{};
        void* regfed_activation{};
        float* regfed_partials{};
        std::uint32_t* regfed_counters{};
        void* regfed_scratch{};
        std::uint64_t regfed_activation_bytes{};
        std::uint64_t regfed_partial_bytes{};
        std::uint64_t regfed_counter_bytes{};
        std::uint64_t regfed_scratch_bytes{};
        std::byte* attention_upload{};
        std::byte* attention_download{};
        std::byte* attention_host_upload{};
        std::byte* attention_host_download{};
        float* attention_scores{};
        std::uint64_t attention_upload_bytes{};
        std::uint64_t attention_download_bytes{};
        std::uint64_t attention_host_upload_bytes{};
        std::uint64_t attention_host_download_bytes{};
        std::uint64_t attention_score_bytes{};
        // Index-query preparation: the head-major query block plus its rope
        // cosines and sines. Grown once and reused, never on a timed path.
        std::byte* dsv4_index_query_workspace{};
        std::uint64_t dsv4_index_query_workspace_bytes{};
        std::byte* dsv4_attention_workspace{};
        std::byte* dsv4_attention_host_upload{};
        std::byte* dsv4_attention_host_download{};
        std::uint64_t dsv4_attention_workspace_bytes{};
        std::uint64_t dsv4_attention_host_upload_bytes{};
        std::uint64_t dsv4_attention_host_download_bytes{};
        std::byte* dsv4_attention_prepare_workspace{};
        std::byte* dsv4_attention_prepare_host_upload{};
        std::byte* dsv4_attention_prepare_host_download{};
        std::byte* dsv4_attention_prepare_fixed_host_upload{};
        std::uint64_t dsv4_attention_prepare_workspace_bytes{};
        std::uint64_t dsv4_attention_prepare_host_upload_bytes{};
        std::uint64_t dsv4_attention_prepare_host_download_bytes{};
        __nv_bfloat16* dsv4_prepared_queries{};
        bool dsv4_attention_prepared{};
        // Index-projection sources left behind by the last preparation on this
        // device: the E4M3-quantized query rank, and the expanded BF16 layer
        // input. Both point into the preparation workspace, so the next
        // preparation on this device overwrites them.
        const float* dsv4_prepared_index_query_source{};
        const float* dsv4_prepared_index_hidden_source{};
        // Device-only index projections awaiting an in-chain selection. Set by
        // dsv4_index_projections when it returns no host output, consumed by
        // the next dsv4_physical_lightning_index on this device.
        const float* dsv4_index_projection_queries{};
        const float* dsv4_index_projection_weights{};
        const unsigned int* dsv4_index_projection_error{};
        std::uint32_t dsv4_index_projection_heads{};
        std::uint32_t dsv4_index_projection_head_dim{};
        std::array<Dsv4AttentionPrepareHostCommand, 43U>
            dsv4_attention_prepare_host_commands{};
        std::uint32_t dsv4_attention_prepare_host_command_count{};
        bool dsv4_host_moe_input_pending{};
        float* dsv4_host_moe_router_logits{};
        unsigned int* dsv4_host_moe_device_failure{};
        const unsigned int* dsv4_host_moe_host_failure{};
        std::byte* dsv4_deferred_attention_host_upload{};
        std::byte* dsv4_deferred_attention_host_download{};
        std::uint32_t dsv4_deferred_attention_command_count{};
        int dsv4_deferred_attention_source_device{-1};
        bool dsv4_deferred_attention_cross_transition{};
        Dsv4MhcWorkspace* dsv4_mhc_workspace{};
        std::byte* glm53_mla_workspace{};
        std::uint64_t glm53_mla_workspace_bytes{};
        std::byte* dsv4_mhc_host_staging{};
        std::uint64_t dsv4_mhc_workspace_bytes{};
        std::uint64_t dsv4_mhc_host_staging_bytes{};
        std::uint32_t dsv4_mhc_stage{};
        std::uint32_t dsv4_mhc_residual_index{};
        bool dsv4_mhc_branch_ready{};
        // Saved fused mHC state of every slot that is not currently selected.
        // The three scalars above are the selected slot's live copy; selecting
        // a different slot writes them back here and loads that slot's copy.
        // The workspace pointer is the arena base plus the slot index, so it
        // is derived rather than stored.
        std::vector<Dsv4MhcSlotState> dsv4_mhc_saved_slots{};
        std::uint32_t dsv4_mhc_active_slot{};
        Dsv4MhcWorkspace* dsv4_mhc_slot_arena{};
        std::uint32_t dsv4_mhc_slot_capacity{};
        bool dsv4_mhc_failed{};
        float* dsv4_mhc_head_input{};
        float* dsv4_mhc_head_output{};
        std::byte* dsv4_mhc_head_host_staging{};
        std::uint64_t dsv4_mhc_head_input_bytes{};
        std::uint64_t dsv4_mhc_head_output_bytes{};
        std::uint64_t dsv4_mhc_head_host_staging_bytes{};
        // Logit bytes of the head currently in flight. The reservation above
        // is a capacity, because one device may serve heads of two shapes:
        // centralized prefill projects the full vocabulary while rank-local
        // decode projects one rank's row shard. A completion must still match
        // the enqueue that produced it exactly, which is what this pins.
        std::uint64_t dsv4_mhc_head_logits_bytes{};
        Dsv4MhcHeadCallbackState dsv4_mhc_head_callback{};
        bool dsv4_mhc_head_in_flight{};
        float* gemma_workspace{};
        float* gemma_scores{};
        unsigned int* gemma_error{};
        std::byte* gemma_host_staging{};
        std::uint64_t gemma_workspace_bytes{};
        std::uint64_t gemma_score_bytes{};
        std::uint64_t gemma_host_staging_bytes{};
        std::byte* lightning_workspace{};
        std::uint64_t lightning_workspace_bytes{};
        float* moe_hidden{};
        float* moe_activations{};
        float* moe_output{};
        float* moe_bf16_silu{};
        unsigned int* moe_error{};
        void* moe_host_staging{};
        // Routed-expert tier: host-side pointer arrays mirrored to device, the
        // pinned selection the callback writes, and its device copy.
        std::vector<const unsigned char*> tier_host_pointers[6];
        const unsigned char** tier_device_pointers[6]{};
        CudaDsv4TierSelection* tier_selection_host{};
        CudaDsv4TierSelection* tier_selection_device{};
        std::uint32_t tier_layers{};
        std::uint32_t tier_experts{};
        std::uint64_t tier_installed{};
        bool tier_committed{};
        // Taken from the first installed triplet rather than assumed, and
        // every later triplet must match: a tier holding two shapes would
        // index one of them wrongly.
        std::uint64_t tier_gate_packed_columns{};
        std::uint64_t tier_gate_scale_columns{};
        std::uint64_t tier_down_packed_columns{};
        std::uint64_t tier_down_scale_columns{};
        float* tier_activations{};
        std::uint64_t tier_activation_bytes{};
        // Overlapped dispatch. The tier runs on its own stream, released by
        // `tier_route_ready` -- recorded between the route half and the host
        // share -- and rejoined through `tier_finished` before the join. Its
        // contribution lands in `tier_partials` rather than in the rank
        // partials, because with the two concurrent the rank-partial upload
        // would race the tier's accumulation into the same buffer.
        cudaStream_t tier_stream{};
        cudaEvent_t tier_route_ready{};
        cudaEvent_t tier_finished{};
        float* tier_partials{};
        std::uint64_t moe_hidden_bytes{};
        std::uint64_t moe_activation_bytes{};
        std::uint64_t moe_output_bytes{};
        std::uint64_t moe_host_staging_bytes{};
        std::uint64_t moe_hidden_columns{};
        std::uint64_t moe_intermediate_columns{};
        std::uint32_t moe_rows{1U};
        std::uint32_t moe_routed_count{};
        std::uint64_t moe_kernel_launches{};
        // Page-path work list: row indices, per-row coefficients, and the group
        // table, all device-side. Sized to the largest page seen so far and
        // reused, because a prefill visits every layer with the same shape.
        std::uint32_t* moe_page_rows{};
        float* moe_page_coefficients{};
        void* moe_page_groups{};
        std::uint32_t* moe_page_shared_rows{};
        std::uint64_t moe_page_rows_bytes{};
        std::uint64_t moe_page_coefficient_bytes{};
        std::uint64_t moe_page_group_bytes{};
        std::uint64_t moe_page_shared_row_bytes{};
        std::uint32_t moe_page_work_count{};
        std::uint32_t moe_page_shared_count{};
        // Rows the shared expert produced. One for the single-row command; the
        // page command sets it to the page's shared row count so collection
        // sizes the download from the command that actually ran.
        std::uint32_t moe_shared_rows{1U};
        // Set while a deferred upload's copies are still in flight on `stream`.
        // Cleared by synchronize_uploads(), which the caller owes before the
        // upload's host source may be released.
        bool pending_uploads{};
        std::vector<std::shared_ptr<CudaWeight::Impl>> moe_weights;
        std::vector<std::shared_ptr<CudaWeight::Impl>> quarantined_weights;
        std::vector<std::shared_ptr<CudaBuffer::Impl>> quarantined_buffers;
        std::shared_ptr<WeightArena> weight_arena;
        bool moe_has_shared{};
        // True only for the single-row FP8 shared-expert path that records all
        // four phase events below. Host-join, page, and generic MoE commands
        // reuse moe_shared_finished for ordering but do not populate the
        // intermediate events.
        bool moe_shared_phase_timing_valid{};
        bool moe_host_join{};
        bool moe_output_to_mhc{};
        bool moe_in_flight{};
        bool moe_poisoned{};
        Dsv4HostMoeCallbackState moe_host_callback;
        std::array<Dsv4HostMoeCallbackState, 43U>
            moe_host_callbacks{};
        std::uint32_t moe_host_callback_count{};
        bool flash_attention_supported{};
        bool dsv4_paged_attention_supported{};
        bool dsv4_mhc_supported{};
        bool lightning_index_supported{};
        bool dsv4_fp8_tensor_page_supported{};
        bool fp8_f32_tensor_page_supported{};
        bool fp8_f32_register_fed_supported{};
    };

    std::unordered_map<int, DeviceState> devices;
    CudaBackendStats stats;
    bool detailed_timing{};
    mutable std::mutex mutex;

    ~Impl() {
        for (auto& [device, state] : devices) {
            static_cast<void>(cudaSetDevice(device));
            if (state.input != nullptr) static_cast<void>(cudaFree(state.input));
            if (state.output != nullptr) static_cast<void>(cudaFree(state.output));
            if (state.attention_upload != nullptr) {
                static_cast<void>(cudaFree(state.attention_upload));
            }
            if (state.attention_download != nullptr) {
                static_cast<void>(cudaFree(state.attention_download));
            }
            if (state.attention_scores != nullptr) {
                static_cast<void>(cudaFree(state.attention_scores));
            }
            if (state.dsv4_attention_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_attention_workspace));
            }
            if (state.dsv4_index_query_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_index_query_workspace));
            }
            if (state.dsv4_attention_prepare_workspace != nullptr) {
                static_cast<void>(
                    cudaFree(state.dsv4_attention_prepare_workspace));
            }
            if (state.dsv4_mhc_slot_arena != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_slot_arena));
            } else if (state.dsv4_mhc_workspace != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_workspace));
            }
            if (state.glm53_mla_workspace != nullptr) {
                static_cast<void>(cudaFree(state.glm53_mla_workspace));
            }
            if (state.gemma_workspace != nullptr) {
                static_cast<void>(cudaFree(state.gemma_workspace));
            }
            if (state.gemma_scores != nullptr) {
                static_cast<void>(cudaFree(state.gemma_scores));
            }
            if (state.gemma_error != nullptr) {
                static_cast<void>(cudaFree(state.gemma_error));
            }
            if (state.gemma_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.gemma_host_staging));
            }
            // moe_regfed_up is an interior pointer into the gate allocation,
            // not an allocation of its own: freeing it is cudaErrorInvalidValue.
            for (void* pointer : {state.moe_regfed_gate_partials,
                                  state.moe_regfed_up_partials,
                                  state.moe_regfed_down_partials,
                                  state.moe_regfed_hidden_fragment,
                                  state.moe_regfed_activation_fragment,
                                  state.moe_regfed_compact}) {
                if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            }
            if (state.upload_prepack_scratch != nullptr) {
                static_cast<void>(cudaFree(state.upload_prepack_scratch));
            }
            if (state.weight_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.weight_host_staging));
            }
            for (void* pointer : {state.moe_regfed.activation,
                                  state.moe_regfed.partials,
                                  state.moe_regfed.counters,
                                  state.moe_regfed.scratch,
                                  static_cast<void*>(state.moe_regfed_gate)}) {
                if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            }
            for (void* pointer : {state.gemma_regfed.activation,
                                  state.gemma_regfed.partials,
                                  state.gemma_regfed.counters,
                                  state.gemma_regfed.scratch}) {
                if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            }
            for (void* pointer : {state.gemma_marlin.activation,
                                  state.gemma_marlin.reduce,
                                  state.gemma_marlin.reorder,
                                  state.gemma_marlin.locks}) {
                if (pointer != nullptr) static_cast<void>(cudaFree(pointer));
            }
            if (state.regfed_activation != nullptr) {
                static_cast<void>(cudaFree(state.regfed_activation));
            }
            if (state.regfed_partials != nullptr) {
                static_cast<void>(cudaFree(state.regfed_partials));
            }
            if (state.regfed_counters != nullptr) {
                static_cast<void>(cudaFree(state.regfed_counters));
            }
            if (state.regfed_scratch != nullptr) {
                static_cast<void>(cudaFree(state.regfed_scratch));
            }
            if (state.glm53_shared_input != nullptr) {
                static_cast<void>(cudaFree(state.glm53_shared_input));
            }
            if (state.glm53_shared_gate != nullptr) {
                static_cast<void>(cudaFree(state.glm53_shared_gate));
            }
            if (state.glm53_shared_up != nullptr) {
                static_cast<void>(cudaFree(state.glm53_shared_up));
            }
            if (state.glm53_shared_activation != nullptr) {
                static_cast<void>(cudaFree(state.glm53_shared_activation));
            }
            if (state.glm53_shared_output != nullptr) {
                static_cast<void>(cudaFree(state.glm53_shared_output));
            }
            if (state.glm53_shared_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.glm53_shared_staging));
            }
            if (state.matmul_host_input != nullptr) {
                static_cast<void>(cudaFreeHost(state.matmul_host_input));
            }
            if (state.matmul_host_output != nullptr) {
                static_cast<void>(cudaFreeHost(state.matmul_host_output));
            }
            if (state.attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_upload));
            }
            if (state.attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(state.attention_host_download));
            }
            if (state.dsv4_attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_host_upload));
            }
            if (state.dsv4_attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_host_download));
            }
            if (state.dsv4_deferred_attention_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_deferred_attention_host_upload));
            }
            if (state.dsv4_deferred_attention_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_deferred_attention_host_download));
            }
            if (state.dsv4_attention_prepare_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_host_upload));
            }
            if (state.dsv4_attention_prepare_host_download != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_host_download));
            }
            if (state.dsv4_attention_prepare_fixed_host_upload != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_attention_prepare_fixed_host_upload));
            }
            if (state.dsv4_mhc_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.dsv4_mhc_host_staging));
            }
            if (state.dsv4_mhc_head_input != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_head_input));
            }
            if (state.dsv4_mhc_head_output != nullptr) {
                static_cast<void>(cudaFree(state.dsv4_mhc_head_output));
            }
            if (state.dsv4_mhc_head_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(
                    state.dsv4_mhc_head_host_staging));
            }
            if (state.lightning_workspace != nullptr) {
                static_cast<void>(cudaFree(state.lightning_workspace));
            }
            if (state.moe_hidden != nullptr) static_cast<void>(cudaFree(state.moe_hidden));
            if (state.moe_activations != nullptr) {
                static_cast<void>(cudaFree(state.moe_activations));
            }
            if (state.moe_output != nullptr) static_cast<void>(cudaFree(state.moe_output));
            if (state.moe_bf16_silu != nullptr) {
                static_cast<void>(cudaFree(state.moe_bf16_silu));
            }
            if (state.moe_error != nullptr) static_cast<void>(cudaFree(state.moe_error));
            if (state.moe_page_rows != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_rows));
            }
            if (state.moe_page_coefficients != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_coefficients));
            }
            if (state.moe_page_groups != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_groups));
            }
            if (state.moe_page_shared_rows != nullptr) {
                static_cast<void>(cudaFree(state.moe_page_shared_rows));
            }
            if (state.moe_host_staging != nullptr) {
                static_cast<void>(cudaFreeHost(state.moe_host_staging));
            }
            if (state.activation_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.activation_start));
                static_cast<void>(cudaEventDestroy(state.activation_uploaded));
                static_cast<void>(cudaEventDestroy(
                    state.mhc_transition_finished));
                static_cast<void>(cudaEventDestroy(state.router_started));
                static_cast<void>(cudaEventDestroy(state.kernel_finished));
                static_cast<void>(cudaEventDestroy(state.activation_downloaded));
            }
            if (state.moe_start != nullptr) {
                static_cast<void>(cudaEventDestroy(state.moe_start));
                static_cast<void>(cudaEventDestroy(state.moe_hidden_uploaded));
                static_cast<void>(cudaEventDestroy(state.moe_kernel_finished));
                static_cast<void>(cudaEventDestroy(state.moe_download_started));
                static_cast<void>(cudaEventDestroy(state.moe_completed));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_input_finished));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_gate_up_finished));
                static_cast<void>(cudaEventDestroy(
                    state.moe_shared_activation_finished));
                static_cast<void>(cudaEventDestroy(state.moe_shared_finished));
                static_cast<void>(cudaEventDestroy(
                    state.dsv4_cross_device_ready));
                static_cast<void>(cudaEventDestroy(
                    state.dsv4_attention_input_ready));
                for (auto event : state.dsv4_attention_aux_finished) {
                    static_cast<void>(cudaEventDestroy(event));
                }
            }
            if (state.upload_ready != nullptr) {
                static_cast<void>(cudaEventDestroy(state.upload_ready));
            }
            if (state.upload_stream != nullptr) {
                static_cast<void>(cudaStreamSynchronize(state.upload_stream));
                static_cast<void>(cudaStreamDestroy(state.upload_stream));
            }
            if (state.moe_shared_stream != nullptr) {
                static_cast<void>(cudaStreamSynchronize(
                    state.moe_shared_stream));
                static_cast<void>(cudaStreamDestroy(state.moe_shared_stream));
            }
            for (auto stream : state.dsv4_attention_aux_streams) {
                if (stream == nullptr) continue;
                static_cast<void>(cudaStreamSynchronize(stream));
                static_cast<void>(cudaStreamDestroy(stream));
            }
            if (state.cublas != nullptr) {
                static_cast<void>(cublasDestroy(state.cublas));
            }
            if (state.stream != nullptr) static_cast<void>(cudaStreamDestroy(state.stream));
        }
    }
};
