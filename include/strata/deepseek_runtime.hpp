#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_admission.hpp"
#include "strata/deepseek_checkpoint.hpp"
#include "strata/deepseek_diagnostics.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct Dsv4RuntimeConfig {
    std::vector<int> devices{0};
    double vram_cache_fraction{0.85};
    std::uint64_t host_memory_limit_bytes{216ULL << 30U};
    std::uint32_t maximum_context_tokens{2048U};
    std::uint32_t logit_trace_top_k{20U};
    std::uint32_t host_attention_threads{};
    std::uint32_t resident_read_workers{8U};
    std::uint32_t spine_warmup_workers{3U};
    bool require_zero_nvme_decode{true};
    bool enable_dspark{};
    bool enable_device_moe{};
    bool enable_logit_trace{};
    bool enable_layer_hash_trace{};
    bool detailed_timing{};
    bool overlap_resident_warmup{true};
    bool verbose{};
    std::string route_trace_path;
};

struct Dsv4CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::uint64_t lease_acquires{};
    std::uint64_t lease_releases{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_bytes;
    std::vector<std::uint64_t> leased_bytes;
    std::vector<std::uint64_t> active_leases;
};

struct Dsv4DeviceMoeStats {
    std::uint64_t batches{};
    std::uint64_t device_commands{};
    std::uint64_t routed_experts{};
    std::uint64_t shared_experts{};
    std::uint64_t nanoseconds{};
};

struct Dsv4PhaseMetrics {
    Dsv4CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Dsv4CacheStats cache;
    Dsv4DeviceMoeStats device_moe;
};

struct Dsv4GenerationMetrics {
    double initialization_seconds{};
    double admission_seconds{};
    double resident_staging_seconds{};
    double resident_warmup_seconds{};
    double prefill_seconds{};
    double decode_seconds{};
    std::uint64_t prompt_tokens{};
    std::uint64_t decode_tokens{};
    Dsv4MemoryPlan memory;
    Dsv4ResidentStageStats resident_stage;
    Dsv4CheckpointReadStats generation_checkpoint_reads;
    Dsv4CheckpointReadStats decode_checkpoint_reads;
    CudaBackendStats cuda;
    Dsv4CacheStats cache;
    Dsv4DeviceMoeStats device_moe;
    Dsv4PhaseMetrics prefill;
    Dsv4PhaseMetrics decode;
    bool detailed_timing{};
    bool dspark_enabled{};
    bool device_moe_enabled{};
    bool resident_warmup_overlapped{};
    std::uint32_t host_attention_threads{};
    std::uint32_t resident_read_workers{};
    std::uint32_t spine_warmup_workers{};
};

struct Dsv4GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    Dsv4GenerationMetrics metrics;
    Dsv4DiagnosticTrace diagnostics;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class DeepSeekV4Runtime {
public:
    DeepSeekV4Runtime();
    ~DeepSeekV4Runtime();
    DeepSeekV4Runtime(DeepSeekV4Runtime&&) noexcept;
    DeepSeekV4Runtime& operator=(DeepSeekV4Runtime&&) noexcept;
    DeepSeekV4Runtime(const DeepSeekV4Runtime&) = delete;
    DeepSeekV4Runtime& operator=(const DeepSeekV4Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory, const Dsv4RuntimeConfig& config);
    [[nodiscard]] Dsv4GenerationResult generate(
        std::string_view prompt, std::uint32_t maximum_new_tokens);
    [[nodiscard]] const Dsv4MemoryPlan& memory_plan() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
