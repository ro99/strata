#pragma once

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_admission.hpp"
#include "strata/deepseek_checkpoint.hpp"

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
    bool require_zero_nvme_decode{true};
    bool enable_dspark{};
    bool detailed_timing{};
    bool verbose{};
    std::string route_trace_path;
};

struct Dsv4CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_bytes;
};

struct Dsv4GenerationMetrics {
    double initialization_seconds{};
    double admission_seconds{};
    double resident_staging_seconds{};
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
    bool dspark_enabled{};
};

struct Dsv4GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    Dsv4GenerationMetrics metrics;
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
