#pragma once

#include "strata/checkpoint.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

struct Glm52RuntimeConfig {
    std::vector<int> devices{0, 1, 2};
    double vram_cache_fraction{0.85};
    std::uint32_t maximum_context_tokens{256};
    bool verbose{true};
    bool diagnostic_trace{};
    bool detailed_timing{};
    std::string route_trace_path;
    std::uint64_t request_id{};
};

struct Glm52CacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t evictions{};
    std::vector<std::uint64_t> used_bytes;
    std::vector<std::uint64_t> peak_bytes;
    std::vector<std::uint64_t> capacity_bytes;
    std::vector<std::uint64_t> pinned_resident_bytes;
    std::vector<std::uint64_t> evictable_expert_bytes;
    std::vector<std::uint64_t> device_hits;
    std::vector<std::uint64_t> device_misses;
    std::vector<std::uint64_t> device_evictions;
};

struct Glm52PhaseMetrics {
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Glm52CacheStats cache;
    std::uint64_t host_aggregation_nanoseconds{};
};

struct Glm52RunMetrics {
    std::uint64_t prompt_tokens{};
    std::uint64_t decode_tokens{};
    double prefill_seconds{};
    double decode_seconds{};
    CheckpointReadStats checkpoint_reads;
    CudaBackendStats cuda;
    Glm52CacheStats cache;
    Glm52PhaseMetrics prefill;
    Glm52PhaseMetrics decode;
    bool detailed_timing{};

    [[nodiscard]] double prefill_tokens_per_second() const noexcept {
        return prefill_seconds > 0.0 ? static_cast<double>(prompt_tokens) / prefill_seconds : 0.0;
    }
    [[nodiscard]] double decode_tokens_per_second() const noexcept {
        return decode_seconds > 0.0 ? static_cast<double>(decode_tokens) / decode_seconds : 0.0;
    }
};

struct Glm52GenerationResult {
    std::string text;
    std::vector<std::uint32_t> prompt_token_ids;
    std::vector<std::uint32_t> generated_token_ids;
    Glm52RunMetrics metrics;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

class Glm52Runtime {
public:
    Glm52Runtime();
    ~Glm52Runtime();
    Glm52Runtime(Glm52Runtime&&) noexcept;
    Glm52Runtime& operator=(Glm52Runtime&&) noexcept;
    Glm52Runtime(const Glm52Runtime&) = delete;
    Glm52Runtime& operator=(const Glm52Runtime&) = delete;

    [[nodiscard]] ValidationResult initialize(
        const std::string& model_directory, const Glm52RuntimeConfig& config = {});
    [[nodiscard]] Glm52GenerationResult generate(
        std::string_view prompt, std::uint32_t maximum_new_tokens);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
