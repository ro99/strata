#pragma once

#include "strata/result.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

// Owns immutable checkpoint shard descriptors and provides bounded positional
// reads. Model-specific readers retain responsibility for tensor semantics and
// accounting.
class CheckpointShardSet {
public:
    CheckpointShardSet() = default;
    ~CheckpointShardSet();
    CheckpointShardSet(CheckpointShardSet&& other) noexcept;
    CheckpointShardSet& operator=(CheckpointShardSet&& other) noexcept;
    CheckpointShardSet(const CheckpointShardSet&) = delete;
    CheckpointShardSet& operator=(const CheckpointShardSet&) = delete;

    [[nodiscard]] ValidationResult open(
        const std::string& model_directory,
        const std::vector<std::string>& shards,
        std::string_view model_label);
    [[nodiscard]] ValidationResult read(
        std::string_view shard, std::uint64_t absolute_offset,
        std::span<std::byte> destination, std::string_view tensor_name) const;
    [[nodiscard]] int descriptor(std::string_view shard) const noexcept;

private:
    void close_all() noexcept;
    std::unordered_map<std::string, int> descriptors_;
};

}  // namespace strata
