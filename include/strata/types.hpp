#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace strata {

inline constexpr std::uint32_t kMinimumQuantBits = 4;

enum class Tier : std::uint8_t {
    Vram,
    Ram,
    Peer,
    Nvme,
};

enum class ReplacementPolicy : std::uint8_t {
    Lru,
    Lfu,
    Lease,
};

struct ExpertKey {
    std::uint32_t layer{};
    std::uint32_t expert{};

    friend bool operator==(const ExpertKey&, const ExpertKey&) = default;
    friend bool operator<(const ExpertKey& lhs, const ExpertKey& rhs) {
        return lhs.layer < rhs.layer ||
               (lhs.layer == rhs.layer && lhs.expert < rhs.expert);
    }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey& key) const noexcept {
        const auto value = (static_cast<std::uint64_t>(key.layer) << 32U) |
                           static_cast<std::uint64_t>(key.expert);
        return std::hash<std::uint64_t>{}(value);
    }
};

[[nodiscard]] constexpr bool quantization_allowed(std::uint32_t bits) noexcept {
    return bits >= kMinimumQuantBits;
}

[[nodiscard]] std::string_view to_string(Tier tier) noexcept;
[[nodiscard]] std::string_view to_string(ReplacementPolicy policy) noexcept;

}  // namespace strata
