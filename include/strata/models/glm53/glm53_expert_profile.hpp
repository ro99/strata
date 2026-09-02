#pragma once

#include <cstdint>
#include <span>

namespace strata {

// A checkpoint-format-independent ordering of representative GLM-5.3-Flash
// routed experts. Each value packs layer in the high seven bits and expert in
// the low nine bits. Admission consumes only what the discovered per-device
// arena can hold; the profile is a ranking, never a fixed hardware plan.
[[nodiscard]] std::span<const std::uint16_t>
glm53_default_expert_ranking() noexcept;

}  // namespace strata
