#pragma once

#include <cstdint>
#include <random>
#include <span>

namespace strata {

[[nodiscard]] std::uint32_t sample_logits_gumbel(
    std::span<const float> logits, double temperature, std::mt19937_64& generator);

}  // namespace strata
