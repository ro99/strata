#pragma once

#include "strata/result.hpp"

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace strata {

struct Gemma4PreparedImage {
    std::vector<float> patches;
    std::vector<std::int32_t> positions;
    std::uint32_t patch_width{};
    std::uint32_t patch_height{};
    std::uint32_t soft_tokens{};
};

[[nodiscard]] ParseResult<std::pair<std::uint32_t, std::uint32_t>>
gemma4_target_image_size(std::uint32_t width, std::uint32_t height);

[[nodiscard]] ParseResult<Gemma4PreparedImage> prepare_gemma4_image(
    std::string_view encoded, std::string_view mime_type);

}  // namespace strata
