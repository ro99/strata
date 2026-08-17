#pragma once

#include "strata/safetensors.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strata::detail {

inline float decode_plain_scalar(const std::byte* bytes,
                                 SafetensorsDtype dtype) noexcept {
    if (dtype == SafetensorsDtype::Bf16) {
        std::uint16_t value = 0U;
        std::memcpy(&value, bytes, sizeof(value));
        return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
    }
    if (dtype == SafetensorsDtype::F16) {
        std::uint16_t value = 0U;
        std::memcpy(&value, bytes, sizeof(value));
        const std::uint32_t sign = (value & 0x8000U) << 16U;
        std::uint32_t exponent = (value >> 10U) & 0x1FU;
        std::uint32_t mantissa = value & 0x3FFU;
        if (exponent == 0U) {
            if (mantissa == 0U) return std::bit_cast<float>(sign);
            std::int32_t shift = 0;
            while ((mantissa & 0x400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x3FFU;
            exponent = static_cast<std::uint32_t>(127 - 15 - shift);
        } else if (exponent == 0x1FU) {
            exponent = 0xFFU;
        } else {
            exponent += 127U - 15U;
        }
        return std::bit_cast<float>(sign | (exponent << 23U) | (mantissa << 13U));
    }
    float value = 0.0F;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

inline bool checked_product(std::uint64_t left, std::uint64_t right,
                            std::uint64_t& output) noexcept {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    output = left * right;
    return true;
}

}  // namespace strata::detail
