#include "test.hpp"

#include <cstdint>
#include <cstring>

namespace {

// The declared E2M1 table, as the CUDA backend's switch spelled it.
[[nodiscard]] float declared_e2m1(unsigned int encoded) {
    switch (encoded & 0x0FU) {
        case 0x0U: return 0.0F;
        case 0x1U: return 0.5F;
        case 0x2U: return 1.0F;
        case 0x3U: return 1.5F;
        case 0x4U: return 2.0F;
        case 0x5U: return 3.0F;
        case 0x6U: return 4.0F;
        case 0x7U: return 6.0F;
        case 0x8U: return 0.0F;
        case 0x9U: return -0.5F;
        case 0xAU: return -1.0F;
        case 0xBU: return -1.5F;
        case 0xCU: return -2.0F;
        case 0xDU: return -3.0F;
        case 0xEU: return -4.0F;
        default: return -6.0F;
    }
}

// Host mirror of fp4_e2m1_value in kernels/cuda/backend.cu. The device cannot
// be called from here, so the arithmetic is duplicated and pinned instead: if
// the kernel's construction is edited, this must be edited to match, and this
// test is what says the two agree with the declared table.
[[nodiscard]] float constructed_e2m1(unsigned int encoded) {
    const unsigned int magnitude = encoded & 0x07U;
    const unsigned int exponent = magnitude >> 1U;
    const unsigned int mantissa = magnitude & 0x01U;
    const unsigned int normal = ((126U + exponent) << 23U) | (mantissa << 22U);
    const unsigned int subnormal = mantissa == 0U ? 0U : 0x3F000000U;
    const unsigned int bits = exponent == 0U ? subnormal : normal;
    const unsigned int sign = magnitude == 0U ? 0U : ((encoded & 0x08U) << 28U);
    const unsigned int combined = bits | sign;
    float value = 0.0F;
    std::memcpy(&value, &combined, sizeof(value));
    return value;
}

[[nodiscard]] std::uint32_t bits_of(float value) {
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

TEST_CASE("fp4 e2m1 branch-free decode is bit-identical to the declared table") {
    // Bit equality, not numeric equality: encoding 0x8 is the case that
    // separates them, because a sign bit on a zero magnitude compares equal to
    // +0.0 while carrying different bits into every downstream rounding.
    for (unsigned int encoded = 0U; encoded < 16U; ++encoded) {
        REQUIRE(bits_of(constructed_e2m1(encoded)) ==
                bits_of(declared_e2m1(encoded)));
    }
}

TEST_CASE("fp4 e2m1 decode covers the declared magnitudes and signs") {
    REQUIRE(constructed_e2m1(0x0U) == 0.0F);
    REQUIRE(constructed_e2m1(0x1U) == 0.5F);
    REQUIRE(constructed_e2m1(0x7U) == 6.0F);
    REQUIRE(constructed_e2m1(0x8U) == 0.0F);
    REQUIRE(bits_of(constructed_e2m1(0x8U)) == 0U);  // +0.0, never -0.0
    REQUIRE(constructed_e2m1(0xFU) == -6.0F);
}
