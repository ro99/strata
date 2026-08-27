#pragma once

#include <cstdint>

// Standalone compile-time subset of vLLM's ScalarType.  The imported Marlin
// templates use only stable type identity and bit width; no Torch type or
// runtime API is needed by Strata's BF16/MXFP4 specialization.
namespace vllm {

struct ScalarType {
    using Id = std::int64_t;

    Id value;
    int bits;

    [[nodiscard]] __host__ __device__ constexpr Id id() const noexcept {
        return value;
    }
    [[nodiscard]] __host__ __device__ constexpr std::int64_t size_bits()
        const noexcept {
        return bits;
    }
    [[nodiscard]] __host__ __device__ static constexpr ScalarType from_id(
        Id id) noexcept {
        return {id, id == 1 || id == 2 || id == 3 || id == 7 ? 4
                    : id == 4 || id == 5 || id == 6 || id == 8 || id == 9
                        ? 8
                        : 16};
    }
    [[nodiscard]] __host__ __device__ constexpr bool operator==(
        ScalarType other) const noexcept {
        return value == other.value;
    }
    [[nodiscard]] __host__ __device__ constexpr bool operator!=(
        ScalarType other) const noexcept {
        return !(*this == other);
    }
};

using ScalarTypeId = ScalarType::Id;

inline constexpr ScalarType kS4{1, 4};
inline constexpr ScalarType kU4{2, 4};
inline constexpr ScalarType kU4B8{3, 4};
inline constexpr ScalarType kS8{4, 8};
inline constexpr ScalarType kU8{5, 8};
inline constexpr ScalarType kU8B128{6, 8};
inline constexpr ScalarType kFE2M1f{7, 4};
inline constexpr ScalarType kFE4M3fn{8, 8};
inline constexpr ScalarType kFE8M0fnu{9, 8};
inline constexpr ScalarType kFloat16{10, 16};
inline constexpr ScalarType kBFloat16{11, 16};

}  // namespace vllm
