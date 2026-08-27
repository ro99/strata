#pragma once

// Storage-format quantization vocabulary.
//
// These describe how weights are *encoded*, not what architecture uses them,
// so they belong below the model tier. They lived in model.hpp only because
// that is where the first consumer needed them, which forced
// compressed_tensors.hpp -- a strata_platform header with no model dependency
// of its own -- to include the header that declares all six models' Spec
// types. That was the last recorded layering exception.

#include "strata/platform/types.hpp"

#include <cstdint>

namespace strata {

enum class QuantizationGranularity : std::uint8_t {
    Group,
    Channel,
};

struct QuantizedWeightSpec {
    std::uint32_t bits{kMinimumQuantBits};
    QuantizationGranularity granularity{QuantizationGranularity::Group};
    std::uint32_t group_size{};
    bool symmetric{true};
};

}  // namespace strata
