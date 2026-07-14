#include "strata/kernel_abi.h"

#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" int strata_quant_format_valid(int format) {
    return format == STRATA_Q4 || format == STRATA_Q8 || format == STRATA_F16 ||
           format == STRATA_BF16 || format == STRATA_F32;
}

extern "C" int8_t strata_q4_unpack(const uint8_t *packed, size_t element) {
    if (packed == nullptr) return 0;
    const uint8_t byte = packed[element / 2U];
    const uint8_t nibble = (element & 1U) == 0U ? byte & 0x0FU : byte >> 4U;
    return static_cast<int8_t>((nibble & 0x08U) != 0U ?
                               static_cast<int>(nibble) - 16 :
                               static_cast<int>(nibble));
}

extern "C" void strata_q4_pack(uint8_t *packed, size_t element, int8_t value) {
    if (packed == nullptr) return;
    const int clamped = std::clamp(static_cast<int>(value), -8, 7);
    const auto nibble = static_cast<uint8_t>(clamped & 0x0F);
    auto& byte = packed[element / 2U];
    if ((element & 1U) == 0U) {
        byte = static_cast<uint8_t>((byte & 0xF0U) | nibble);
    } else {
        byte = static_cast<uint8_t>((byte & 0x0FU) | static_cast<uint8_t>(nibble << 4U));
    }
}

extern "C" int strata_q4_matvec_f32(float *y, const float *x,
                                      const uint8_t *weights, const float *scales,
                                      size_t rows, size_t cols) {
    if (y == nullptr || x == nullptr || weights == nullptr || scales == nullptr ||
        rows == 0 || cols == 0) {
        return 0;
    }
    const size_t row_bytes = (cols + 1U) / 2U;
    for (size_t row = 0; row < rows; ++row) {
        float sum = 0.0F;
        const uint8_t *packed_row = weights + row * row_bytes;
        for (size_t col = 0; col < cols; ++col) {
            sum += x[col] * static_cast<float>(strata_q4_unpack(packed_row, col));
        }
        y[row] = sum * scales[row];
    }
    return 1;
}
