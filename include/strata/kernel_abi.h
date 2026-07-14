#ifndef STRATA_KERNEL_ABI_H
#define STRATA_KERNEL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum strata_quant_format {
    STRATA_Q4 = 4,
    STRATA_Q8 = 8,
    STRATA_F16 = 16,
    STRATA_BF16 = 17,
    STRATA_F32 = 32,
};

/* Strata forbids every weight format below four bits. */
int strata_quant_format_valid(int format);

/* Signed q4: two two's-complement nibbles per byte. */
int8_t strata_q4_unpack(const uint8_t *packed, size_t element);
void strata_q4_pack(uint8_t *packed, size_t element, int8_t value);

/* Numerical oracle: y[rows] = W_q4[rows,cols] * x[cols], per-row scale. */
int strata_q4_matvec_f32(float *y, const float *x, const uint8_t *weights,
                         const float *scales, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif
