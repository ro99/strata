#pragma once

// This is the only production-code import boundary for the vendored Marlin
// subset. Keep third-party paths and Strata's required FP32 epilogue switch
// out of the rest of the backend.
#define STRATA_MARLIN_FP32_OUTPUT 1
#include "../../../third_party/marlin/marlin_template.h"
#undef STRATA_MARLIN_FP32_OUTPUT
