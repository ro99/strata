#pragma once

// Shared result mapping for the six ModelExecutor implementations.
//
// Every concrete runtime returns its own *GenerationResult type. The fields
// below exist on all six with identical meaning, so the copy is written once
// here instead of once per model -- which is what it was before Phase 4, and
// where two models quietly lost fields nobody noticed for a month.
//
// The two incremental-KV metrics are deliberately NOT here: only four of the
// six runtimes implement prefix reuse, and std::optional lets the other two
// say "not applicable" rather than report a zero indistinguishable from a
// measurement. Each executor sets them, or leaves them unset, explicitly.

#include "strata/model_executor.hpp"

#include <utility>

namespace strata::detail {

template <typename Concrete>
void copy_common_generation(GenerationResult& out, Concrete& in) {
    out.text = std::move(in.text);
    out.prompt_token_ids = std::move(in.prompt_token_ids);
    out.generated_token_ids = std::move(in.generated_token_ids);
    out.logprobs = std::move(in.logprobs);
    out.metrics.prompt_tokens = in.metrics.prompt_tokens;
    out.metrics.prefill_tokens = in.metrics.prefill_tokens;
    out.metrics.decode_tokens = in.metrics.decode_tokens;
    out.metrics.prefill_seconds = in.metrics.prefill_seconds;
    out.metrics.decode_seconds = in.metrics.decode_seconds;
    out.errors = std::move(in.errors);
    out.stopped = in.stopped;
}

}  // namespace strata::detail
