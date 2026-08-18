#pragma once

#include "strata/diagnostics.hpp"

#include <cstdint>
#include <ostream>
#include <string>

namespace strata {

[[nodiscard]] std::string hex_u64(std::uint64_t value);

// Emits the layer_hidden_hashes object body -- the value a caller's own
// "layer_hidden_hashes" key should map to:
//   {"enabled":bool[,"aggregate":{"entry_count":N,"trace_hash":".."},"entries":[...]]}
// Byte-identical to what strata-deepseek-run and strata-gemma4-run each
// hand-copied before this promotion (brief 05, F9) -- a pure move, not a
// reconciliation of the two callers' surrounding structure, which differs
// (DeepSeek nests this inside a larger diagnostics object with a logits
// section before it and an index_selections section after; Gemma 4 does
// not have either).
void print_layer_hidden_hashes_object(
    std::ostream& output, const DiagnosticTrace& diagnostics);

// Emits the operation-hash aggregate and array as two sibling fields, each
// prefixed with its own leading comma:
//   ,"operation_hash_trace_hash":".."
//   ,"operation_hashes":[...]
// Deliberately does not decide *whether* to call this -- DeepSeek and
// Gemma 4 diverged at birth on that question (DeepSeek additionally
// requires the list be non-empty; Gemma 4 does not), and reconciling that is
// a behaviour change this promotion does not make. Each caller keeps its
// own guard condition and calls this only when it decides to.
void print_operation_hashes_fields(
    std::ostream& output, const DiagnosticTrace& diagnostics);

}  // namespace strata
