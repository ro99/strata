#include "strata/diagnostics_json.hpp"

#include <iomanip>
#include <sstream>

namespace strata {

std::string hex_u64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

void print_layer_hidden_hashes_object(
    std::ostream& output, const DiagnosticTrace& diagnostics) {
    output << "{\"enabled\":"
           << (diagnostics.layer_hash_trace_enabled ? "true" : "false");
    if (diagnostics.layer_hash_trace_enabled) {
        output << ",\"aggregate\":{\"entry_count\":"
               << diagnostics.layer_hashes.size()
               << ",\"trace_hash\":\""
               << hex_u64(diagnostics.layer_hash_trace_hash)
               << "\"},\"entries\":[";
        for (std::size_t index = 0U; index < diagnostics.layer_hashes.size(); ++index) {
            const auto& record = diagnostics.layer_hashes[index];
            if (index != 0U) output << ',';
            output << "{\"position\":" << record.position
                   << ",\"input_token\":" << record.input_token
                   << ",\"layer\":" << record.layer
                   << ",\"bf16_hash\":\"" << hex_u64(record.bf16_hash)
                   << "\"}";
        }
        output << ']';
    }
    output << '}';
}

void print_operation_hashes_fields(
    std::ostream& output, const DiagnosticTrace& diagnostics) {
    output << ",\"operation_hash_trace_hash\":\""
           << hex_u64(diagnostics.operation_hash_trace_hash) << '"';
    output << ",\"operation_hashes\":[";
    for (std::size_t index = 0U; index < diagnostics.operation_hashes.size();
         ++index) {
        const auto& record = diagnostics.operation_hashes[index];
        if (index != 0U) output << ',';
        output << "{\"position\":" << record.position
               << ",\"input_token\":" << record.input_token
               << ",\"layer\":" << record.layer
               << ",\"operation\":\"" << record.operation << '"'
               << ",\"bf16_hash\":\"" << hex_u64(record.bf16_hash)
               << "\"}";
    }
    output << ']';
}

}  // namespace strata
