#include "strata/deepseek_runtime.hpp"

#include "cli_common.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt{"Hello"};
    std::vector<int> devices{0, 1, 2};
    std::uint32_t maximum_new_tokens{16U};
    std::uint32_t maximum_context_tokens{2048U};
    std::uint32_t prefill_page_tokens{64U};
    std::uint32_t logit_trace_top_k{20U};
    std::uint32_t host_attention_threads{28U};
    std::uint32_t flash_attention_minimum_rows{256U};
    std::uint32_t resident_read_workers{8U};
    std::uint32_t spine_warmup_workers{3U};
    std::uint64_t host_memory_bytes{216ULL << 30U};
    std::uint64_t host_kv_cache_bytes{};
    std::vector<std::uint64_t> device_kv_cache_bytes;
    double vram_fraction{0.85};
    bool admission_only{};
    bool json{};
    bool quiet{};
    bool detailed_timing{};
    bool device_moe{true};
    bool flash_attention{};
    bool block_kv_cache{};
    bool logit_trace{};
    bool layer_hash_trace{};
    bool overlap_resident_warmup{true};
    std::string route_trace;
};

void usage() {
    std::cerr
        << "usage: strata-deepseek-run --model DIR [--prompt TEXT] [--max-new N]\n"
        << "       [--devices 0,1,2] [--max-context N] [--host-memory 216G]\n"
        << "       [--prefill-page-tokens N]\n"
        << "       [--host-attention-threads N|--serial-host-attention]\n"
        << "       [--resident-read-workers N]\n"
        << "       [--spine-warmup-workers N]\n"
        << "       [--overlap-resident-warmup|--serial-resident-warmup]\n"
        << "       [--vram-fraction F] [--admission-only] [--route-trace PATH]\n"
        << "       [--device-moe|--serial-device-moe]\n"
        << "       [--flash-attention|--scalar-attention]\n"
        << "       [--block-kv-cache|--scalar-kv-cache]\n"
        << "       [--kv-host-cache BYTES] [--kv-device-cache B0,B1,...]\n"
        << "       [--flash-attention-minimum-rows N]\n"
        << "       [--logit-trace] [--logit-trace-top-k 20] [--layer-hash-trace]\n"
        << "       [--detailed-timing] [--json] [--quiet]\n";
}

bool parse_bytes(std::string_view text, std::uint64_t& value) {
    if (text.empty()) return false;
    std::uint64_t multiplier = 1U;
    const char suffix = text.back();
    if (suffix == 'K' || suffix == 'k') {
        multiplier = 1ULL << 10U;
        text.remove_suffix(1U);
    } else if (suffix == 'M' || suffix == 'm') {
        multiplier = 1ULL << 20U;
        text.remove_suffix(1U);
    } else if (suffix == 'G' || suffix == 'g') {
        multiplier = 1ULL << 30U;
        text.remove_suffix(1U);
    }
    std::uint64_t base = 0U;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), base);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() ||
        base > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        return false;
    }
    value = base * multiplier;
    return true;
}

bool parse_byte_list(std::string_view text,
                     std::vector<std::uint64_t>& values) {
    values.clear();
    while (!text.empty()) {
        const auto separator = text.find(',');
        const auto item = text.substr(0U, separator);
        std::uint64_t value = 0U;
        if (!parse_bytes(item, value)) return false;
        values.push_back(value);
        if (separator == std::string_view::npos) return true;
        text.remove_prefix(separator + 1U);
    }
    return false;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](std::string_view name) -> const char* {
            if (index + 1 >= argc) {
                std::cerr << "missing value for " << name << '\n';
                return nullptr;
            }
            return argv[++index];
        };
        if (argument == "--model") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.model = value;
        } else if (argument == "--prompt") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.prompt = value;
        } else if (argument == "--devices") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_devices(value, options.devices)) return false;
        } else if (argument == "--max-new") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.maximum_new_tokens)) return false;
        } else if (argument == "--max-context") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.maximum_context_tokens)) return false;
        } else if (argument == "--prefill-page-tokens") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(
                    value, options.prefill_page_tokens) ||
                options.prefill_page_tokens == 0U ||
                options.prefill_page_tokens > 512U) return false;
        } else if (argument == "--host-attention-threads") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.host_attention_threads) ||
                options.host_attention_threads > 64U) return false;
        } else if (argument == "--serial-host-attention") {
            options.host_attention_threads = 0U;
        } else if (argument == "--resident-read-workers") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.resident_read_workers) ||
                options.resident_read_workers == 0U ||
                options.resident_read_workers > 64U) return false;
        } else if (argument == "--spine-warmup-workers") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.spine_warmup_workers) ||
                options.spine_warmup_workers == 0U ||
                options.spine_warmup_workers > 64U) return false;
        } else if (argument == "--host-memory") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_bytes(value, options.host_memory_bytes)) return false;
        } else if (argument == "--kv-host-cache") {
            const auto* value = next(argument);
            if (value == nullptr ||
                !parse_bytes(value, options.host_kv_cache_bytes)) return false;
        } else if (argument == "--kv-device-cache") {
            const auto* value = next(argument);
            if (value == nullptr || !parse_byte_list(
                    value, options.device_kv_cache_bytes)) return false;
        } else if (argument == "--vram-fraction") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            if (!strata::cli::parse_double(value, options.vram_fraction, 0.0, 0.95)) return false;
        } else if (argument == "--route-trace") {
            const auto* value = next(argument);
            if (value == nullptr) return false;
            options.route_trace = value;
        } else if (argument == "--device-moe") {
            options.device_moe = true;
        } else if (argument == "--serial-device-moe") {
            options.device_moe = false;
        } else if (argument == "--flash-attention") {
            options.flash_attention = true;
        } else if (argument == "--scalar-attention") {
            options.flash_attention = false;
        } else if (argument == "--block-kv-cache") {
            options.block_kv_cache = true;
        } else if (argument == "--scalar-kv-cache") {
            options.block_kv_cache = false;
        } else if (argument == "--flash-attention-minimum-rows") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(
                    value, options.flash_attention_minimum_rows)) return false;
        } else if (argument == "--logit-trace") {
            options.logit_trace = true;
        } else if (argument == "--logit-trace-top-k") {
            const auto* value = next(argument);
            if (value == nullptr || !strata::cli::parse_u32(value, options.logit_trace_top_k) ||
                options.logit_trace_top_k == 0U) return false;
            options.logit_trace = true;
        } else if (argument == "--layer-hash-trace") {
            options.layer_hash_trace = true;
        } else if (argument == "--overlap-resident-warmup") {
            options.overlap_resident_warmup = true;
        } else if (argument == "--serial-resident-warmup") {
            options.overlap_resident_warmup = false;
        } else if (argument == "--admission-only") {
            options.admission_only = true;
        } else if (argument == "--detailed-timing") {
            options.detailed_timing = true;
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << argument << '\n';
            return false;
        }
    }
    return !options.model.empty();
}

void print_cuda_stats(std::ostream& output, const strata::CudaBackendStats& stats) {
    output << "{\"weight_h2d_bytes\":" << stats.weight_upload_bytes
           << ",\"activation_h2d_bytes\":" << stats.activation_h2d_bytes
           << ",\"activation_d2h_bytes\":" << stats.activation_d2h_bytes
           << ",\"matmul_calls\":" << stats.matmul_calls
           << ",\"weight_allocation_calls\":" << stats.weight_allocation_calls
           << ",\"weight_allocation_bytes\":" << stats.weight_allocation_bytes
           << ",\"workspace_allocation_calls\":" << stats.workspace_allocation_calls
           << ",\"workspace_allocation_bytes\":" << stats.workspace_allocation_bytes
           << ",\"synchronization_calls\":" << stats.synchronization_calls
           << ",\"critical_path_synchronization_seconds\":"
           << static_cast<double>(stats.synchronization_nanoseconds) / 1.0e9
           << ",\"critical_path_upload_wait_seconds\":"
           << static_cast<double>(stats.upload_wait_nanoseconds) / 1.0e9
           << ",\"critical_path_activation_h2d_seconds\":"
           << static_cast<double>(stats.activation_h2d_nanoseconds) / 1.0e9
           << ",\"critical_path_kernel_seconds\":"
           << static_cast<double>(stats.kernel_nanoseconds) / 1.0e9
           << ",\"critical_path_activation_d2h_seconds\":"
           << static_cast<double>(stats.activation_d2h_nanoseconds) / 1.0e9
           << ",\"deepseek_moe_calls\":" << stats.deepseek_moe_calls
           << ",\"deepseek_moe_kernel_launches\":"
           << stats.deepseek_moe_kernel_launches
           << ",\"deepseek_moe_h2d_transfers\":"
           << stats.deepseek_moe_h2d_transfers
           << ",\"deepseek_moe_d2h_transfers\":"
           << stats.deepseek_moe_d2h_transfers
           << ",\"deepseek_moe_h2d_bytes\":" << stats.deepseek_moe_h2d_bytes
           << ",\"deepseek_moe_d2h_bytes\":" << stats.deepseek_moe_d2h_bytes
           << ",\"maximum_device_deepseek_moe_h2d_seconds\":"
           << static_cast<double>(stats.deepseek_moe_h2d_nanoseconds) / 1.0e9
           << ",\"maximum_device_deepseek_moe_kernel_seconds\":"
           << static_cast<double>(stats.deepseek_moe_kernel_nanoseconds) / 1.0e9
           << ",\"maximum_device_deepseek_moe_d2h_seconds\":"
           << static_cast<double>(stats.deepseek_moe_d2h_nanoseconds) / 1.0e9
           << ",\"maximum_device_deepseek_moe_seconds\":"
           << static_cast<double>(stats.deepseek_moe_nanoseconds) / 1.0e9
           << ",\"flash_attention_calls\":" << stats.flash_attention_calls
           << ",\"flash_attention_kernel_launches\":"
           << stats.flash_attention_kernel_launches
           << ",\"flash_attention_h2d_transfers\":"
           << stats.flash_attention_h2d_transfers
           << ",\"flash_attention_d2h_transfers\":"
           << stats.flash_attention_d2h_transfers
           << ",\"flash_attention_h2d_bytes\":"
           << stats.flash_attention_h2d_bytes
           << ",\"flash_attention_d2h_bytes\":"
           << stats.flash_attention_d2h_bytes
           << ",\"flash_attention_useful_staging_bytes\":"
           << stats.flash_attention_useful_staging_bytes
           << ",\"flash_attention_wasted_staging_bytes\":"
           << stats.flash_attention_wasted_staging_bytes
           << ",\"maximum_device_flash_attention_h2d_seconds\":"
           << static_cast<double>(stats.flash_attention_h2d_nanoseconds) / 1.0e9
           << ",\"maximum_device_flash_attention_kernel_seconds\":"
           << static_cast<double>(stats.flash_attention_kernel_nanoseconds) / 1.0e9
           << ",\"maximum_device_flash_attention_d2h_seconds\":"
           << static_cast<double>(stats.flash_attention_d2h_nanoseconds) / 1.0e9
           << ",\"maximum_device_flash_attention_seconds\":"
           << static_cast<double>(stats.flash_attention_nanoseconds) / 1.0e9
           << ",\"devices\":[";
    for (std::size_t index = 0U; index < stats.devices.size(); ++index) {
        const auto& device = stats.devices[index];
        if (index != 0U) output << ',';
        output << "{\"device\":" << device.device
               << ",\"weight_h2d_bytes\":" << device.weight_upload_bytes
               << ",\"activation_h2d_bytes\":" << device.activation_h2d_bytes
               << ",\"activation_d2h_bytes\":" << device.activation_d2h_bytes
               << ",\"matmul_calls\":" << device.matmul_calls
               << ",\"synchronization_calls\":" << device.synchronization_calls
               << ",\"synchronization_seconds\":"
               << static_cast<double>(device.synchronization_nanoseconds) / 1.0e9
               << ",\"upload_wait_seconds\":"
               << static_cast<double>(device.upload_wait_nanoseconds) / 1.0e9
               << ",\"activation_h2d_seconds\":"
               << static_cast<double>(device.activation_h2d_nanoseconds) / 1.0e9
               << ",\"kernel_seconds\":"
               << static_cast<double>(device.kernel_nanoseconds) / 1.0e9
               << ",\"activation_d2h_seconds\":"
               << static_cast<double>(device.activation_d2h_nanoseconds) / 1.0e9
               << ",\"deepseek_moe_calls\":" << device.deepseek_moe_calls
               << ",\"deepseek_moe_kernel_launches\":"
               << device.deepseek_moe_kernel_launches
               << ",\"deepseek_moe_h2d_transfers\":"
               << device.deepseek_moe_h2d_transfers
               << ",\"deepseek_moe_d2h_transfers\":"
               << device.deepseek_moe_d2h_transfers
               << ",\"deepseek_moe_h2d_bytes\":"
               << device.deepseek_moe_h2d_bytes
               << ",\"deepseek_moe_d2h_bytes\":"
               << device.deepseek_moe_d2h_bytes
               << ",\"deepseek_moe_h2d_seconds\":"
               << static_cast<double>(device.deepseek_moe_h2d_nanoseconds) / 1.0e9
               << ",\"deepseek_moe_kernel_seconds\":"
               << static_cast<double>(device.deepseek_moe_kernel_nanoseconds) / 1.0e9
               << ",\"deepseek_moe_d2h_seconds\":"
               << static_cast<double>(device.deepseek_moe_d2h_nanoseconds) / 1.0e9
               << ",\"deepseek_moe_seconds\":"
               << static_cast<double>(device.deepseek_moe_nanoseconds) / 1.0e9
               << ",\"flash_attention_calls\":"
               << device.flash_attention_calls
               << ",\"flash_attention_kernel_launches\":"
               << device.flash_attention_kernel_launches
               << ",\"flash_attention_h2d_transfers\":"
               << device.flash_attention_h2d_transfers
               << ",\"flash_attention_d2h_transfers\":"
               << device.flash_attention_d2h_transfers
               << ",\"flash_attention_h2d_bytes\":"
               << device.flash_attention_h2d_bytes
               << ",\"flash_attention_d2h_bytes\":"
               << device.flash_attention_d2h_bytes
               << ",\"flash_attention_useful_staging_bytes\":"
               << device.flash_attention_useful_staging_bytes
               << ",\"flash_attention_wasted_staging_bytes\":"
               << device.flash_attention_wasted_staging_bytes
               << ",\"flash_attention_h2d_seconds\":"
               << static_cast<double>(device.flash_attention_h2d_nanoseconds) / 1.0e9
               << ",\"flash_attention_kernel_seconds\":"
               << static_cast<double>(device.flash_attention_kernel_nanoseconds) / 1.0e9
               << ",\"flash_attention_d2h_seconds\":"
               << static_cast<double>(device.flash_attention_d2h_nanoseconds) / 1.0e9
               << ",\"flash_attention_seconds\":"
               << static_cast<double>(device.flash_attention_nanoseconds) / 1.0e9
               << '}';
    }
    output << "]}";
}

void print_cache_stats(std::ostream& output, const strata::Dsv4CacheStats& stats) {
    output << "{\"hits\":" << stats.hits << ",\"misses\":" << stats.misses
           << ",\"evictions\":" << stats.evictions
           << ",\"lease_acquires\":" << stats.lease_acquires
           << ",\"lease_releases\":" << stats.lease_releases
           << ",\"used_bytes\":";
    strata::cli::print_array(output, stats.used_bytes);
    output << ",\"capacity_bytes\":";
    strata::cli::print_array(output, stats.capacity_bytes);
    output << ",\"pinned_bytes\":";
    strata::cli::print_array(output, stats.pinned_bytes);
    output << ",\"leased_bytes\":";
    strata::cli::print_array(output, stats.leased_bytes);
    output << ",\"active_leases\":";
    strata::cli::print_array(output, stats.active_leases);
    output << '}';
}

void print_kv_cache_stats(
    std::ostream& output, const strata::Dsv4KvCacheStats& stats) {
    output << "{\"host_capacity_bytes\":" << stats.host_capacity_bytes
           << ",\"host_used_bytes\":" << stats.host_used_bytes
           << ",\"host_peak_bytes\":" << stats.host_peak_bytes
           << ",\"device_capacity_bytes\":";
    strata::cli::print_array(output, stats.device_capacity_bytes);
    output << ",\"device_used_bytes\":";
    strata::cli::print_array(output, stats.device_used_bytes);
    output << ",\"device_peak_bytes\":";
    strata::cli::print_array(output, stats.device_peak_bytes);
    output << ",\"allocated_blocks\":" << stats.allocated_blocks
           << ",\"used_blocks\":" << stats.used_blocks
           << ",\"allocation_calls\":" << stats.allocation_calls
           << ",\"allocation_seconds\":"
           << static_cast<double>(stats.allocation_nanoseconds) / 1.0e9
           << ",\"hits\":" << stats.hits
           << ",\"misses\":" << stats.misses
           << ",\"evictions\":" << stats.evictions
           << ",\"promotions\":" << stats.promotions
           << ",\"promotion_seconds\":"
           << static_cast<double>(stats.promotion_nanoseconds) / 1.0e9
           << ",\"host_to_device_bytes\":" << stats.host_to_device_bytes
           << ",\"device_to_host_bytes\":" << stats.device_to_host_bytes
           << ",\"host_write_bytes\":" << stats.host_write_bytes
           << ",\"gather_bytes\":" << stats.gather_bytes
           << ",\"copy_on_write_blocks\":" << stats.copy_on_write_blocks
           << '}';
}

void print_device_moe_stats(
    std::ostream& output, const strata::Dsv4DeviceMoeStats& stats) {
    output << "{\"batches\":" << stats.batches
           << ",\"device_commands\":" << stats.device_commands
           << ",\"routed_experts\":" << stats.routed_experts
           << ",\"shared_experts\":" << stats.shared_experts
           << ",\"execution_seconds\":"
           << static_cast<double>(stats.nanoseconds) / 1.0e9
           << '}';
}

void print_graph_stats(std::ostream& output, const strata::Dsv4GraphStats& stats) {
    const auto seconds = [](std::uint64_t nanoseconds) {
        return static_cast<double>(nanoseconds) / 1.0e9;
    };
    output << "{\"forward_tokens\":" << stats.forward_tokens
           << ",\"prefill_pages\":" << stats.prefill_pages
           << ",\"prefill_max_page_tokens\":"
           << stats.prefill_max_page_tokens
           << ",\"prefill_max_workspace_bytes\":"
           << stats.prefill_max_workspace_bytes
           << ",\"embedding_seconds\":" << seconds(stats.embedding_nanoseconds)
           << ",\"mhc_pre_seconds\":" << seconds(stats.mhc_pre_nanoseconds)
           << ",\"branch_norm_seconds\":"
           << seconds(stats.branch_norm_nanoseconds)
           << ",\"attention_seconds\":" << seconds(stats.attention_nanoseconds)
           << ",\"attention_query_seconds\":"
           << seconds(stats.attention_query_nanoseconds)
           << ",\"attention_kv_seconds\":"
           << seconds(stats.attention_kv_nanoseconds)
           << ",\"attention_projection_matmul_calls\":"
           << stats.attention_projection_matmul_calls
           << ",\"attention_projection_matmul_rows\":"
           << stats.attention_projection_matmul_rows
           << ",\"attention_index_seconds\":"
           << seconds(stats.attention_index_nanoseconds)
           << ",\"attention_index_queries\":"
           << stats.attention_index_queries
           << ",\"attention_index_candidates\":"
           << stats.attention_index_candidates
           << ",\"attention_index_selected\":"
           << stats.attention_index_selected
           << ",\"attention_cuda_dispatches\":"
           << stats.attention_cuda_dispatches
           << ",\"attention_scalar_dispatches\":"
           << stats.attention_scalar_dispatches
           << ",\"attention_score_seconds\":"
           << seconds(stats.attention_score_nanoseconds)
           << ",\"attention_output_seconds\":"
           << seconds(stats.attention_output_nanoseconds)
           << ",\"moe_seconds\":" << seconds(stats.moe_nanoseconds)
           << ",\"moe_router_seconds\":"
           << seconds(stats.moe_router_nanoseconds)
           << ",\"moe_prepare_seconds\":"
           << seconds(stats.moe_prepare_nanoseconds)
           << ",\"mhc_post_seconds\":" << seconds(stats.mhc_post_nanoseconds)
           << ",\"output_head_seconds\":"
           << seconds(stats.output_head_nanoseconds)
           << '}';
}

void print_phase(std::ostream& output, const strata::Dsv4PhaseMetrics& phase) {
    output << "{\"checkpoint_read_calls\":" << phase.checkpoint_reads.calls
           << ",\"checkpoint_read_bytes\":" << phase.checkpoint_reads.bytes
           << ",\"checkpoint_read_seconds\":"
           << static_cast<double>(phase.checkpoint_reads.nanoseconds) / 1.0e9
           << ",\"cuda\":";
    print_cuda_stats(output, phase.cuda);
    output << ",\"cache\":";
    print_cache_stats(output, phase.cache);
    output << ",\"kv_cache\":";
    print_kv_cache_stats(output, phase.kv_cache);
    output << ",\"device_moe_runtime\":";
    print_device_moe_stats(output, phase.device_moe);
    output << ",\"graph\":";
    print_graph_stats(output, phase.graph);
    output << '}';
}

void print_plan(std::ostream& output, const strata::Dsv4MemoryPlan& plan) {
    output << "{\"required_host_bytes\":" << plan.required_host_bytes
           << ",\"routed_expert_host_bytes\":" << plan.routed_expert_host_bytes
           << ",\"host_parameter_bytes\":" << plan.host_parameter_bytes
           << ",\"kv_state_bytes\":" << plan.kv_state_bytes
           << ",\"index_state_bytes\":" << plan.index_state_bytes
           << ",\"kv_cache_payload_bytes\":" << plan.kv_cache_payload_bytes
           << ",\"kv_cache_metadata_bytes\":" << plan.kv_cache_metadata_bytes
           << ",\"kv_cache_alignment_bytes\":" << plan.kv_cache_alignment_bytes
           << ",\"host_kv_cache_bytes\":" << plan.host_kv_cache_bytes
           << ",\"device_kv_cache_bytes\":" << plan.device_kv_cache_bytes
           << ",\"per_device_kv_cache_bytes\":";
    strata::cli::print_array(output, plan.per_device_kv_cache_bytes);
    output
           << ",\"host_workspace_bytes\":" << plan.host_workspace_bytes
           << ",\"total_vram_budget_bytes\":" << plan.total_vram_budget_bytes
           << ",\"resident_spine_vram_bytes\":" << plan.resident_spine_vram_bytes
           << ",\"vram_workspace_bytes\":" << plan.vram_workspace_bytes
           << ",\"expert_vram_cache_bytes\":" << plan.expert_vram_cache_bytes
           << ",\"maximum_expert_placement_bytes\":" << plan.maximum_expert_bytes
           << ",\"steady_state_nvme_bytes\":" << plan.steady_state_nvme_bytes
           << ",\"maximum_context_tokens\":" << plan.maximum_context_tokens
           << ",\"zero_nvme_decode\":" << (plan.zero_nvme_decode ? "true" : "false")
           << ",\"dspark_enabled\":" << (plan.dspark_enabled ? "true" : "false")
           << '}';
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

template <typename T>
void print_json_number(std::ostream& output, T value) {
    if (std::isfinite(value)) output << value;
    else output << "null";
}

void print_logit_summary(std::ostream& output,
                         const strata::Dsv4LogitSummary& summary) {
    output << "{\"value_count\":" << summary.value_count
           << ",\"finite_count\":" << summary.finite_count
           << ",\"non_finite_count\":" << summary.non_finite_count
           << ",\"sum\":";
    print_json_number(output, summary.sum);
    output << ",\"absolute_sum\":";
    print_json_number(output, summary.absolute_sum);
    output << ",\"square_sum\":";
    print_json_number(output, summary.square_sum);
    output << ",\"minimum\":";
    if (summary.has_finite) print_json_number(output, summary.minimum);
    else output << "null";
    output << ",\"maximum\":";
    if (summary.has_finite) print_json_number(output, summary.maximum);
    else output << "null";
    output << ",\"raw_f32_hash\":\"" << hex_u64(summary.raw_f32_hash)
           << "\"}";
}

void print_diagnostics(std::ostream& output,
                       const strata::Dsv4DiagnosticTrace& diagnostics) {
    const auto previous_precision = output.precision();
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "{\"hash_algorithm\":\"fnv1a64-little-endian\""
           << ",\"logits\":{\"enabled\":"
           << (diagnostics.logit_trace_enabled ? "true" : "false")
           << ",\"top_k\":" << diagnostics.logit_top_k;
    if (diagnostics.logit_trace_enabled) {
        const auto& aggregate = diagnostics.logit_aggregate;
        output << ",\"aggregate\":{\"forward_count\":"
               << aggregate.forward_count
               << ",\"value_count\":" << aggregate.value_count
               << ",\"finite_count\":" << aggregate.finite_count
               << ",\"non_finite_count\":" << aggregate.non_finite_count
               << ",\"sum\":";
        print_json_number(output, aggregate.sum);
        output << ",\"absolute_sum\":";
        print_json_number(output, aggregate.absolute_sum);
        output << ",\"square_sum\":";
        print_json_number(output, aggregate.square_sum);
        output << ",\"minimum\":";
        if (aggregate.has_finite) print_json_number(output, aggregate.minimum);
        else output << "null";
        output << ",\"maximum\":";
        if (aggregate.has_finite) print_json_number(output, aggregate.maximum);
        else output << "null";
        output << ",\"trace_hash\":\"" << hex_u64(aggregate.trace_hash)
               << "\"},\"forwards\":[";
        for (std::size_t index = 0U; index < diagnostics.logits.size(); ++index) {
            const auto& record = diagnostics.logits[index];
            if (index != 0U) output << ',';
            output << "{\"position\":" << record.position
                   << ",\"input_token\":" << record.input_token
                   << ",\"selected_token\":" << record.selected_token
                   << ",\"summary\":";
            print_logit_summary(output, record.summary);
            output << ",\"top\":[";
            for (std::size_t rank = 0U; rank < record.top.size(); ++rank) {
                if (rank != 0U) output << ',';
                output << "{\"token_id\":" << record.top[rank].token_id
                       << ",\"raw_logit\":";
                print_json_number(output, record.top[rank].raw_logit);
                output << '}';
            }
            output << "]}";
        }
        output << ']';
    }
    output << "},\"layer_hidden_hashes\":{\"enabled\":"
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
    if (diagnostics.layer_hash_trace_enabled && !diagnostics.operation_hashes.empty()) {
        output << ",\"operation_hashes\":[";
        for (std::size_t index = 0U; index < diagnostics.operation_hashes.size(); ++index) {
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
    output << '}';
    output.precision(previous_precision);
}

bool device_budgets(const Options& options, std::vector<std::uint64_t>& budgets,
                    std::vector<std::string>& errors) {
    if (!std::isfinite(options.vram_fraction) || options.vram_fraction <= 0.0 ||
        options.vram_fraction > 0.95) {
        errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
        return false;
    }
    for (const auto device : options.devices) {
        auto memory = strata::CudaBackend::device_memory(device);
        if (!memory.ok()) {
            errors.insert(errors.end(), memory.errors.begin(), memory.errors.end());
            return false;
        }
        budgets.push_back(static_cast<std::uint64_t>(
            static_cast<double>(memory.value.free_bytes) * options.vram_fraction));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
        static_cast<void>(setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0));
    }
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    if (!options.quiet) {
        std::cerr
            << "[contract] exact DeepSeek-V4-Flash base-model greedy decode\n"
            << "[contract] FP4/FP8 checkpoint semantics unchanged; "
               "user-sized context <=1048576\n"
            << "[contract] experts and embedding resident in RAM; decode NVMe bytes = 0\n"
            << "[contract] optional DSpark proposals disabled, never approximated\n";
    }

    if (options.admission_only) {
        auto checkpoint = strata::Dsv4CheckpointReader::open(options.model);
        if (!checkpoint.ok()) {
            for (const auto& error : checkpoint.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        std::vector<std::uint64_t> budgets;
        std::vector<std::string> errors;
        if (!device_budgets(options, budgets, errors)) {
            for (const auto& error : errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        strata::Dsv4AdmissionConfig config;
        config.host_memory_ceiling_bytes = options.host_memory_bytes;
        config.host_kv_cache_bytes = options.host_kv_cache_bytes;
        config.device_kv_cache_bytes = options.device_kv_cache_bytes;
        config.vram_weight_budgets = budgets;
        config.maximum_context_tokens = options.maximum_context_tokens;
        config.compact_kv_cache = options.block_kv_cache;
        config.require_zero_nvme_decode = true;
        const auto admission = strata::plan_dsv4_resident_topology(
            checkpoint.value->manifest(), config);
        if (!admission.ok()) {
            for (const auto& error : admission.errors) std::cerr << "error: " << error << '\n';
            return 1;
        }
        if (options.json) {
            std::cout << "{\"status\":\"admitted\",\"memory_plan\":";
            print_plan(std::cout, admission.plan);
            std::cout << "}\n";
        } else {
            std::cout << "admitted: required RAM " << admission.plan.required_host_bytes
                      << " bytes; VRAM budget " << admission.plan.total_vram_budget_bytes
                      << " bytes; steady-state NVMe 0 bytes\n";
        }
        return 0;
    }

    strata::Dsv4RuntimeConfig config;
    config.devices = options.devices;
    config.vram_cache_fraction = options.vram_fraction;
    config.host_memory_limit_bytes = options.host_memory_bytes;
    config.host_kv_cache_bytes = options.host_kv_cache_bytes;
    config.device_kv_cache_bytes = options.device_kv_cache_bytes;
    config.maximum_context_tokens = options.maximum_context_tokens;
    config.prefill_page_tokens = options.prefill_page_tokens;
    config.logit_trace_top_k = options.logit_trace_top_k;
    config.host_attention_threads = options.host_attention_threads;
    config.resident_read_workers = options.resident_read_workers;
    config.spine_warmup_workers = options.spine_warmup_workers;
    config.require_zero_nvme_decode = true;
    config.enable_dspark = false;
    config.enable_device_moe = options.device_moe;
    config.enable_flash_attention = options.flash_attention;
    config.kv_cache_mode = options.block_kv_cache
        ? strata::Dsv4KvCacheMode::Block
        : strata::Dsv4KvCacheMode::ScalarOracle;
    config.flash_attention_minimum_rows =
        options.flash_attention_minimum_rows;
    config.enable_logit_trace = options.logit_trace;
    config.enable_layer_hash_trace = options.layer_hash_trace;
    config.overlap_resident_warmup = options.overlap_resident_warmup;
    config.detailed_timing = options.detailed_timing;
    config.verbose = !options.quiet;
    config.route_trace_path = options.route_trace;
    strata::DeepSeekV4Runtime runtime;
    const auto initialized = runtime.initialize(options.model, config);
    if (!initialized.ok()) {
        for (const auto& error : initialized.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto generated = runtime.generate(options.prompt, options.maximum_new_tokens);
    if (!generated.ok()) {
        for (const auto& error : generated.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto& metrics = generated.metrics;
    if (options.json) {
        std::cout << std::setprecision(10)
                  << "{\"answer\":\"" << strata::cli::json_escape(generated.text)
                  << "\",\"execution\":\"exact_base_autoregressive\""
                  << ",\"dspark\":\"disabled\""
                  << ",\"device_moe\":"
                  << (metrics.device_moe_enabled ? "true" : "false")
                  << ",\"host_attention_threads\":"
                  << metrics.host_attention_threads
                  << ",\"prefill_page_tokens\":"
                  << metrics.prefill_page_tokens
                  << ",\"flash_attention\":"
                  << (metrics.flash_attention_enabled ? "true" : "false")
                  << ",\"block_kv_cache\":"
                  << (metrics.block_kv_cache_enabled ? "true" : "false")
                  << ",\"kv_block_rows\":" << metrics.kv_block_rows
                  << ",\"flash_attention_minimum_rows\":"
                  << metrics.flash_attention_minimum_rows
                  << ",\"resident_read_workers\":"
                  << metrics.resident_read_workers
                  << ",\"spine_warmup_workers\":"
                  << metrics.spine_warmup_workers
                  << ",\"resident_warmup_overlapped\":"
                  << (metrics.resident_warmup_overlapped ? "true" : "false")
                  << ",\"detailed_timing\":"
                  << (metrics.detailed_timing ? "true" : "false")
                  << ",\"initialization_seconds\":" << metrics.initialization_seconds
                  << ",\"resident_staging_seconds\":" << metrics.resident_staging_seconds
                  << ",\"resident_warmup_seconds\":" << metrics.resident_warmup_seconds
                  << ",\"prompt_tokens\":" << metrics.prompt_tokens
                  << ",\"generated_tokens\":" << generated.generated_token_ids.size()
                  << ",\"decode_steps\":" << metrics.decode_tokens
                  << ",\"rss_bytes\":" << metrics.rss_bytes
                  << ",\"device_vram_used_bytes\":";
        strata::cli::print_array(std::cout, metrics.device_vram_used_bytes);
        std::cout
                  << ",\"prefill_seconds\":" << metrics.prefill_seconds
                  << ",\"decode_seconds\":" << metrics.decode_seconds
                  << ",\"decode_checkpoint_read_bytes\":"
                  << metrics.decode_checkpoint_reads.bytes
                  << ",\"generation_checkpoint_read_bytes\":"
                  << metrics.generation_checkpoint_reads.bytes
                  << ",\"resident_stage_bytes\":" << metrics.resident_stage.bytes
                  << ",\"vram_cache_hits\":" << metrics.cache.hits
                  << ",\"vram_cache_misses\":" << metrics.cache.misses
                  << ",\"vram_cache_evictions\":" << metrics.cache.evictions
                  << ",\"kv_cache\":";
        print_kv_cache_stats(std::cout, metrics.kv_cache);
        std::cout
                  << ",\"cuda\":";
        print_cuda_stats(std::cout, metrics.cuda);
        std::cout << ",\"phases\":{\"prefill\":";
        print_phase(std::cout, metrics.prefill);
        std::cout << ",\"decode\":";
        print_phase(std::cout, metrics.decode);
        std::cout << '}'
                  << ",\"device_moe_runtime\":";
        print_device_moe_stats(std::cout, metrics.device_moe);
        std::cout
                  << ",\"memory_plan\":";
        print_plan(std::cout, metrics.memory);
        std::cout << ",\"prompt_token_ids\":";
        strata::cli::print_array(std::cout, generated.prompt_token_ids);
        std::cout << ",\"generated_token_ids\":";
        strata::cli::print_array(std::cout, generated.generated_token_ids);
        if (generated.diagnostics.logit_trace_enabled ||
            generated.diagnostics.layer_hash_trace_enabled) {
            std::cout << ",\"diagnostics\":";
            print_diagnostics(std::cout, generated.diagnostics);
        }
        std::cout << "}\n";
    } else {
        std::cout << generated.text << "\n\n"
                  << "initialization: " << metrics.initialization_seconds << " s\n"
                  << "resident staging: " << metrics.resident_staging_seconds << " s\n"
                  << "prefill: " << metrics.prompt_tokens << " tokens in "
                  << metrics.prefill_seconds << " s\n"
                  << "generation: " << generated.generated_token_ids.size()
                  << " tokens, " << metrics.decode_tokens << " decode steps in "
                  << metrics.decode_seconds << " s\n"
                  << "decode checkpoint reads: "
                  << metrics.decode_checkpoint_reads.bytes << " bytes\n";
    }
    return 0;
}
