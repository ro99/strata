// Times DSV4 physical-page attention at the candidate widths a decode step
// actually produces, so the attention term in
//     tau = max_r W_r/B_r + Sigma_serial
// is measured rather than assumed at long context.
//
// Only one attention term grows with context. Per the runtime's candidate
// assembly (see attention_physical in deepseek_runtime.cpp):
//
//   ratio 0   layers   0 compressed + 128 sliding                      = 128
//   ratio 4   layers   512 (index top-k) + 128 sliding                 = 640
//   ratio 128 layers   roundup(context/128, 128) + 128 sliding         = grows
//
// The ratio-4 and ratio-0 widths are fixed by the contract and are already at
// their 1,048,576-token value in any short-context arm. The 20 ratio-128
// layers are the whole context-dependent attention term, which is what this
// probe measures.
//
// Usage: strata-dsv4-attention-probe [device] [repetitions]

#include "strata/cuda_backend.hpp"
#include "strata/deepseek_kv_cache.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

// dsv4_paged_attention() attends 32 heads of 512; the fused mHC entry point
// attends 32 or 64 depending on rank_local. The candidate loop, which is the
// only context-dependent part, is shared.
constexpr std::uint32_t kProbeHeads = 32U;
constexpr std::uint32_t kHeadDim = strata::kDeepSeekV4ExecutionContract.head_dim;
constexpr std::uint32_t kWindow =
    strata::kDeepSeekV4ExecutionContract.sliding_window;
constexpr std::uint32_t kCandidateBlock = 128U;
constexpr std::uint32_t kCompressedBlockRows = strata::kDsv4PhysicalKvBlockRows / 128U;

// The contract's ratio-128 layers, counted rather than assumed.
constexpr std::uint32_t dense_compressed_layers() {
    std::uint32_t count = 0U;
    for (std::uint32_t layer = 0U;
         layer < strata::kDeepSeekV4ExecutionContract.layer_count; ++layer) {
        if (strata::kDeepSeekV4ExecutionContract.compression_ratios[layer] ==
            128U) {
            ++count;
        }
    }
    return count;
}

constexpr std::uint32_t kLayers = dense_compressed_layers();

std::uint32_t parse(std::string_view text, std::uint32_t fallback) {
    std::uint32_t value = 0U;
    const auto* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end ? value : fallback;
}

// The backend rejects any query that is not exactly BF16-representable, so
// clearing the low 16 bits is the cheapest way to stay inside the contract.
float bf16_exact(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value) & 0xFFFF0000U;
    const auto rounded = std::bit_cast<float>(bits);
    return std::isfinite(rounded) ? rounded : 0.0F;
}

double median(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2U;
    return samples.size() % 2U == 1U
        ? samples[middle]
        : (samples[middle - 1U] + samples[middle]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) {
        std::cerr << "no CUDA device available\n";
        return 1;
    }
    const int device = argc > 1
        ? static_cast<int>(parse(argv[1], static_cast<std::uint32_t>(
                                              devices.front())))
        : devices.front();
    const auto repetitions = argc > 2 ? parse(argv[2], 5U) : 5U;

    strata::CudaBackend backend;
    const std::array<int, 1> selected{device};
    if (auto result = backend.initialize(selected, true); !result.ok()) {
        for (const auto& error : result.errors) std::cerr << error << '\n';
        return 1;
    }

    const auto compressed_layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Hca, kCompressedBlockRows);
    const auto sliding_layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Sliding, strata::kDsv4PhysicalKvBlockRows);
    if (!compressed_layout.ok() || !sliding_layout.ok()) {
        std::cerr << "physical KV layout unavailable\n";
        return 1;
    }

    std::vector<float> queries(
        static_cast<std::size_t>(kProbeHeads) * kHeadDim);
    for (std::size_t index = 0U; index < queries.size(); ++index) {
        queries[index] = bf16_exact(
            static_cast<float>((index * 37U) % 23U) / 8.0F - 1.5F);
    }
    std::vector<float> sinks(kProbeHeads);
    for (std::uint32_t head = 0U; head < kProbeHeads; ++head) {
        sinks[head] = static_cast<float>(static_cast<int>(head % 7U) - 3) / 8.0F;
    }
    std::vector<float> output(queries.size());
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    // One sliding block covers the whole 128-row window.
    std::vector<std::byte> sliding_storage(sliding_layout.value.block_bytes);
    {
        std::vector<float> row(kHeadDim);
        for (std::uint32_t index = 0U; index < kWindow; ++index) {
            for (std::uint32_t column = 0U; column < kHeadDim; ++column) {
                row[column] = static_cast<float>(
                    static_cast<int>((index * 13U + column * 7U) % 19U) - 9) /
                    4.0F;
            }
            if (auto result = strata::dsv4_physical_encode_kv_row(
                    strata::Dsv4KvBlockKind::Sliding, row, index,
                    sliding_storage);
                !result.ok()) {
                std::cerr << "sliding encode failed at row " << index << '\n';
                return 1;
            }
        }
    }
    strata::CudaBuffer sliding_resident;
    if (auto result = backend.upload_buffer(device, sliding_storage,
                                            sliding_resident);
        !result.ok()) {
        for (const auto& error : result.errors) std::cerr << error << '\n';
        return 1;
    }

    std::cout << "DSV4 physical-page attention, device " << device << ", "
              << repetitions << " interleaved repetitions\n"
              << "heads " << kProbeHeads << "  head_dim " << kHeadDim
              << "  sliding window " << kWindow << "  ratio-128 layers "
              << kLayers << " of "
              << strata::kDeepSeekV4ExecutionContract.layer_count << "\n"
              << "compressed page rows " << kCompressedBlockRows
              << "  block bytes " << compressed_layout.value.block_bytes
              << "\n\n"
              << std::setw(12) << "context" << std::setw(12) << "comp rows"
              << std::setw(12) << "cands" << std::setw(10) << "pages"
              << std::setw(14) << "ms/layer" << std::setw(14) << "ms/token"
              << std::setw(14) << "kernel ms" << std::setw(10) << "status"
              << '\n';

    // The first five contexts walk every candidate width the kernel accepts
    // (256, 256, 384, 512, 640), so the term's scaling is fitted rather than
    // assumed from its endpoints. The rest are the widths production needs at
    // long context.
    for (const std::uint64_t context :
         {std::uint64_t{4'096}, std::uint64_t{16'384}, std::uint64_t{32'768},
          std::uint64_t{49'152}, std::uint64_t{65'536},
          std::uint64_t{131'072}, std::uint64_t{262'144},
          std::uint64_t{1'048'576}}) {
        // Exactly the runtime's ratio-128 candidate assembly.
        const auto compressed_rows = static_cast<std::uint32_t>(context / 128U);
        const auto compressed_width =
            ((std::max(1U, compressed_rows) + kCandidateBlock - 1U) /
             kCandidateBlock) * kCandidateBlock;
        const auto candidates = compressed_width + kWindow;
        const auto blocks = (compressed_rows + kCompressedBlockRows - 1U) /
                            kCompressedBlockRows;

        std::vector<strata::CudaBuffer> compressed_resident(blocks);
        {
            std::vector<std::byte> block_storage(
                compressed_layout.value.block_bytes);
            std::vector<float> row(kHeadDim);
            for (std::uint32_t block = 0U; block < blocks; ++block) {
                for (std::uint32_t local = 0U; local < kCompressedBlockRows;
                     ++local) {
                    const auto absolute = block * kCompressedBlockRows + local;
                    for (std::uint32_t column = 0U; column < kHeadDim;
                         ++column) {
                        row[column] = static_cast<float>(
                            static_cast<int>(
                                (absolute * 31U + column * 11U) % 17U) - 8) /
                            4.0F;
                    }
                    if (auto result = strata::dsv4_physical_encode_kv_row(
                            strata::Dsv4KvBlockKind::Hca, row, local,
                            block_storage);
                        !result.ok()) {
                        std::cerr << "compressed encode failed\n";
                        return 1;
                    }
                }
                if (auto result = backend.upload_buffer(
                        device, block_storage, compressed_resident[block]);
                    !result.ok()) {
                    for (const auto& error : result.errors) {
                        std::cerr << error << '\n';
                    }
                    return 1;
                }
            }
        }

        std::vector<strata::CudaDsv4PhysicalPage> pages;
        pages.reserve(static_cast<std::size_t>(blocks) + 1U);
        for (std::uint32_t block = 0U; block < blocks; ++block) {
            pages.push_back({&compressed_resident[block],
                             kCompressedBlockRows});
        }
        const auto sliding_page = static_cast<std::uint32_t>(pages.size());
        pages.push_back({&sliding_resident, strata::kDsv4PhysicalKvBlockRows});

        std::vector<strata::CudaDsv4AttentionCandidate> candidate_table(
            candidates);
        for (std::uint32_t item = 0U; item < compressed_rows; ++item) {
            candidate_table[item] = {item / kCompressedBlockRows,
                                     item % kCompressedBlockRows, true};
        }
        for (std::uint32_t item = 0U; item < kWindow; ++item) {
            candidate_table[static_cast<std::size_t>(compressed_width) + item] =
                {sliding_page, item, true};
        }

        strata::CudaDsv4PagedAttentionRequest request;
        request.queries = queries;
        request.head_sinks = sinks;
        request.pages = pages;
        request.candidates = candidate_table;
        request.scale = scale;

        auto attempt = backend.dsv4_paged_attention(device, request, output);
        if (!attempt.ok()) {
            std::cout << std::setw(12) << context << std::setw(12)
                      << compressed_rows << std::setw(12) << candidates
                      << std::setw(10) << pages.size() << std::setw(14) << "-"
                      << std::setw(14) << "-" << std::setw(14) << "-"
                      << std::setw(10) << "REJECTED" << '\n';
            for (const auto& error : attempt.errors) {
                std::cout << "             " << error << '\n';
            }
            continue;
        }

        const auto before = backend.stats();
        std::vector<double> samples;
        samples.reserve(repetitions);
        for (std::uint32_t repetition = 0U; repetition < repetitions;
             ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            if (auto result =
                    backend.dsv4_paged_attention(device, request, output);
                !result.ok()) {
                for (const auto& error : result.errors) {
                    std::cerr << error << '\n';
                }
                return 1;
            }
            samples.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count());
        }
        const auto after = backend.stats();
        const auto kernel_ms =
            static_cast<double>(after.dsv4_paged_attention_kernel_nanoseconds -
                                before.dsv4_paged_attention_kernel_nanoseconds) /
            1.0e6 / static_cast<double>(repetitions);
        const auto per_layer = median(samples);
        std::cout << std::setw(12) << context << std::setw(12)
                  << compressed_rows << std::setw(12) << candidates
                  << std::setw(10) << pages.size() << std::setw(14)
                  << std::fixed << std::setprecision(3) << per_layer
                  << std::setw(14) << per_layer * kLayers << std::setw(14)
                  << kernel_ms << std::setw(10) << "ok" << '\n';
    }
    return 0;
}
