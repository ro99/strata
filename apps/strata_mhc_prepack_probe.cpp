#include "strata/deepseek_ops.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace {

constexpr std::size_t kState = 16'384U;
constexpr std::size_t kReduced = 4'096U;
constexpr std::size_t kRows = 24U;
constexpr std::size_t kCalls = 86U;

strata::ValidationResult prepacked(
    std::span<float> reduced, strata::Dsv4MhcMix& mix,
    std::span<const float> hidden, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base) {
    return strata::dsv4_mhc_prepacked_f32(
        reduced, mix, hidden, projection, scale, base);
}

strata::ValidationResult scalar(
    std::span<float> reduced, strata::Dsv4MhcMix& mix,
    std::span<const float> hidden, std::span<const float> projection,
    std::span<const float> scale, std::span<const float> base) {
    return strata::dsv4_mhc_pre_f32(
        reduced, mix, hidden, projection, scale, base);
}

bool equal(const strata::Dsv4MhcMix& left,
           const strata::Dsv4MhcMix& right) {
    const auto same = [](const std::vector<float>& a,
                         const std::vector<float>& b) {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    };
    return same(left.pre, right.pre) && same(left.post, right.post) &&
           same(left.combination, right.combination);
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

}  // namespace

int main() {
    if (!strata::dsv4_mhc_prepacked_supported()) {
        std::cerr << "AVX2 is unavailable\n";
        return 1;
    }
    const std::size_t stride = kRows * kState;
    std::vector<float> hidden(kCalls * kState);
    std::vector<float> row_major(kCalls * stride);
    std::vector<float> packed(row_major.size());
    for (std::size_t index = 0U; index < hidden.size(); ++index) {
        hidden[index] = static_cast<float>(static_cast<int>(index % 31U) - 15) /
                        64.0F;
    }
    for (std::size_t index = 0U; index < row_major.size(); ++index) {
        row_major[index] =
            static_cast<float>(static_cast<int>(index % 29U) - 14) / 4096.0F;
    }
    for (std::size_t call = 0U; call < kCalls; ++call) {
        if (!strata::dsv4_pack_mhc_projection_f32(
                 std::span<float>(packed).subspan(call * stride, stride),
                 std::span<const float>(row_major).subspan(call * stride, stride),
                 kRows, kState).ok()) return 1;
    }
    const std::array<float, 3> scale{0.75F, 0.5F, 0.25F};
    std::array<float, kRows> base{};

    std::vector<float> oracle(kReduced);
    std::vector<float> candidate(kReduced);
    strata::Dsv4MhcMix oracle_mix;
    strata::Dsv4MhcMix candidate_mix;
    if (!scalar(oracle, oracle_mix,
                std::span<const float>(hidden).first(kState),
                std::span<const float>(row_major).first(stride), scale, base).ok() ||
        !prepacked(candidate, candidate_mix,
                   std::span<const float>(hidden).first(kState),
                   std::span<const float>(packed).first(stride), scale, base).ok() ||
        std::memcmp(oracle.data(), candidate.data(),
                    oracle.size() * sizeof(float)) != 0 ||
        !equal(oracle_mix, candidate_mix)) {
        std::cerr << "prepacked AVX2 differs from scalar oracle\n";
        return 1;
    }

    std::array<std::vector<double>, 2> samples;
    double checksum = 0.0;
    for (std::size_t repetition = 0U; repetition < 7U; ++repetition) {
        for (std::size_t offset = 0U; offset < 2U; ++offset) {
            const std::size_t arm = (repetition + offset) % 2U;
            const auto& projections = arm == 0U ? row_major : packed;
            const auto kernel = arm == 0U ? scalar : prepacked;
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t call = 0U; call < kCalls; ++call) {
                const auto status = kernel(
                    candidate, candidate_mix,
                    std::span<const float>(hidden).subspan(call * kState, kState),
                    std::span<const float>(projections).subspan(
                        call * stride, stride), scale, base);
                if (!status.ok()) return 1;
                checksum += candidate[(call + repetition) % candidate.size()];
            }
            const double nanoseconds = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
            samples[arm].push_back(nanoseconds / static_cast<double>(kCalls));
        }
    }

    const double baseline = median(samples[0]);
    const double packed_ns = median(samples[1]);
    std::cout << "scalar_ns_per_call " << baseline << '\n'
              << "prepacked_ns_per_call " << packed_ns << '\n'
              << "speedup " << baseline / packed_ns << '\n'
              << "projected_ms_per_step "
              << (baseline - packed_ns) * static_cast<double>(kCalls) / 1.0e6
              << '\n'
              << "checksum " << checksum << '\n';
    return 0;
}
