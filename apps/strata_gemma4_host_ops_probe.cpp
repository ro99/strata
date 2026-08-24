#include "strata/gemma4_ops.hpp"
#include "strata/worker_pool.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

double median_ms(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() {
    constexpr std::size_t rows = 23U;
    constexpr std::size_t columns = 21'504U;
    constexpr std::size_t elements = rows * columns;
    constexpr std::size_t repetitions = 7U;
    constexpr std::array<std::size_t, 4U> widths{7U, 14U, 28U, 56U};

    std::vector<float> gate(elements);
    std::vector<float> up(elements);
    for (std::size_t index = 0U; index < elements; ++index) {
        gate[index] = std::sin(static_cast<float>(index % 4093U) * 0.001F);
        up[index] = std::cos(static_cast<float>(index % 4057U) * 0.001F);
    }
    std::vector<float> oracle(elements);
    std::vector<double> scalar_samples;
    for (std::size_t repetition = 0U; repetition <= repetitions; ++repetition) {
        const auto started = std::chrono::steady_clock::now();
        const auto status = strata::gemma4_geglu_bf16(oracle, gate, up);
        if (!status.ok()) return 1;
        const auto milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (repetition != 0U) scalar_samples.push_back(milliseconds);
    }
    const auto scalar = median_ms(scalar_samples);
    std::cout << "elements=" << elements << " repetitions=" << repetitions
              << " scalar_median_ms=" << scalar << '\n';

    strata::HostWorkerPool pool(widths.back());
    for (const auto width : widths) {
        std::vector<float> output(elements);
        std::vector<double> samples;
        for (std::size_t repetition = 0U; repetition <= repetitions; ++repetition) {
            const auto started = std::chrono::steady_clock::now();
            const auto status = pool.parallel_for(width, [&](std::size_t task) {
                const auto begin = elements * task / width;
                const auto end = elements * (task + 1U) / width;
                auto destination = std::span<float>(output).subspan(begin, end - begin);
                const auto gate_slice = std::span<const float>(gate).subspan(
                    begin, end - begin);
                const auto up_slice = std::span<const float>(up).subspan(
                    begin, end - begin);
                const auto task_status = strata::gemma4_geglu_bf16(
                    destination, gate_slice, up_slice);
                if (!task_status.ok()) throw std::runtime_error("GeGLU task failed");
            });
            if (!status.ok()) return 1;
            const auto milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (repetition != 0U) samples.push_back(milliseconds);
        }
        if (output != oracle) {
            std::cerr << "width " << width << " diverged from the scalar oracle\n";
            return 1;
        }
        const auto parallel = median_ms(samples);
        std::cout << "width=" << width << " median_ms=" << parallel
                  << " speedup=" << scalar / parallel
                  << " projected_60_layer_ms=" << parallel * 60.0 << '\n';
    }
    return 0;
}
