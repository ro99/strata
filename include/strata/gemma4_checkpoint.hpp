#pragma once

#include "strata/checkpoint_io.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/safetensors.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace strata {

struct Gemma4Tensor {
    std::string name;
    std::string shard;
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> shape;
    std::uint64_t offset{};
    std::uint64_t bytes{};
};

class Gemma4CheckpointReader;

struct Gemma4CheckpointOpenResult {
    std::unique_ptr<Gemma4CheckpointReader> value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty() && value != nullptr;
    }
};

class Gemma4CheckpointReader {
public:
    [[nodiscard]] static Gemma4CheckpointOpenResult open(
        std::string model_directory);

    [[nodiscard]] const Gemma4Tensor* find(std::string_view name) const noexcept;
    [[nodiscard]] ParseResult<std::vector<std::byte>> read(
        std::string_view name, std::uint64_t maximum_bytes) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32(
        std::string_view name, std::uint64_t maximum_elements) const;
    [[nodiscard]] ParseResult<std::vector<float>> read_f32_row(
        std::string_view name, std::uint64_t row) const;
    [[nodiscard]] std::span<const Gemma4Tensor> tensors() const noexcept {
        return tensors_;
    }

private:
    Gemma4CheckpointReader() = default;
    std::string model_directory_;
    std::vector<Gemma4Tensor> tensors_;
    std::unordered_map<std::string_view, std::size_t> by_name_;
    CheckpointShardSet shards_;
};

[[nodiscard]] ParseResult<CudaWeightDescriptor> describe_gemma4_cuda_linear(
    const Gemma4Tensor& packed, const Gemma4Tensor& scales,
    std::uint64_t rows, std::uint64_t columns);

[[nodiscard]] ValidationResult load_gemma4_cuda_linear(
    const Gemma4CheckpointReader& checkpoint, std::string_view base_name,
    std::uint64_t rows, std::uint64_t columns, int device,
    CudaBackend& backend, CudaWeight& output);

}  // namespace strata
