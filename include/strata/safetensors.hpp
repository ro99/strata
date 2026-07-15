#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strata {

enum class SafetensorsDtype : std::uint8_t {
    Bf16,
    F16,
    F32,
    F8E4M3,
    F8E8M0,
    I8,
    U8,
    I32,
    I64,
    Other,
};

struct SafetensorsIndexEntry {
    std::string name;
    std::string shard;
};

struct SafetensorsIndex {
    std::uint64_t total_size{};
    std::vector<SafetensorsIndexEntry> entries;
    std::vector<std::string> shards;
};

struct SafetensorsTensor {
    std::string name;
    SafetensorsDtype dtype{SafetensorsDtype::Other};
    std::vector<std::uint64_t> shape;
    std::uint64_t relative_begin{};
    std::uint64_t relative_end{};
    std::uint64_t absolute_begin{};

    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return relative_end - relative_begin;
    }
};

struct SafetensorsShard {
    std::string path;
    std::uint64_t file_size{};
    std::uint64_t header_size{};
    std::uint64_t data_start{};
    std::vector<SafetensorsTensor> tensors;
};

template <typename T>
struct ParseResult {
    T value;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
};

[[nodiscard]] ParseResult<SafetensorsIndex> parse_safetensors_index(
    std::string_view json);
[[nodiscard]] ParseResult<SafetensorsShard> parse_safetensors_header(
    std::string_view json, std::uint64_t data_start, std::uint64_t file_size);
[[nodiscard]] ParseResult<SafetensorsShard> load_safetensors_shard(
    const std::string& path, std::uint64_t maximum_header_bytes = 64ULL << 20U);
[[nodiscard]] ParseResult<std::vector<std::byte>> read_safetensors_tensor(
    const std::string& path, const SafetensorsTensor& tensor,
    std::uint64_t maximum_bytes);
[[nodiscard]] ParseResult<std::string> load_bounded_text_file(
    const std::string& path, std::uint64_t maximum_bytes);

[[nodiscard]] std::uint32_t safetensors_dtype_bytes(SafetensorsDtype dtype) noexcept;
[[nodiscard]] std::string_view to_string(SafetensorsDtype dtype) noexcept;

}  // namespace strata
