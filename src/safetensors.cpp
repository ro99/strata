#include "strata/safetensors.hpp"

#include "json_cursor.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

namespace strata {

namespace {

using detail::JsonCursor;

[[nodiscard]] SafetensorsDtype parse_dtype(std::string_view value) noexcept {
    if (value == "BF16") return SafetensorsDtype::Bf16;
    if (value == "F16") return SafetensorsDtype::F16;
    if (value == "F32") return SafetensorsDtype::F32;
    if (value == "F8_E4M3" || value == "F8_E4M3FN") {
        return SafetensorsDtype::F8E4M3;
    }
    if (value == "F8_E8M0" || value == "F8_E8M0FNU") {
        return SafetensorsDtype::F8E8M0;
    }
    if (value == "I8") return SafetensorsDtype::I8;
    if (value == "U8") return SafetensorsDtype::U8;
    if (value == "I32") return SafetensorsDtype::I32;
    if (value == "I64") return SafetensorsDtype::I64;
    return SafetensorsDtype::Other;
}

[[nodiscard]] bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                    std::uint64_t& output) noexcept {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    output = left * right;
    return true;
}

[[nodiscard]] std::vector<std::uint64_t> parse_uint_array(JsonCursor& cursor) {
    std::vector<std::uint64_t> values;
    cursor.expect('[');
    if (cursor.consume(']')) return values;
    for (;;) {
        values.push_back(cursor.parse_uint64());
        if (cursor.consume(']')) return values;
        cursor.expect(',');
    }
}

void parse_index_metadata(JsonCursor& cursor, SafetensorsIndex& index) {
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        const auto key = cursor.parse_string();
        cursor.expect(':');
        if (key == "total_size") {
            index.total_size = cursor.parse_uint64();
        } else {
            cursor.skip_value();
        }
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

void parse_weight_map(JsonCursor& cursor, SafetensorsIndex& index) {
    std::unordered_set<std::string> names;
    cursor.expect('{');
    if (cursor.consume('}')) return;
    for (;;) {
        auto name = cursor.parse_string();
        cursor.expect(':');
        auto shard = cursor.parse_string();
        if (!names.insert(name).second) {
            throw detail::JsonError(cursor.offset(), "duplicate tensor name " + name);
        }
        index.entries.push_back({std::move(name), std::move(shard)});
        if (cursor.consume('}')) return;
        cursor.expect(',');
    }
}

[[nodiscard]] std::uint64_t read_le_u64(const unsigned char* bytes) noexcept {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8U; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] bool pread_exact(int file, void* output, std::size_t bytes,
                               std::uint64_t offset, std::string& error) {
    auto* cursor = static_cast<unsigned char*>(output);
    std::size_t completed = 0;
    while (completed < bytes) {
        const auto result = pread(file, cursor + completed, bytes - completed,
                                  static_cast<off_t>(offset + completed));
        if (result == 0) {
            error = "unexpected end of file";
            return false;
        }
        if (result < 0) {
            if (errno == EINTR) continue;
            error = std::strerror(errno);
            return false;
        }
        completed += static_cast<std::size_t>(result);
    }
    return true;
}

}  // namespace

std::uint32_t safetensors_dtype_bytes(SafetensorsDtype dtype) noexcept {
    switch (dtype) {
        case SafetensorsDtype::Bf16:
        case SafetensorsDtype::F16: return 2;
        case SafetensorsDtype::F32:
        case SafetensorsDtype::I32: return 4;
        case SafetensorsDtype::I64: return 8;
        case SafetensorsDtype::I8:
        case SafetensorsDtype::U8:
        case SafetensorsDtype::F8E4M3:
        case SafetensorsDtype::F8E8M0: return 1;
        case SafetensorsDtype::Other: return 0;
    }
    return 0;
}

std::string_view to_string(SafetensorsDtype dtype) noexcept {
    switch (dtype) {
        case SafetensorsDtype::Bf16: return "BF16";
        case SafetensorsDtype::F16: return "F16";
        case SafetensorsDtype::F32: return "F32";
        case SafetensorsDtype::F8E4M3: return "F8_E4M3";
        case SafetensorsDtype::F8E8M0: return "F8_E8M0";
        case SafetensorsDtype::I8: return "I8";
        case SafetensorsDtype::U8: return "U8";
        case SafetensorsDtype::I32: return "I32";
        case SafetensorsDtype::I64: return "I64";
        case SafetensorsDtype::Other: return "OTHER";
    }
    return "OTHER";
}

ParseResult<SafetensorsIndex> parse_safetensors_index(std::string_view json) {
    ParseResult<SafetensorsIndex> result;
    try {
        JsonCursor cursor(json);
        bool saw_metadata = false;
        bool saw_weight_map = false;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                const auto key = cursor.parse_string();
                cursor.expect(':');
                if (key == "metadata") {
                    if (saw_metadata) throw detail::JsonError(cursor.offset(), "duplicate metadata");
                    parse_index_metadata(cursor, result.value);
                    saw_metadata = true;
                } else if (key == "weight_map") {
                    if (saw_weight_map) throw detail::JsonError(cursor.offset(), "duplicate weight_map");
                    parse_weight_map(cursor, result.value);
                    saw_weight_map = true;
                } else {
                    cursor.skip_value();
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!cursor.finished()) throw detail::JsonError(cursor.offset(), "trailing content");
        if (!saw_metadata || result.value.total_size == 0U) {
            result.errors.emplace_back("Safetensors index is missing metadata.total_size");
        }
        if (!saw_weight_map || result.value.entries.empty()) {
            result.errors.emplace_back("Safetensors index has an empty weight_map");
        }
        std::unordered_set<std::string> shards;
        for (const auto& entry : result.value.entries) {
            if (entry.shard.empty()) {
                result.errors.emplace_back("tensor " + entry.name + " has an empty shard name");
            } else {
                shards.insert(entry.shard);
            }
        }
        result.value.shards.assign(shards.begin(), shards.end());
        std::sort(result.value.shards.begin(), result.value.shards.end());
    } catch (const std::exception& exception) {
        result.errors.emplace_back(exception.what());
    }
    return result;
}

ParseResult<SafetensorsShard> parse_safetensors_header(std::string_view json,
                                                       std::uint64_t data_start,
                                                       std::uint64_t file_size) {
    ParseResult<SafetensorsShard> result;
    result.value.header_size = data_start >= 8U ? data_start - 8U : 0U;
    result.value.data_start = data_start;
    result.value.file_size = file_size;
    try {
        if (data_start > file_size) throw std::runtime_error("header extends beyond shard file");
        JsonCursor cursor(json);
        std::unordered_set<std::string> names;
        cursor.expect('{');
        if (!cursor.consume('}')) {
            for (;;) {
                auto name = cursor.parse_string();
                cursor.expect(':');
                if (name == "__metadata__") {
                    cursor.skip_value();
                } else {
                    if (!names.insert(name).second) {
                        throw detail::JsonError(cursor.offset(), "duplicate tensor name " + name);
                    }
                    SafetensorsTensor tensor;
                    tensor.name = std::move(name);
                    bool saw_dtype = false;
                    bool saw_shape = false;
                    bool saw_offsets = false;
                    cursor.expect('{');
                    if (!cursor.consume('}')) {
                        for (;;) {
                            const auto key = cursor.parse_string();
                            cursor.expect(':');
                            if (key == "dtype") {
                                tensor.dtype = parse_dtype(cursor.parse_string());
                                saw_dtype = true;
                            } else if (key == "shape") {
                                tensor.shape = parse_uint_array(cursor);
                                saw_shape = true;
                            } else if (key == "data_offsets") {
                                const auto offsets = parse_uint_array(cursor);
                                if (offsets.size() != 2U) {
                                    throw detail::JsonError(cursor.offset(),
                                                            "data_offsets must have two elements");
                                }
                                tensor.relative_begin = offsets[0];
                                tensor.relative_end = offsets[1];
                                saw_offsets = true;
                            } else {
                                cursor.skip_value();
                            }
                            if (cursor.consume('}')) break;
                            cursor.expect(',');
                        }
                    }
                    if (!saw_dtype || !saw_shape || !saw_offsets) {
                        throw std::runtime_error("tensor " + tensor.name +
                                                 " is missing dtype, shape, or data_offsets");
                    }
                    if (tensor.dtype == SafetensorsDtype::Other) {
                        throw std::runtime_error("tensor " + tensor.name +
                                                 " has an unsupported dtype");
                    }
                    if (tensor.relative_end < tensor.relative_begin) {
                        throw std::runtime_error("tensor " + tensor.name +
                                                 " has reversed data offsets");
                    }
                    std::uint64_t elements = 1;
                    for (const auto dimension : tensor.shape) {
                        if (!checked_multiply(elements, dimension, elements)) {
                            throw std::runtime_error("tensor " + tensor.name + " shape overflows");
                        }
                    }
                    std::uint64_t expected_bytes = 0;
                    if (!checked_multiply(elements, safetensors_dtype_bytes(tensor.dtype),
                                          expected_bytes) ||
                        expected_bytes != tensor.bytes()) {
                        throw std::runtime_error("tensor " + tensor.name +
                                                 " byte count disagrees with dtype and shape");
                    }
                    if (tensor.relative_end > file_size - data_start) {
                        throw std::runtime_error("tensor " + tensor.name +
                                                 " extends beyond the shard file");
                    }
                    tensor.absolute_begin = data_start + tensor.relative_begin;
                    result.value.tensors.push_back(std::move(tensor));
                }
                if (cursor.consume('}')) break;
                cursor.expect(',');
            }
        }
        if (!cursor.finished()) throw detail::JsonError(cursor.offset(), "trailing content");
        if (result.value.tensors.empty()) throw std::runtime_error("Safetensors shard is empty");

        auto by_offset = result.value.tensors;
        std::sort(by_offset.begin(), by_offset.end(), [](const auto& left, const auto& right) {
            return left.relative_begin < right.relative_begin;
        });
        std::uint64_t expected_begin = 0;
        for (const auto& tensor : by_offset) {
            if (tensor.relative_begin != expected_begin) {
                throw std::runtime_error("Safetensors data offsets overlap or leave a gap before " +
                                         tensor.name);
            }
            expected_begin = tensor.relative_end;
        }
        if (expected_begin != file_size - data_start) {
            throw std::runtime_error("Safetensors tensor extents do not cover the shard payload");
        }
        std::sort(result.value.tensors.begin(), result.value.tensors.end(),
                  [](const auto& left, const auto& right) { return left.name < right.name; });
    } catch (const std::exception& exception) {
        result.errors.emplace_back(exception.what());
    }
    return result;
}

ParseResult<std::string> load_bounded_text_file(const std::string& path,
                                                std::uint64_t maximum_bytes) {
    ParseResult<std::string> result;
    const auto file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0) {
        result.errors.emplace_back(path + ": " + std::strerror(errno));
        return result;
    }
    struct stat status {};
    if (fstat(file, &status) != 0) {
        result.errors.emplace_back(path + ": " + std::strerror(errno));
        close(file);
        return result;
    }
    if (status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) > maximum_bytes) {
        result.errors.emplace_back(path + ": file exceeds bounded-read limit");
        close(file);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(status.st_size));
    std::string error;
    if (!result.value.empty() &&
        !pread_exact(file, result.value.data(), result.value.size(), 0U, error)) {
        result.errors.emplace_back(path + ": " + error);
    }
    close(file);
    return result;
}

ParseResult<SafetensorsShard> load_safetensors_shard(const std::string& path,
                                                     std::uint64_t maximum_header_bytes) {
    ParseResult<SafetensorsShard> result;
    const auto file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0) {
        result.errors.emplace_back(path + ": " + std::strerror(errno));
        return result;
    }
    struct stat status {};
    if (fstat(file, &status) != 0 || status.st_size < 8) {
        result.errors.emplace_back(path + ": invalid or truncated shard");
        close(file);
        return result;
    }
    unsigned char length_bytes[8]{};
    std::string error;
    if (!pread_exact(file, length_bytes, sizeof(length_bytes), 0U, error)) {
        result.errors.emplace_back(path + ": " + error);
        close(file);
        return result;
    }
    const auto header_size = read_le_u64(length_bytes);
    if (header_size == 0U || header_size > maximum_header_bytes ||
        header_size > static_cast<std::uint64_t>(status.st_size) - 8U) {
        result.errors.emplace_back(path + ": invalid Safetensors header length");
        close(file);
        return result;
    }
    std::string header(static_cast<std::size_t>(header_size), '\0');
    if (!pread_exact(file, header.data(), header.size(), 8U, error)) {
        result.errors.emplace_back(path + ": " + error);
        close(file);
        return result;
    }
    close(file);
    result = parse_safetensors_header(header, 8U + header_size,
                                      static_cast<std::uint64_t>(status.st_size));
    result.value.path = path;
    return result;
}

ParseResult<std::vector<std::byte>> read_safetensors_tensor(
    const std::string& path, const SafetensorsTensor& tensor,
    std::uint64_t maximum_bytes) {
    ParseResult<std::vector<std::byte>> result;
    if (tensor.bytes() > maximum_bytes ||
        tensor.bytes() > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        result.errors.emplace_back("tensor " + tensor.name + " exceeds bounded-read limit");
        return result;
    }
    const auto file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (file < 0) {
        result.errors.emplace_back(path + ": " + std::strerror(errno));
        return result;
    }
    struct stat status {};
    if (fstat(file, &status) != 0 || status.st_size < 0 ||
        tensor.absolute_begin > static_cast<std::uint64_t>(status.st_size) ||
        tensor.bytes() > static_cast<std::uint64_t>(status.st_size) - tensor.absolute_begin) {
        result.errors.emplace_back(path + ": tensor extent is outside the shard file");
        close(file);
        return result;
    }
    result.value.resize(static_cast<std::size_t>(tensor.bytes()));
    std::string error;
    if (!result.value.empty() &&
        !pread_exact(file, result.value.data(), result.value.size(), tensor.absolute_begin,
                     error)) {
        result.errors.emplace_back(path + ": " + error);
    }
    close(file);
    return result;
}

}  // namespace strata
