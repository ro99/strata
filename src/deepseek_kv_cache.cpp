#include "strata/deepseek_kv_cache.hpp"

#include "strata/deepseek_ops.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace strata {

namespace {

constexpr std::uint32_t kBlockMagic = 0x344B'5653U;
constexpr std::uint16_t kBlockHeaderBytes = kDsv4KvBlockHeaderBytes;
constexpr std::uint64_t kBlockAlignment = 64U;

struct BlockHeader {
    std::uint32_t magic{};
    std::uint16_t version{};
    std::uint16_t header_bytes{};
    std::uint8_t format{};
    std::uint8_t kind{};
    std::uint16_t reserved0{};
    std::uint32_t row_width{};
    std::uint32_t capacity_rows{};
    std::uint32_t layer{};
    std::uint32_t compression_ratio{};
    std::uint32_t reserved1{};
    std::uint64_t payload_bytes{};
    std::uint64_t physical_bytes{};
    std::array<std::byte, 16> reserved2{};
};

static_assert(sizeof(BlockHeader) == kBlockHeaderBytes);

struct TableKey {
    Dsv4KvBlockKind kind{Dsv4KvBlockKind::Sliding};
    std::uint32_t layer{};

    friend bool operator==(const TableKey&, const TableKey&) = default;
};

struct TableKeyHash {
    std::size_t operator()(const TableKey& key) const noexcept {
        return (static_cast<std::size_t>(key.layer) << 3U) ^
               static_cast<std::size_t>(key.kind);
    }
};

[[nodiscard]] std::uint32_t required_ratio(Dsv4KvBlockKind kind) noexcept {
    switch (kind) {
        case Dsv4KvBlockKind::Sliding: return 1U;
        case Dsv4KvBlockKind::Csa: return 4U;
        case Dsv4KvBlockKind::Hca: return 128U;
        case Dsv4KvBlockKind::LearnedIndex: return 4U;
    }
    return 0U;
}

[[nodiscard]] std::uint32_t required_width(Dsv4KvBlockKind kind) noexcept {
    return kind == Dsv4KvBlockKind::LearnedIndex
        ? kDeepSeekV4ExecutionContract.index_head_dim
        : kDeepSeekV4ExecutionContract.head_dim;
}

[[nodiscard]] std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) != 0x7F80'0000U) {
        bits += 0x7FFFU + ((bits >> 16U) & 1U);
    }
    return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(std::uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

[[nodiscard]] bool scale_code(float maximum, float format_max,
                              int minimum_exponent,
                              std::uint8_t& encoded) noexcept {
    const float bounded = maximum == 0.0F
        ? std::ldexp(format_max, minimum_exponent) : maximum;
    const int exponent = static_cast<int>(
        std::ceil(std::log2(bounded / format_max)));
    if (exponent < -127 || exponent > 127) return false;
    encoded = static_cast<std::uint8_t>(exponent + 127);
    return encoded != 0xFFU;
}

[[nodiscard]] bool encode_fp8_exact(float value,
                                    std::uint8_t& encoded) noexcept {
    static const auto magnitudes = [] {
        std::array<float, 127> values{};
        for (std::size_t code = 0U; code < values.size(); ++code) {
            values[code] = dsv4_fp8_e4m3_f32(
                static_cast<std::uint8_t>(code));
        }
        return values;
    }();
    const float magnitude = std::abs(value);
    const auto found = std::lower_bound(
        magnitudes.begin(), magnitudes.end(), magnitude);
    if (found == magnitudes.end() || *found != magnitude) return false;
    encoded = static_cast<std::uint8_t>(found - magnitudes.begin());
    if (std::signbit(value)) encoded |= 0x80U;
    return true;
}

[[nodiscard]] bool encode_fp4_exact(float value,
                                    std::uint8_t& encoded) noexcept {
    constexpr std::array<float, 8> magnitudes{
        0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const auto found = std::lower_bound(
        magnitudes.begin(), magnitudes.end(), std::abs(value));
    if (found == magnitudes.end() || *found != std::abs(value)) return false;
    encoded = static_cast<std::uint8_t>(found - magnitudes.begin());
    if (std::signbit(value)) encoded |= 0x08U;
    return true;
}

[[nodiscard]] std::uint64_t align_block(std::uint64_t bytes) noexcept {
    if (bytes > std::numeric_limits<std::uint64_t>::max() -
                    (kBlockAlignment - 1U)) {
        return 0U;
    }
    return (bytes + kBlockAlignment - 1U) & ~(kBlockAlignment - 1U);
}

}  // namespace

Dsv4KvFormat dsv4_kv_format(Dsv4KvBlockKind kind,
                            bool f32_oracle,
                            bool physical_layout) noexcept {
    if (f32_oracle) return Dsv4KvFormat::F32;
    if (physical_layout) {
        return kind == Dsv4KvBlockKind::LearnedIndex
            ? Dsv4KvFormat::PhysicalFp8E4m3PerTensor
            : Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope;
    }
    return kind == Dsv4KvBlockKind::LearnedIndex
        ? Dsv4KvFormat::Fp4E2m1Group32
        : Dsv4KvFormat::Fp8E4m3Group64Bf16Rope;
}

std::uint32_t dsv4_kv_block_rows(Dsv4KvBlockKind kind,
                                 std::uint32_t compression_ratio,
                                 bool physical_layout) noexcept {
    if (!physical_layout) return kDsv4KvBlockRows;
    if (compression_ratio != required_ratio(kind)) return 0U;
    return kDsv4PhysicalKvBlockRows / compression_ratio;
}

std::uint64_t dsv4_kv_row_bytes(Dsv4KvBlockKind kind,
                                Dsv4KvFormat format) noexcept {
    const auto width = static_cast<std::uint64_t>(required_width(kind));
    if (format == Dsv4KvFormat::F32) return width * sizeof(float);
    if (format == Dsv4KvFormat::Fp8E4m3Group64Bf16Rope &&
        kind != Dsv4KvBlockKind::LearnedIndex) {
        constexpr std::uint64_t rope =
            kDeepSeekV4ExecutionContract.rope_head_dim;
        const auto nope = width - rope;
        return nope + nope / 64U + rope * sizeof(std::uint16_t);
    }
    if (format == Dsv4KvFormat::Fp4E2m1Group32 &&
        kind == Dsv4KvBlockKind::LearnedIndex) {
        return width / 2U + width / 32U;
    }
    if (format == Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope &&
        kind != Dsv4KvBlockKind::LearnedIndex) {
        return 584U;
    }
    if (format == Dsv4KvFormat::PhysicalFp8E4m3PerTensor &&
        kind == Dsv4KvBlockKind::LearnedIndex) {
        return 132U;
    }
    return 0U;
}

std::uint64_t dsv4_kv_block_bytes(Dsv4KvBlockKind kind,
                                  Dsv4KvFormat format,
                                  std::uint32_t capacity_rows) noexcept {
    const auto row_bytes = dsv4_kv_row_bytes(kind, format);
    if (row_bytes == 0U || capacity_rows == 0U ||
        row_bytes > (std::numeric_limits<std::uint64_t>::max() -
                     kBlockHeaderBytes) / capacity_rows) {
        return 0U;
    }
    return align_block(kBlockHeaderBytes +
                       row_bytes * static_cast<std::uint64_t>(capacity_rows));
}

ValidationResult dsv4_encode_kv_row(
    Dsv4KvBlockKind kind, Dsv4KvFormat format,
    std::span<const float> values, std::span<std::byte> output) {
    ValidationResult result;
    const auto expected = dsv4_kv_row_bytes(kind, format);
    if (expected == 0U || values.size() != required_width(kind) ||
        output.size() != expected ||
        std::any_of(values.begin(), values.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back(
            "DeepSeek KV encode format, shape, or values are invalid");
        return result;
    }
    if (format == Dsv4KvFormat::F32) {
        std::memcpy(output.data(), values.data(), output.size());
        return result;
    }
    if (format == Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope ||
        format == Dsv4KvFormat::PhysicalFp8E4m3PerTensor) {
        result.errors.emplace_back(
            "DeepSeek physical KV rows require a block-major page encoder");
        return result;
    }
    if (format == Dsv4KvFormat::Fp8E4m3Group64Bf16Rope) {
        constexpr std::size_t rope =
            kDeepSeekV4ExecutionContract.rope_head_dim;
        const auto nope = values.size() - rope;
        const auto groups = nope / 64U;
        for (std::size_t group = 0U; group < groups; ++group) {
            const auto input = values.subspan(group * 64U, 64U);
            float maximum = 0.0F;
            for (const float value : input) {
                maximum = std::max(maximum, std::abs(value));
            }
            std::uint8_t scale_encoded = 0U;
            if (!scale_code(maximum, 448.0F, 0, scale_encoded)) {
                result.errors.emplace_back(
                    "DeepSeek KV FP8 scale is outside E8M0 range");
                return result;
            }
            const float scale = dsv4_fp8_e8m0_scale_f32(scale_encoded);
            output[nope + group] = static_cast<std::byte>(scale_encoded);
            for (std::size_t column = 0U; column < input.size(); ++column) {
                std::uint8_t encoded = 0U;
                if (!encode_fp8_exact(input[column] / scale, encoded) ||
                    decode_bf16(encode_bf16(
                        dsv4_fp8_e4m3_f32(encoded) * scale)) != input[column]) {
                    result.errors.emplace_back(
                        "DeepSeek KV FP8 storage would change a cache value");
                    return result;
                }
                output[group * 64U + column] = static_cast<std::byte>(encoded);
            }
        }
        const auto bf16_offset = nope + groups;
        for (std::size_t column = 0U; column < rope; ++column) {
            const auto encoded = encode_bf16(values[nope + column]);
            if (decode_bf16(encoded) != values[nope + column]) {
                result.errors.emplace_back(
                    "DeepSeek KV BF16 storage would change a RoPE value");
                return result;
            }
            std::memcpy(output.data() + bf16_offset +
                            column * sizeof(encoded),
                        &encoded, sizeof(encoded));
        }
        return result;
    }

    const auto packed = values.size() / 2U;
    const auto groups = values.size() / 32U;
    for (std::size_t group = 0U; group < groups; ++group) {
        const auto input = values.subspan(group * 32U, 32U);
        float maximum = 0.0F;
        for (const float value : input) {
            maximum = std::max(maximum, std::abs(value));
        }
        std::uint8_t scale_encoded = 0U;
        if (!scale_code(maximum, 6.0F, -126, scale_encoded)) {
            result.errors.emplace_back(
                "DeepSeek index FP4 scale is outside E8M0 range");
            return result;
        }
        const float scale = dsv4_fp8_e8m0_scale_f32(scale_encoded);
        output[packed + group] = static_cast<std::byte>(scale_encoded);
        for (std::size_t column = 0U; column < input.size(); column += 2U) {
            std::uint8_t low = 0U;
            std::uint8_t high = 0U;
            if (!encode_fp4_exact(input[column] / scale, low) ||
                !encode_fp4_exact(input[column + 1U] / scale, high) ||
                decode_bf16(encode_bf16(dsv4_fp4_e2m1_f32(low) * scale)) !=
                    input[column] ||
                decode_bf16(encode_bf16(dsv4_fp4_e2m1_f32(high) * scale)) !=
                    input[column + 1U]) {
                result.errors.emplace_back(
                    "DeepSeek index FP4 storage would change a cache value");
                return result;
            }
            output[group * 16U + column / 2U] =
                static_cast<std::byte>(low | (high << 4U));
        }
    }
    return result;
}

ValidationResult dsv4_decode_kv_row(
    Dsv4KvBlockKind kind, Dsv4KvFormat format,
    std::span<const std::byte> encoded, std::span<float> output) {
    ValidationResult result;
    const auto expected = dsv4_kv_row_bytes(kind, format);
    if (expected == 0U || encoded.size() != expected ||
        output.size() != required_width(kind)) {
        result.errors.emplace_back(
            "DeepSeek KV decode format or shape is invalid");
        return result;
    }
    if (format == Dsv4KvFormat::F32) {
        std::memcpy(output.data(), encoded.data(), encoded.size());
    } else if (format == Dsv4KvFormat::Fp8E4m3Group64Bf16Rope) {
        constexpr std::size_t rope =
            kDeepSeekV4ExecutionContract.rope_head_dim;
        const auto nope = output.size() - rope;
        const auto groups = nope / 64U;
        for (std::size_t group = 0U; group < groups; ++group) {
            const auto scale_encoded = std::to_integer<std::uint8_t>(
                encoded[nope + group]);
            if (scale_encoded == 0xFFU) {
                result.errors.emplace_back("DeepSeek KV FP8 scale is corrupt");
                return result;
            }
            const float scale = dsv4_fp8_e8m0_scale_f32(scale_encoded);
            for (std::size_t column = 0U; column < 64U; ++column) {
                const auto value_encoded = std::to_integer<std::uint8_t>(
                    encoded[group * 64U + column]);
                if ((value_encoded & 0x7FU) == 0x7FU) {
                    result.errors.emplace_back("DeepSeek KV FP8 value is corrupt");
                    return result;
                }
                output[group * 64U + column] = decode_bf16(encode_bf16(
                    dsv4_fp8_e4m3_f32(value_encoded) * scale));
            }
        }
        const auto bf16_offset = nope + groups;
        for (std::size_t column = 0U; column < rope; ++column) {
            std::uint16_t value = 0U;
            std::memcpy(&value, encoded.data() + bf16_offset +
                                    column * sizeof(value),
                        sizeof(value));
            output[nope + column] = decode_bf16(value);
        }
    } else if (format == Dsv4KvFormat::Fp4E2m1Group32) {
        const auto packed = output.size() / 2U;
        const auto groups = output.size() / 32U;
        for (std::size_t group = 0U; group < groups; ++group) {
            const auto scale_encoded = std::to_integer<std::uint8_t>(
                encoded[packed + group]);
            if (scale_encoded == 0xFFU) {
                result.errors.emplace_back("DeepSeek index FP4 scale is corrupt");
                return result;
            }
            const float scale = dsv4_fp8_e8m0_scale_f32(scale_encoded);
            for (std::size_t column = 0U; column < 32U; column += 2U) {
                const auto values = std::to_integer<std::uint8_t>(
                    encoded[group * 16U + column / 2U]);
                output[group * 32U + column] = decode_bf16(encode_bf16(
                    dsv4_fp4_e2m1_f32(values) * scale));
                output[group * 32U + column + 1U] = decode_bf16(encode_bf16(
                    dsv4_fp4_e2m1_f32(values >> 4U) * scale));
            }
        }
    } else {
        result.errors.emplace_back(
            "DeepSeek physical KV rows require a block-major page decoder");
        return result;
    }
    if (std::any_of(output.begin(), output.end(),
                    [](float value) { return !std::isfinite(value); })) {
        result.errors.emplace_back("DeepSeek KV decoded values are corrupt");
    }
    return result;
}

struct Dsv4KvCacheState {
    struct Table {
        std::vector<std::uint64_t> blocks;
        std::uint64_t minimum_row{};
        std::uint64_t end_row{};
        std::uint32_t compression_ratio{};
    };

    struct Sequence {
        std::uint64_t tokens{};
        std::unordered_map<TableKey, Table, TableKeyHash> tables;
    };

    struct Block {
        Dsv4KvBlockInfo info;
        std::vector<std::byte> host;
        std::vector<CudaBuffer> devices;
        std::vector<std::uint64_t> device_last_use;
        std::vector<std::uint32_t> device_leases;
    };

    explicit Dsv4KvCacheState(Dsv4KvCacheConfig value, CudaBackend* backend)
        : config(std::move(value)), cuda(backend) {
        metrics.host_capacity_bytes = config.host_capacity_bytes;
        metrics.device_capacity_bytes = config.device_capacity_bytes;
        metrics.device_used_bytes.resize(config.devices.size());
        metrics.device_peak_bytes.resize(config.devices.size());
    }

    [[nodiscard]] ValidationResult validate() const {
        ValidationResult result;
        if (config.block_rows == 0U || config.sliding_window_rows == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV block rows and sliding window must be positive");
        }
        if (config.physical_layout && config.f32_oracle) {
            result.errors.emplace_back(
                "DeepSeek physical KV and the F32 oracle are mutually exclusive");
        }
        if (config.physical_layout &&
            config.block_rows != kDsv4PhysicalKvBlockRows) {
            result.errors.emplace_back(
                "DeepSeek physical KV requires 256-source-token blocks");
        }
        if (config.host_capacity_bytes == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV cache requires an explicit host capacity");
        }
        if (config.devices.size() != config.device_capacity_bytes.size()) {
            result.errors.emplace_back(
                "DeepSeek KV device ids and capacities must have equal lengths");
        }
        std::unordered_set<int> unique;
        for (const int device : config.devices) {
            if (device < 0 || !unique.insert(device).second) {
                result.errors.emplace_back(
                    "DeepSeek KV devices must be unique non-negative ids");
                break;
            }
        }
        return result;
    }

    [[nodiscard]] Sequence* sequence(Dsv4SequenceHandle handle) noexcept {
        const auto found = sequences.find(handle);
        return found == sequences.end() ? nullptr : &found->second;
    }

    [[nodiscard]] const Sequence* sequence(
        Dsv4SequenceHandle handle) const noexcept {
        const auto found = sequences.find(handle);
        return found == sequences.end() ? nullptr : &found->second;
    }

    [[nodiscard]] std::uint64_t total_leases(const Block& block) const noexcept {
        std::uint64_t total = 0U;
        for (const auto leases : block.device_leases) total += leases;
        return total;
    }

    [[nodiscard]] ValidationResult validate_block(const Block& block) const {
        ValidationResult result;
        if (block.host.size() < sizeof(BlockHeader)) {
            result.errors.emplace_back("DeepSeek KV block header is truncated");
            return result;
        }
        BlockHeader header;
        std::memcpy(&header, block.host.data(), sizeof(header));
        const auto row_bytes = dsv4_kv_row_bytes(block.info.kind,
                                                 block.info.format);
        const auto physical = dsv4_kv_block_bytes(
            block.info.kind, block.info.format, block.info.capacity_rows);
        if (header.magic != kBlockMagic ||
            header.version != kDsv4KvFormatVersion ||
            header.header_bytes != kBlockHeaderBytes ||
            header.format != static_cast<std::uint8_t>(block.info.format) ||
            header.kind != static_cast<std::uint8_t>(block.info.kind) ||
            header.row_width != block.info.row_width ||
            header.capacity_rows != block.info.capacity_rows ||
            header.layer != block.info.layer ||
            header.compression_ratio != block.info.compression_ratio ||
            header.payload_bytes != block.info.payload_bytes ||
            header.physical_bytes != block.info.physical_bytes ||
            block.info.format_version != kDsv4KvFormatVersion ||
            block.info.used_rows > block.info.capacity_rows ||
            row_bytes == 0U || physical == 0U ||
            block.info.payload_bytes !=
                row_bytes * block.info.capacity_rows ||
            block.info.physical_bytes != physical ||
            block.host.size() != physical ||
            std::any_of(header.reserved2.begin(), header.reserved2.end(),
                        [](std::byte value) { return value != std::byte{}; })) {
            result.errors.emplace_back(
                "DeepSeek KV block format version or shape is corrupt");
        }
        return result;
    }

    void erase_block(std::uint64_t id) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || found->second.info.refcount != 0U ||
            total_leases(found->second) != 0U) {
            return;
        }
        auto& block = found->second;
        const auto bytes = static_cast<std::uint64_t>(block.host.size());
        metrics.host_used_bytes -= bytes;
        for (std::size_t slot = 0U; slot < block.devices.size(); ++slot) {
            if (block.devices[slot].valid()) {
                metrics.device_used_bytes[slot] -=
                    block.devices[slot].device_bytes();
            }
        }
        blocks.erase(found);
    }

    void release_block(std::uint64_t id) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || found->second.info.refcount == 0U) return;
        --found->second.info.refcount;
        erase_block(id);
    }

    void release_tables(Sequence& target) noexcept {
        for (auto& [key, table] : target.tables) {
            static_cast<void>(key);
            for (const auto id : table.blocks) release_block(id);
        }
        target.tables.clear();
        target.tokens = 0U;
    }

    [[nodiscard]] ValidationResult allocate_block(
        Dsv4SequenceHandle owner, Dsv4KvBlockKind kind,
        std::uint32_t layer, std::uint32_t ratio, std::uint64_t start_row,
        const Block* source, std::uint64_t& output) {
        ValidationResult result;
        const auto width = required_width(kind);
        const auto format = dsv4_kv_format(
            kind, config.f32_oracle, config.physical_layout);
        const auto capacity_rows = config.physical_layout
            ? dsv4_kv_block_rows(kind, ratio, true) : config.block_rows;
        const auto row_bytes = dsv4_kv_row_bytes(kind, format);
        const auto payload_bytes = row_bytes * capacity_rows;
        const auto bytes = dsv4_kv_block_bytes(
            kind, format, capacity_rows);
        if (row_bytes == 0U || bytes == 0U ||
            capacity_rows == 0U ||
            payload_bytes / capacity_rows != row_bytes) {
            result.errors.emplace_back("DeepSeek KV block byte count overflows");
            return result;
        }
        if (bytes > config.host_capacity_bytes -
                        std::min(config.host_capacity_bytes,
                                 metrics.host_used_bytes)) {
            result.errors.emplace_back(
                "DeepSeek KV host cache capacity is exhausted");
            return result;
        }
        if (next_block_id == std::numeric_limits<std::uint64_t>::max() ||
            start_row > std::numeric_limits<std::uint64_t>::max() / ratio) {
            result.errors.emplace_back("DeepSeek KV block identity overflows");
            return result;
        }
        const auto started = std::chrono::steady_clock::now();
        try {
            Block block;
            block.info.id = next_block_id++;
            block.info.owner_sequence = owner;
            block.info.logical_begin = start_row * ratio;
            block.info.used_rows = source == nullptr ? 0U : source->info.used_rows;
            block.info.capacity_rows = capacity_rows;
            block.info.row_width = width;
            block.info.layer = layer;
            block.info.compression_ratio = ratio;
            block.info.refcount = 1U;
            block.info.kind = kind;
            block.info.format = format;
            block.info.format_version = kDsv4KvFormatVersion;
            block.info.payload_bytes = payload_bytes;
            block.info.physical_bytes = bytes;
            block.host.resize(static_cast<std::size_t>(bytes));
            BlockHeader header;
            header.magic = kBlockMagic;
            header.version = kDsv4KvFormatVersion;
            header.header_bytes = kBlockHeaderBytes;
            header.format = static_cast<std::uint8_t>(format);
            header.kind = static_cast<std::uint8_t>(kind);
            header.row_width = width;
            header.capacity_rows = capacity_rows;
            header.layer = layer;
            header.compression_ratio = ratio;
            header.payload_bytes = payload_bytes;
            header.physical_bytes = bytes;
            std::memcpy(block.host.data(), &header, sizeof(header));
            if (source != nullptr) block.host = source->host;
            block.devices.resize(config.devices.size());
            block.device_last_use.resize(config.devices.size());
            block.device_leases.resize(config.devices.size());
            output = block.info.id;
            blocks.emplace(output, std::move(block));
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back("cannot allocate DeepSeek KV block");
            return result;
        }
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        ++metrics.allocated_blocks;
        ++metrics.allocation_calls;
        metrics.allocation_nanoseconds += elapsed;
        metrics.host_used_bytes += bytes;
        metrics.host_peak_bytes = std::max(metrics.host_peak_bytes,
                                           metrics.host_used_bytes);
        return result;
    }

    [[nodiscard]] Block* find_block(Table& table,
                                    std::uint64_t logical_row) noexcept {
        const auto found_id = std::lower_bound(
            table.blocks.begin(), table.blocks.end(), logical_row,
            [this](std::uint64_t id, std::uint64_t row) {
                const auto& block = blocks.at(id);
                const auto begin = block.info.logical_begin /
                                   block.info.compression_ratio;
                return begin + block.info.capacity_rows <= row;
            });
        if (found_id == table.blocks.end()) return nullptr;
        auto& block = blocks.at(*found_id);
        const auto begin = block.info.logical_begin /
                           block.info.compression_ratio;
        return logical_row >= begin ? &block : nullptr;
    }

    [[nodiscard]] const Block* find_block(
        const Table& table, std::uint64_t logical_row) const noexcept {
        const auto found_id = std::lower_bound(
            table.blocks.begin(), table.blocks.end(), logical_row,
            [this](std::uint64_t id, std::uint64_t row) {
                const auto& block = blocks.at(id);
                const auto begin = block.info.logical_begin /
                                   block.info.compression_ratio;
                return begin + block.info.capacity_rows <= row;
            });
        if (found_id == table.blocks.end()) return nullptr;
        const auto& block = blocks.at(*found_id);
        const auto begin = block.info.logical_begin /
                           block.info.compression_ratio;
        return logical_row >= begin ? &block : nullptr;
    }

    void release_device_lease(std::uint64_t id, std::size_t slot) noexcept {
        const auto found = blocks.find(id);
        if (found == blocks.end() || slot >= found->second.device_leases.size() ||
            found->second.device_leases[slot] == 0U) {
            return;
        }
        --found->second.device_leases[slot];
        --found->second.info.in_flight;
        erase_block(id);
    }

    Dsv4KvCacheConfig config;
    CudaBackend* cuda{};
    std::unordered_map<Dsv4SequenceHandle, Sequence> sequences;
    std::unordered_map<std::uint64_t, Block> blocks;
    Dsv4KvCacheStats metrics;
    Dsv4SequenceHandle next_sequence_id{1U};
    std::uint64_t next_block_id{1U};
    std::uint64_t clock{};
};

Dsv4KvDeviceLease::Dsv4KvDeviceLease() = default;

Dsv4KvDeviceLease::Dsv4KvDeviceLease(
    std::shared_ptr<Dsv4KvCacheState> state, std::uint64_t block_id,
    std::size_t device_slot)
    : state_(std::move(state)), block_id_(block_id), device_slot_(device_slot) {}

Dsv4KvDeviceLease::~Dsv4KvDeviceLease() { release(); }

Dsv4KvDeviceLease::Dsv4KvDeviceLease(Dsv4KvDeviceLease&& other) noexcept
    : state_(std::move(other.state_)), block_id_(other.block_id_),
      device_slot_(other.device_slot_) {
    other.block_id_ = 0U;
}

Dsv4KvDeviceLease& Dsv4KvDeviceLease::operator=(
    Dsv4KvDeviceLease&& other) noexcept {
    if (this == &other) return *this;
    release();
    state_ = std::move(other.state_);
    block_id_ = other.block_id_;
    device_slot_ = other.device_slot_;
    other.block_id_ = 0U;
    return *this;
}

void Dsv4KvDeviceLease::release() noexcept {
    if (state_ != nullptr && block_id_ != 0U) {
        state_->release_device_lease(block_id_, device_slot_);
    }
    state_.reset();
    block_id_ = 0U;
}

bool Dsv4KvDeviceLease::valid() const noexcept {
    return buffer() != nullptr;
}

const CudaBuffer* Dsv4KvDeviceLease::buffer() const noexcept {
    if (state_ == nullptr || block_id_ == 0U) return nullptr;
    const auto found = state_->blocks.find(block_id_);
    if (found == state_->blocks.end() ||
        device_slot_ >= found->second.devices.size() ||
        !found->second.devices[device_slot_].valid()) {
        return nullptr;
    }
    return &found->second.devices[device_slot_];
}

Dsv4KvPhysicalAppend::Dsv4KvPhysicalAppend() = default;
Dsv4KvPhysicalAppend::~Dsv4KvPhysicalAppend() = default;
Dsv4KvPhysicalAppend::Dsv4KvPhysicalAppend(
    Dsv4KvPhysicalAppend&&) noexcept = default;
Dsv4KvPhysicalAppend& Dsv4KvPhysicalAppend::operator=(
    Dsv4KvPhysicalAppend&&) noexcept = default;

Dsv4KvPhysicalAppend::Dsv4KvPhysicalAppend(
    std::shared_ptr<Dsv4KvCacheState> state, Dsv4KvDeviceLease lease,
    std::uint64_t block_id, std::uint32_t physical_row,
    std::uint64_t data_offset, std::uint64_t scale_offset,
    std::uint32_t data_bytes, std::uint32_t scale_bytes,
    std::byte* payload, std::uint64_t payload_bytes,
    std::uint32_t row_width, Dsv4KvBlockKind kind,
    Dsv4KvFormat format)
    : state_(std::move(state)), lease_(std::move(lease)),
      block_id_(block_id), physical_row_(physical_row),
      data_offset_(data_offset), scale_offset_(scale_offset),
      data_bytes_(data_bytes), scale_bytes_(scale_bytes),
      payload_(payload), payload_bytes_(payload_bytes),
      row_width_(row_width), kind_(kind), format_(format) {}

bool Dsv4KvPhysicalAppend::valid() const noexcept {
    return state_ != nullptr && block_id_ != 0U && lease_.valid() &&
           data_bytes_ != 0U && scale_bytes_ != 0U && payload_ != nullptr &&
           payload_bytes_ != 0U && row_width_ != 0U;
}

const CudaBuffer* Dsv4KvPhysicalAppend::buffer() const noexcept {
    return lease_.buffer();
}

std::uint64_t Dsv4KvPhysicalAppend::data_offset() const noexcept {
    return data_offset_;
}

std::uint64_t Dsv4KvPhysicalAppend::scale_offset() const noexcept {
    return scale_offset_;
}

std::uint32_t Dsv4KvPhysicalAppend::data_bytes() const noexcept {
    return data_bytes_;
}

std::uint32_t Dsv4KvPhysicalAppend::scale_bytes() const noexcept {
    return scale_bytes_;
}

std::uint64_t Dsv4KvPhysicalAppend::patch_bytes() const noexcept {
    return static_cast<std::uint64_t>(data_bytes_) + scale_bytes_;
}

ValidationResult Dsv4KvPhysicalAppend::commit(
    std::span<const float> values, std::span<std::byte> patch) {
    ValidationResult result;
    if (!valid() || committed_ || patch.size() != patch_bytes()) {
        result.errors.emplace_back(
            "DeepSeek physical KV append commit is invalid (valid=" +
            std::to_string(valid()) + ", committed=" +
            std::to_string(committed_) + ", patch=" +
            std::to_string(patch.size()) + ", expected=" +
            std::to_string(patch_bytes()) + ")");
        return result;
    }
    if (values.size() != row_width_ ||
        std::any_of(values.begin(), values.end(), [](float value) {
            return !std::isfinite(value) ||
                   !std::isfinite(decode_bf16(encode_bf16(value)));
        })) {
        result.errors.emplace_back(
            "DeepSeek physical KV append values are invalid");
        return result;
    }
    auto payload = std::span<std::byte>(
        payload_, static_cast<std::size_t>(payload_bytes_));
    result = dsv4_physical_encode_kv_row(
        kind_, values, physical_row_, payload);
    if (!result.ok()) return result;
    std::copy_n(
        payload.begin() + static_cast<std::ptrdiff_t>(data_offset_),
        data_bytes_, patch.begin());
    std::copy_n(
        payload.begin() + static_cast<std::ptrdiff_t>(scale_offset_),
        scale_bytes_,
        patch.begin() + static_cast<std::ptrdiff_t>(data_bytes_));
    committed_ = true;
    return result;
}

ValidationResult Dsv4KvPhysicalAppend::account() {
    ValidationResult result;
    if (!valid() || !committed_ || accounted_) {
        result.errors.emplace_back(
            "DeepSeek physical KV append accounting is invalid (valid=" +
            std::to_string(valid()) + ", committed=" +
            std::to_string(committed_) + ", accounted=" +
            std::to_string(accounted_) + ")");
        return result;
    }
    state_->metrics.host_write_bytes += dsv4_kv_row_bytes(kind_, format_);
    state_->metrics.host_to_device_bytes += patch_bytes();
    accounted_ = true;
    return result;
}

Dsv4KvCache::Dsv4KvCache(Dsv4KvCacheConfig config, CudaBackend* cuda)
    : state_(std::make_shared<Dsv4KvCacheState>(std::move(config), cuda)) {}

ValidationResult Dsv4KvCache::validate() const { return state_->validate(); }

ParseResult<Dsv4SequenceHandle> Dsv4KvCache::create_sequence() {
    ParseResult<Dsv4SequenceHandle> result;
    result.errors = state_->validate().errors;
    if (!result.ok()) return result;
    if (state_->next_sequence_id == std::numeric_limits<std::uint64_t>::max()) {
        result.errors.emplace_back("DeepSeek KV sequence identity overflows");
        return result;
    }
    result.value = state_->next_sequence_id++;
    try {
        state_->sequences.emplace(result.value, Dsv4KvCacheState::Sequence{});
    } catch (const std::bad_alloc&) {
        result.value = 0U;
        result.errors.emplace_back("cannot allocate DeepSeek KV sequence metadata");
        return result;
    }
    ++state_->metrics.sequence_creations;
    return result;
}

ParseResult<Dsv4SequenceHandle> Dsv4KvCache::fork_sequence(
    Dsv4SequenceHandle source) {
    ParseResult<Dsv4SequenceHandle> result;
    const auto* original = state_->sequence(source);
    if (original == nullptr) {
        result.errors.emplace_back("DeepSeek KV source sequence does not exist");
        return result;
    }
    if (state_->next_sequence_id == std::numeric_limits<std::uint64_t>::max()) {
        result.errors.emplace_back("DeepSeek KV sequence identity overflows");
        return result;
    }
    for (const auto& [key, table] : original->tables) {
        static_cast<void>(key);
        for (const auto id : table.blocks) {
            const auto found = state_->blocks.find(id);
            if (found == state_->blocks.end() ||
                found->second.info.refcount ==
                    std::numeric_limits<std::uint32_t>::max()) {
                result.errors.emplace_back(
                    "DeepSeek KV block reference count overflows");
                return result;
            }
        }
    }
    Dsv4KvCacheState::Sequence copy;
    try {
        copy = *original;
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot allocate DeepSeek KV fork metadata");
        return result;
    }
    result.value = state_->next_sequence_id++;
    try {
        state_->sequences.emplace(result.value, std::move(copy));
    } catch (const std::bad_alloc&) {
        result.value = 0U;
        result.errors.emplace_back("cannot allocate DeepSeek KV fork metadata");
        return result;
    }
    auto* fork = state_->sequence(result.value);
    for (const auto& [key, table] : fork->tables) {
        static_cast<void>(key);
        for (const auto id : table.blocks) ++state_->blocks.at(id).info.refcount;
    }
    ++state_->metrics.sequence_creations;
    return result;
}

ValidationResult Dsv4KvCache::reset_sequence(Dsv4SequenceHandle sequence) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    state_->release_tables(*target);
    ++state_->metrics.sequence_resets;
    return result;
}

ValidationResult Dsv4KvCache::truncate_sequence(
    Dsv4SequenceHandle sequence, std::uint64_t tokens) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    if (tokens > target->tokens) {
        result.errors.emplace_back(
            "DeepSeek KV truncation exceeds the sequence length");
        return result;
    }
    for (const auto& [key, table] : target->tables) {
        if (key.kind == Dsv4KvBlockKind::Sliding &&
            tokens < table.minimum_row) {
            result.errors.emplace_back(
                "DeepSeek KV truncation precedes retained sliding state");
            return result;
        }
    }
    for (auto& [key, table] : target->tables) {
        const auto keep_rows = key.kind == Dsv4KvBlockKind::Sliding
            ? tokens : tokens / table.compression_ratio;
        table.end_row = std::min(table.end_row, keep_rows);
        while (!table.blocks.empty()) {
            const auto id = table.blocks.back();
            const auto& block = state_->blocks.at(id);
            const auto begin = block.info.logical_begin /
                               block.info.compression_ratio;
            if (begin < table.end_row) break;
            table.blocks.pop_back();
            state_->release_block(id);
        }
        table.minimum_row = std::min(table.minimum_row, table.end_row);
    }
    target->tokens = tokens;
    ++state_->metrics.sequence_truncations;
    return result;
}

ValidationResult Dsv4KvCache::release_sequence(Dsv4SequenceHandle sequence) {
    ValidationResult result;
    const auto found = state_->sequences.find(sequence);
    if (found == state_->sequences.end()) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    state_->release_tables(found->second);
    state_->sequences.erase(found);
    ++state_->metrics.sequence_releases;
    return result;
}

ValidationResult Dsv4KvCache::append(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint32_t compression_ratio,
    std::uint64_t logical_row, std::span<const float> values,
    std::uint64_t sliding_retention_floor) {
    ValidationResult result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    const auto wanted_ratio = required_ratio(kind);
    const auto width = required_width(kind);
    const auto format = dsv4_kv_format(
        kind, state_->config.f32_oracle,
        state_->config.physical_layout);
    const auto capacity_rows = state_->config.physical_layout
        ? dsv4_kv_block_rows(kind, compression_ratio, true)
        : state_->config.block_rows;
    if (layer >= kDeepSeekV4ExecutionContract.layer_count ||
        compression_ratio != wanted_ratio || values.size() != width ||
        capacity_rows == 0U ||
        std::any_of(values.begin(), values.end(), [](float value) {
            return !std::isfinite(value) ||
                   !std::isfinite(decode_bf16(encode_bf16(value)));
        })) {
        result.errors.emplace_back(
            "DeepSeek KV block kind, layer, ratio, row width, or values are invalid");
        return result;
    }
    std::array<std::byte,
               kDeepSeekV4ExecutionContract.head_dim * sizeof(float)>
        encoded_storage{};
    auto encoded = std::span<std::byte>(encoded_storage).first(
        static_cast<std::size_t>(dsv4_kv_row_bytes(
            kind, format)));
    if (!state_->config.physical_layout) {
        result = dsv4_encode_kv_row(kind, format, values, encoded);
        if (!result.ok()) return result;
    }
    if (logical_row == std::numeric_limits<std::uint64_t>::max() ||
        logical_row + 1U > std::numeric_limits<std::uint64_t>::max() /
                               compression_ratio) {
        result.errors.emplace_back("DeepSeek KV logical position overflows");
        return result;
    }
    const TableKey key{kind, layer};
    auto table_found = target->tables.find(key);
    if (table_found == target->tables.end()) {
        if (logical_row != 0U) {
            result.errors.emplace_back(
                "DeepSeek KV first row must start at logical row zero");
            return result;
        }
        try {
            table_found = target->tables.emplace(
                key,
                Dsv4KvCacheState::Table{{}, 0U, 0U, compression_ratio}).first;
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate DeepSeek KV block-table metadata");
            return result;
        }
    }
    auto& table = table_found->second;
    if (table.compression_ratio != compression_ratio ||
        logical_row != table.end_row) {
        result.errors.emplace_back(
            "DeepSeek KV append position is not contiguous");
        return result;
    }
    const auto block_begin =
        logical_row / capacity_rows * capacity_rows;
    auto* block = state_->find_block(table, logical_row);
    if (block != nullptr && block->info.refcount > 1U) {
        const auto old_id = block->info.id;
        std::uint64_t replacement = 0U;
        result = state_->allocate_block(sequence, kind, layer,
                                        compression_ratio, block_begin,
                                        block, replacement);
        if (!result.ok()) return result;
        const auto id = std::find(table.blocks.begin(), table.blocks.end(), old_id);
        *id = replacement;
        state_->release_block(old_id);
        block = &state_->blocks.at(replacement);
        ++state_->metrics.copy_on_write_blocks;
    } else if (block == nullptr) {
        std::uint64_t id = 0U;
        result = state_->allocate_block(sequence, kind, layer,
                                        compression_ratio, block_begin,
                                        nullptr, id);
        if (!result.ok()) return result;
        try {
            table.blocks.push_back(id);
        } catch (const std::bad_alloc&) {
            state_->release_block(id);
            result.errors.emplace_back(
                "cannot grow DeepSeek KV block-table metadata");
            return result;
        }
        block = &state_->blocks.at(id);
    }
    result = state_->validate_block(*block);
    if (!result.ok()) return result;
    if (state_->total_leases(*block) != 0U) {
        result.errors.emplace_back(
            "DeepSeek KV cannot mutate an in-flight device block");
        return result;
    }
    if (state_->config.physical_layout) {
        auto payload = std::span<std::byte>(block->host).subspan(
            kBlockHeaderBytes,
            static_cast<std::size_t>(block->info.payload_bytes));
        const auto row = static_cast<std::uint32_t>(logical_row - block_begin);
        result = dsv4_physical_encode_kv_row(
            kind, values, row, payload);
        if (!result.ok()) return result;
        const auto layout = dsv4_physical_kv_layout(kind, capacity_rows);
        if (!layout.ok()) return {layout.errors};
        const auto data_offset =
            static_cast<std::uint64_t>(row) * layout.value.token_data_bytes;
        const auto scale_offset =
            static_cast<std::uint64_t>(capacity_rows) *
                layout.value.token_data_bytes +
            static_cast<std::uint64_t>(row) *
                layout.value.token_scale_bytes;
        const std::array<CudaBufferPatch, 2U> patches{{
            {data_offset,
             std::span<const std::byte>(payload).subspan(
                 static_cast<std::size_t>(data_offset),
                 layout.value.token_data_bytes)},
            {scale_offset,
             std::span<const std::byte>(payload).subspan(
                 static_cast<std::size_t>(scale_offset),
                 layout.value.token_scale_bytes)},
        }};
        for (std::size_t slot = 0U; slot < block->devices.size(); ++slot) {
            if (!block->devices[slot].valid()) continue;
            result = state_->cuda->update_buffer(block->devices[slot], patches);
            if (!result.ok()) return result;
            state_->metrics.host_to_device_bytes +=
                layout.value.token_data_bytes +
                layout.value.token_scale_bytes;
        }
    } else {
        for (std::size_t slot = 0U; slot < block->devices.size(); ++slot) {
            if (!block->devices[slot].valid()) continue;
            state_->metrics.device_used_bytes[slot] -=
                block->devices[slot].device_bytes();
            block->devices[slot] = {};
        }
        const auto offset = static_cast<std::size_t>(kBlockHeaderBytes) +
            static_cast<std::size_t>(logical_row - block_begin) * encoded.size();
        std::copy(encoded.begin(), encoded.end(), block->host.begin() + offset);
    }
    state_->metrics.host_write_bytes += encoded.size();
    block->info.used_rows = std::max(
        block->info.used_rows,
        static_cast<std::uint32_t>(logical_row - block_begin + 1U));
    ++table.end_row;
    target->tokens = std::max(target->tokens,
                              table.end_row * compression_ratio);
    if (kind == Dsv4KvBlockKind::Sliding) {
        const auto natural_minimum =
            table.end_row > state_->config.sliding_window_rows
                ? table.end_row - state_->config.sliding_window_rows : 0U;
        table.minimum_row = std::min(natural_minimum,
                                     sliding_retention_floor);
        while (!table.blocks.empty()) {
            const auto id = table.blocks.front();
            const auto& first = state_->blocks.at(id);
            const auto begin = first.info.logical_begin;
            if (begin + first.info.capacity_rows > table.minimum_row) break;
            table.blocks.erase(table.blocks.begin());
            state_->release_block(id);
        }
    }
    return result;
}

ParseResult<Dsv4KvPhysicalAppend>
Dsv4KvCache::reserve_physical_append(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint32_t compression_ratio,
    std::uint64_t logical_row, std::size_t device_slot) {
    ParseResult<Dsv4KvPhysicalAppend> result;
    auto* target = state_->sequence(sequence);
    const auto wanted_ratio = required_ratio(kind);
    const auto width = required_width(kind);
    const auto capacity_rows = dsv4_kv_block_rows(
        kind, compression_ratio, true);
    if (target == nullptr || !state_->config.physical_layout ||
        state_->cuda == nullptr || layer >= kDeepSeekV4ExecutionContract.layer_count ||
        compression_ratio != wanted_ratio || capacity_rows == 0U ||
        width == 0U || device_slot >= state_->config.devices.size() ||
        state_->config.device_capacity_bytes[device_slot] == 0U ||
        logical_row == std::numeric_limits<std::uint64_t>::max() ||
        logical_row + 1U > std::numeric_limits<std::uint64_t>::max() /
                               compression_ratio) {
        result.errors.emplace_back(
            "DeepSeek physical KV append reservation is invalid");
        return result;
    }

    const TableKey key{kind, layer};
    auto table_found = target->tables.find(key);
    if (table_found == target->tables.end()) {
        if (logical_row != 0U) {
            result.errors.emplace_back(
                "DeepSeek KV first reserved row must start at zero");
            return result;
        }
        try {
            table_found = target->tables.emplace(
                key, Dsv4KvCacheState::Table{
                         {}, 0U, 0U, compression_ratio}).first;
        } catch (const std::bad_alloc&) {
            result.errors.emplace_back(
                "cannot allocate DeepSeek reserved KV metadata");
            return result;
        }
    }
    auto& table = table_found->second;
    if (table.compression_ratio != compression_ratio ||
        logical_row != table.end_row) {
        result.errors.emplace_back(
            "DeepSeek reserved KV append position is not contiguous");
        return result;
    }
    const auto block_begin = logical_row / capacity_rows * capacity_rows;
    auto* block = state_->find_block(table, logical_row);
    if (block != nullptr && block->info.refcount > 1U) {
        const auto old_id = block->info.id;
        std::uint64_t replacement = 0U;
        auto status = state_->allocate_block(
            sequence, kind, layer, compression_ratio, block_begin,
            block, replacement);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        const auto id = std::find(
            table.blocks.begin(), table.blocks.end(), old_id);
        *id = replacement;
        state_->release_block(old_id);
        block = &state_->blocks.at(replacement);
        ++state_->metrics.copy_on_write_blocks;
    } else if (block == nullptr) {
        std::uint64_t id = 0U;
        auto status = state_->allocate_block(
            sequence, kind, layer, compression_ratio, block_begin,
            nullptr, id);
        if (!status.ok()) {
            result.errors = std::move(status.errors);
            return result;
        }
        try {
            table.blocks.push_back(id);
        } catch (const std::bad_alloc&) {
            state_->release_block(id);
            result.errors.emplace_back(
                "cannot grow DeepSeek reserved KV block table");
            return result;
        }
        block = &state_->blocks.at(id);
    }
    auto checked = state_->validate_block(*block);
    if (!checked.ok() || state_->total_leases(*block) != 0U) {
        result.errors = std::move(checked.errors);
        if (result.ok()) {
            result.errors.emplace_back(
                "DeepSeek KV cannot reserve an in-flight device block");
        }
        return result;
    }

    const auto physical_row = static_cast<std::uint32_t>(
        logical_row - block_begin);
    block->info.used_rows = std::max(
        block->info.used_rows, physical_row + 1U);
    ++table.end_row;
    target->tokens = std::max(
        target->tokens, table.end_row * compression_ratio);
    if (kind == Dsv4KvBlockKind::Sliding) {
        table.minimum_row = table.end_row > state_->config.sliding_window_rows
            ? table.end_row - state_->config.sliding_window_rows : 0U;
        while (!table.blocks.empty()) {
            const auto id = table.blocks.front();
            const auto& first = state_->blocks.at(id);
            const auto begin = first.info.logical_begin;
            if (begin + first.info.capacity_rows > table.minimum_row) break;
            table.blocks.erase(table.blocks.begin());
            state_->release_block(id);
        }
    }

    auto lease = acquire_device(
        sequence, kind, layer, logical_row, device_slot);
    if (!lease.ok()) {
        result.errors = std::move(lease.errors);
        return result;
    }
    const auto layout = dsv4_physical_kv_layout(kind, capacity_rows);
    if (!layout.ok()) {
        result.errors = layout.errors;
        return result;
    }
    const auto data_offset =
        static_cast<std::uint64_t>(physical_row) *
        layout.value.token_data_bytes;
    const auto scale_offset =
        static_cast<std::uint64_t>(capacity_rows) *
            layout.value.token_data_bytes +
        static_cast<std::uint64_t>(physical_row) *
            layout.value.token_scale_bytes;
    result.value = Dsv4KvPhysicalAppend(
        state_, std::move(lease.value), block->info.id, physical_row,
        data_offset, scale_offset, layout.value.token_data_bytes,
        layout.value.token_scale_bytes,
        block->host.data() + kBlockHeaderBytes,
        block->info.payload_bytes, block->info.row_width,
        block->info.kind, block->info.format);
    return result;
}

ParseResult<std::vector<float>> Dsv4KvCache::row(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint64_t logical_row) {
    ParseResult<std::vector<float>> result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        ++state_->metrics.misses;
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end() ||
        logical_row < table_found->second.minimum_row ||
        logical_row >= table_found->second.end_row) {
        result.errors.emplace_back("DeepSeek KV row is not retained");
        ++state_->metrics.misses;
        return result;
    }
    auto* block = state_->find_block(table_found->second, logical_row);
    if (block == nullptr) {
        result.errors.emplace_back("DeepSeek KV row block is unavailable");
        ++state_->metrics.misses;
        return result;
    }
    result.errors = state_->validate_block(*block).errors;
    if (!result.ok()) {
        ++state_->metrics.misses;
        return result;
    }
    const auto block_begin = block->info.logical_begin /
                             block->info.compression_ratio;
    const auto row_bytes = dsv4_kv_row_bytes(block->info.kind,
                                             block->info.format);
    try {
        result.value.resize(block->info.row_width);
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot allocate DeepSeek KV decoded row");
        ++state_->metrics.misses;
        return result;
    }
    if (block->info.format ==
            Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope ||
        block->info.format == Dsv4KvFormat::PhysicalFp8E4m3PerTensor) {
        const auto payload = std::span<const std::byte>(block->host).subspan(
            kBlockHeaderBytes,
            static_cast<std::size_t>(block->info.payload_bytes));
        result.errors = dsv4_physical_decode_kv_row(
            block->info.kind, payload,
            static_cast<std::uint32_t>(logical_row - block_begin),
            result.value).errors;
    } else {
        const auto offset = static_cast<std::size_t>(kBlockHeaderBytes) +
            static_cast<std::size_t>(logical_row - block_begin) * row_bytes;
        result.errors = dsv4_decode_kv_row(
            block->info.kind, block->info.format,
            std::span<const std::byte>(block->host).subspan(offset, row_bytes),
            result.value).errors;
    }
    if (!result.ok()) {
        result.value.clear();
        ++state_->metrics.misses;
        return result;
    }
    ++state_->metrics.hits;
    state_->metrics.gather_bytes += row_bytes;
    return result;
}

ParseResult<std::vector<float>> Dsv4KvCache::gather(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::span<const std::uint32_t> logical_rows) {
    ParseResult<std::vector<float>> result;
    const auto width = required_width(kind);
    if (logical_rows.size() >
        std::numeric_limits<std::size_t>::max() / width) {
        result.errors.emplace_back("DeepSeek KV gather size overflows");
        return result;
    }
    try {
        result.value.reserve(logical_rows.size() * width);
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back("cannot allocate DeepSeek KV gather output");
        return result;
    }
    for (const auto logical_row : logical_rows) {
        auto fetched = row(sequence, kind, layer, logical_row);
        if (!fetched.ok()) {
            result.errors = std::move(fetched.errors);
            result.value.clear();
            return result;
        }
        result.value.insert(result.value.end(), fetched.value.begin(),
                            fetched.value.end());
    }
    return result;
}

ParseResult<std::vector<CudaLightningIndexSegment>>
Dsv4KvCache::learned_index_segments(
    Dsv4SequenceHandle sequence, std::uint32_t layer,
    std::uint64_t rows) const {
    ParseResult<std::vector<CudaLightningIndexSegment>> result;
    const auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    const auto found = target->tables.find(
        TableKey{Dsv4KvBlockKind::LearnedIndex, layer});
    if (rows == 0U) return result;
    if (found == target->tables.end() || found->second.minimum_row != 0U ||
        rows > found->second.end_row) {
        result.errors.emplace_back(
            "DeepSeek learned-index compact rows are not retained");
        return result;
    }
    try {
        result.value.reserve(found->second.blocks.size());
    } catch (const std::bad_alloc&) {
        result.errors.emplace_back(
            "cannot allocate DeepSeek learned-index segment metadata");
        return result;
    }
    std::uint64_t emitted = 0U;
    for (const auto id : found->second.blocks) {
        if (emitted == rows) break;
        const auto block_found = state_->blocks.find(id);
        if (block_found == state_->blocks.end()) {
            result.errors.emplace_back(
                "DeepSeek learned-index block is unavailable");
            result.value.clear();
            return result;
        }
        const auto& block = block_found->second;
        const auto checked = state_->validate_block(block);
        const auto block_begin = block.info.logical_begin /
                                 block.info.compression_ratio;
        if (!checked.ok() ||
            block.info.format != Dsv4KvFormat::Fp4E2m1Group32 ||
            block_begin != emitted || block.info.used_rows == 0U) {
            result.errors.emplace_back(
                "DeepSeek learned-index compact block is invalid");
            result.value.clear();
            return result;
        }
        const auto segment_rows = static_cast<std::uint32_t>(std::min(
            rows - emitted,
            static_cast<std::uint64_t>(block.info.used_rows)));
        result.value.push_back(CudaLightningIndexSegment{
            nullptr, block.host, kBlockHeaderBytes, segment_rows});
        emitted += segment_rows;
    }
    if (emitted != rows) {
        result.errors.emplace_back(
            "DeepSeek learned-index compact history is incomplete");
        result.value.clear();
    }
    return result;
}

ParseResult<Dsv4KvDeviceLease> Dsv4KvCache::acquire_device(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer, std::uint64_t logical_row,
    std::size_t device_slot) {
    ParseResult<Dsv4KvDeviceLease> result;
    auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV device row is not retained");
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end() ||
        logical_row < table_found->second.minimum_row ||
        logical_row >= table_found->second.end_row) {
        result.errors.emplace_back("DeepSeek KV device row is not retained");
        return result;
    }
    auto* block = state_->find_block(table_found->second, logical_row);
    if (block == nullptr || device_slot >= state_->config.devices.size()) {
        result.errors.emplace_back("DeepSeek KV device slot or block is invalid");
        return result;
    }
    if (!block->devices[device_slot].valid()) {
        ++state_->metrics.misses;
        if (state_->cuda == nullptr ||
            state_->config.device_capacity_bytes[device_slot] == 0U) {
            result.errors.emplace_back(
                "DeepSeek KV device cache is not configured");
            return result;
        }
        result.errors = state_->validate_block(*block).errors;
        if (!result.ok()) return result;
        const bool physical_page =
            block->info.format ==
                Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope ||
            block->info.format ==
                Dsv4KvFormat::PhysicalFp8E4m3PerTensor;
        const auto upload = physical_page
            ? std::span<const std::byte>(block->host).subspan(
                  kBlockHeaderBytes,
                  static_cast<std::size_t>(block->info.payload_bytes))
            : std::span<const std::byte>(block->host);
        const auto bytes = static_cast<std::uint64_t>(upload.size());
        const auto capacity = state_->config.device_capacity_bytes[device_slot];
        if (bytes > capacity) {
            result.errors.emplace_back(
                "DeepSeek KV block exceeds the device cache capacity");
            return result;
        }
        while (state_->metrics.device_used_bytes[device_slot] >
               capacity - bytes) {
            Dsv4KvCacheState::Block* victim = nullptr;
            for (auto& [id, candidate] : state_->blocks) {
                static_cast<void>(id);
                if (!candidate.devices[device_slot].valid() ||
                    candidate.device_leases[device_slot] != 0U ||
                    &candidate == block) {
                    continue;
                }
                if (victim == nullptr ||
                    candidate.device_last_use[device_slot] <
                        victim->device_last_use[device_slot]) {
                    victim = &candidate;
                }
            }
            if (victim == nullptr) {
                result.errors.emplace_back(
                    "DeepSeek KV in-flight blocks exhaust the device cache");
                return result;
            }
            state_->metrics.device_used_bytes[device_slot] -=
                victim->devices[device_slot].device_bytes();
            victim->devices[device_slot] = {};
            ++state_->metrics.evictions;
        }
        CudaBuffer uploaded;
        const auto promotion_started = std::chrono::steady_clock::now();
        const auto uploaded_result = state_->cuda->upload_buffer(
            state_->config.devices[device_slot],
            upload, uploaded);
        if (!uploaded_result.ok()) {
            result.errors = uploaded_result.errors;
            return result;
        }
        block->devices[device_slot] = std::move(uploaded);
        state_->metrics.device_used_bytes[device_slot] += bytes;
        state_->metrics.device_peak_bytes[device_slot] = std::max(
            state_->metrics.device_peak_bytes[device_slot],
            state_->metrics.device_used_bytes[device_slot]);
        ++state_->metrics.promotions;
        state_->metrics.promotion_nanoseconds +=
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - promotion_started)
                    .count());
        state_->metrics.host_to_device_bytes += bytes;
    } else {
        ++state_->metrics.hits;
    }
    if (block->device_leases[device_slot] ==
            std::numeric_limits<std::uint32_t>::max() ||
        block->info.in_flight == std::numeric_limits<std::uint32_t>::max()) {
        result.errors.emplace_back("DeepSeek KV in-flight count overflows");
        return result;
    }
    block->device_last_use[device_slot] = ++state_->clock;
    ++block->device_leases[device_slot];
    ++block->info.in_flight;
    result.value = Dsv4KvDeviceLease(state_, block->info.id, device_slot);
    return result;
}

ValidationResult Dsv4KvCache::block_table_into(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind, std::uint32_t layer,
    std::vector<Dsv4KvBlockInfo>& output) const {
    ValidationResult result;
    output.clear();
    const auto* target = state_->sequence(sequence);
    if (target == nullptr) {
        result.errors.emplace_back("DeepSeek KV sequence does not exist");
        return result;
    }
    const auto table_found = target->tables.find(TableKey{kind, layer});
    if (table_found == target->tables.end()) return result;
    output.reserve(table_found->second.blocks.size());
    for (const auto id : table_found->second.blocks) {
        const auto found = state_->blocks.find(id);
        if (found == state_->blocks.end()) continue;
        output.push_back(found->second.info);
    }
    return result;
}

ParseResult<std::vector<Dsv4KvBlockInfo>> Dsv4KvCache::block_table(
    Dsv4SequenceHandle sequence, Dsv4KvBlockKind kind,
    std::uint32_t layer) const {
    ParseResult<std::vector<Dsv4KvBlockInfo>> result;
    auto filled = block_table_into(sequence, kind, layer, result.value);
    if (!filled.ok()) result.errors = std::move(filled.errors);
    return result;
}

Dsv4KvCacheStats Dsv4KvCache::stats() const noexcept {
    auto result = state_->metrics;
    result.used_blocks = state_->blocks.size();
    return result;
}

}  // namespace strata
