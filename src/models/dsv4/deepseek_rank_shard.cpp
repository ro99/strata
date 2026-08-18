#include "strata/deepseek_rank_shard.hpp"

#include "strata/deepseek_checkpoint.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

namespace strata {

namespace {

constexpr std::int32_t kNoShardAxis = -1;

template <typename T>
void append_errors(ParseResult<T>& result, std::vector<std::string> errors) {
    for (auto& error : errors) result.errors.push_back(std::move(error));
}

bool checked_product(std::uint64_t left, std::uint64_t right,
                     std::uint64_t& output) noexcept {
    if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    output = left + right;
    return true;
}

std::uint64_t divide_round_up(std::uint64_t value,
                              std::uint64_t divisor) noexcept {
    return value / divisor + static_cast<std::uint64_t>(value % divisor != 0U);
}

bool shape2(const std::vector<std::uint64_t>& shape) noexcept {
    return shape.size() == 2U && shape[0] != 0U && shape[1] != 0U;
}

bool shape1(const std::vector<std::uint64_t>& shape) noexcept {
    return shape.size() == 1U && shape[0] != 0U;
}

bool shape_bytes(const std::vector<std::uint64_t>& shape,
                 SafetensorsDtype dtype, std::uint64_t& bytes) noexcept {
    if (shape.empty()) return false;
    std::uint64_t elements = 1U;
    for (const auto dimension : shape) {
        if (dimension == 0U || !checked_product(elements, dimension, elements)) {
            return false;
        }
    }
    const auto width = static_cast<std::uint64_t>(safetensors_dtype_bytes(dtype));
    return width != 0U && checked_product(elements, width, bytes);
}

std::uint64_t shard_alignment(Dsv4TensorEncoding encoding) noexcept {
    switch (encoding) {
        case Dsv4TensorEncoding::Fp8E4m3Block128: return 128U;
        case Dsv4TensorEncoding::Fp4E2m1Group32: return 32U;
        case Dsv4TensorEncoding::Plain: return 1U;
    }
    return 0U;
}

std::uint64_t weight_slice_alignment(Dsv4TensorEncoding encoding,
                                     SafetensorsDtype dtype) noexcept {
    switch (encoding) {
        case Dsv4TensorEncoding::Fp8E4m3Block128: return 128U;
        case Dsv4TensorEncoding::Fp4E2m1Group32: return 16U;
        case Dsv4TensorEncoding::Plain:
            return static_cast<std::uint64_t>(safetensors_dtype_bytes(dtype));
    }
    return 0U;
}

void append_slice_error(ValidationResult& result, std::string_view name,
                        std::string_view detail) {
    result.errors.emplace_back(std::string(name) + " " + std::string(detail));
}

void validate_slices(const std::vector<Dsv4RankShardSlice>& slices,
                     std::uint64_t source_offset, std::uint64_t source_bytes,
                     std::uint64_t local_bytes,
                     std::uint64_t alignment_bytes, std::string_view name,
                     ValidationResult& result) {
    if (slices.empty()) {
        append_slice_error(result, name, "has no payload slices");
        return;
    }
    if (alignment_bytes == 0U) {
        append_slice_error(result, name, "has zero slice alignment");
        return;
    }
    std::uint64_t expected_destination = 0U;
    std::uint64_t previous_relative_end = 0U;
    for (const auto& slice : slices) {
        if (slice.bytes == 0U) {
            append_slice_error(result, name, "contains an empty payload slice");
            continue;
        }
        if (slice.destination_offset != expected_destination) {
            append_slice_error(result, name,
                               "payload slices do not cover local storage contiguously");
        }
        if (slice.relative_offset < previous_relative_end ||
            slice.relative_offset % alignment_bytes != 0U ||
            slice.bytes % alignment_bytes != 0U) {
            append_slice_error(result, name,
                               "payload slice violates ordering or alignment");
        }
        std::uint64_t source_end = 0U;
        std::uint64_t destination_end = 0U;
        if (!checked_add(slice.relative_offset, slice.bytes, source_end) ||
            source_end > source_bytes ||
            !checked_add(slice.destination_offset, slice.bytes, destination_end) ||
            destination_end > local_bytes) {
            append_slice_error(result, name, "payload slice exceeds its declared extent");
        }
        std::uint64_t expected_source_offset = 0U;
        if (!checked_add(source_offset, slice.relative_offset,
                         expected_source_offset) ||
            slice.source_offset != expected_source_offset) {
            append_slice_error(result, name,
                               "payload slice absolute offset disagrees with its base");
        }
        if (!checked_add(expected_destination, slice.bytes, expected_destination)) {
            append_slice_error(result, name, "local payload extent overflows");
            return;
        }
        previous_relative_end = source_end;
    }
    if (expected_destination != local_bytes) {
        append_slice_error(result, name,
                           "payload slices do not match the local byte count");
    }
}

void validate_scale_shape(const Dsv4RankShardDescriptor& descriptor,
                          ValidationResult& result) {
    const auto packed_rows = descriptor.packed_shape[0];
    const auto packed_columns = descriptor.packed_shape[1];
    const auto expected_rows = descriptor.encoding == Dsv4TensorEncoding::Fp4E2m1Group32
                                   ? packed_rows
                                   : divide_round_up(packed_rows, 128U);
    const auto expected_columns = descriptor.encoding == Dsv4TensorEncoding::Fp4E2m1Group32
                                      ? divide_round_up(descriptor.logical_shape[1], 32U)
                                      : divide_round_up(packed_columns, 128U);
    if (descriptor.scale_shape !=
        std::vector<std::uint64_t>{expected_rows, expected_columns}) {
        result.errors.emplace_back(
            "rank-shard scale shape does not match the source block/group layout");
    }
}

bool source_shapes_match(const Dsv4RankShardDescriptor& descriptor,
                         const Dsv4ManifestTensor& weight,
                         const Dsv4ManifestTensor* scale,
                         ValidationResult& result) {
    bool matches = true;
    if (weight.shard != descriptor.weight_shard ||
        weight.source_offset != descriptor.weight_source_offset ||
        weight.source_bytes != descriptor.weight_source_bytes ||
        weight.source_dtype != descriptor.weight_dtype ||
        weight.source_shape != descriptor.packed_shape ||
        weight.encoding != descriptor.encoding) {
        result.errors.emplace_back(
            "rank-shard descriptor no longer matches the weight manifest entry");
        matches = false;
    }
    if (descriptor.encoding == Dsv4TensorEncoding::Plain) {
        if (scale != nullptr || !descriptor.scale_name.empty() ||
            descriptor.scale_source_bytes != 0U || !descriptor.scale_slices.empty()) {
            result.errors.emplace_back(
                "plain rank-shard descriptor unexpectedly carries a scale");
            matches = false;
        }
    } else if (scale == nullptr || descriptor.scale_name.empty() ||
               scale->shard != descriptor.scale_shard ||
               scale->source_offset != descriptor.scale_source_offset ||
               scale->source_bytes != descriptor.scale_source_bytes ||
               scale->source_dtype != descriptor.scale_dtype ||
               scale->source_shape != descriptor.scale_shape ||
               scale->encoding != descriptor.encoding) {
        result.errors.emplace_back(
            "rank-shard descriptor no longer matches the scale manifest entry");
        matches = false;
    }
    return matches;
}

void build_slices(Dsv4RankShardDescriptor& descriptor) {
    const auto weight_width = static_cast<std::uint64_t>(
        safetensors_dtype_bytes(descriptor.weight_dtype));
    const auto scale_width = static_cast<std::uint64_t>(
        safetensors_dtype_bytes(descriptor.scale_dtype));
    const auto rows = descriptor.packed_shape[0];
    const auto packed_columns = descriptor.packed_shape[1];

    if (descriptor.ownership == Dsv4ShardOwnership::Replicated) {
        descriptor.weight_slices.push_back(
            {descriptor.weight_source_offset, 0U, 0U,
             descriptor.weight_source_bytes});
        if (descriptor.encoding != Dsv4TensorEncoding::Plain) {
            descriptor.scale_slices.push_back(
                {descriptor.scale_source_offset, 0U, 0U,
                 descriptor.scale_source_bytes});
        }
        return;
    }

    if (descriptor.ownership == Dsv4ShardOwnership::ContiguousRows) {
        const auto local_rows = descriptor.local_packed_shape[0];
        const auto weight_row_bytes = packed_columns * weight_width;
        const auto relative = descriptor.rank * local_rows * weight_row_bytes;
        descriptor.weight_slices.push_back(
            {descriptor.weight_source_offset + relative, relative, 0U,
             local_rows * weight_row_bytes});
        if (descriptor.encoding != Dsv4TensorEncoding::Plain) {
            const auto local_scale_rows = descriptor.local_scale_shape[0];
            const auto scale_row_bytes = descriptor.scale_shape[1] * scale_width;
            const auto scale_relative =
                descriptor.rank * local_scale_rows * scale_row_bytes;
            descriptor.scale_slices.push_back(
                {descriptor.scale_source_offset + scale_relative, scale_relative,
                 0U, local_scale_rows * scale_row_bytes});
        }
        return;
    }

    const auto local_packed_columns = descriptor.local_packed_shape[1];
    const auto weight_row_bytes = packed_columns * weight_width;
    const auto local_weight_row_bytes = local_packed_columns * weight_width;
    for (std::uint64_t row = 0U; row < rows; ++row) {
        const auto relative = row * weight_row_bytes +
                              descriptor.rank * local_weight_row_bytes;
        descriptor.weight_slices.push_back(
            {descriptor.weight_source_offset + relative, relative,
             row * local_weight_row_bytes, local_weight_row_bytes});
    }
    if (descriptor.encoding != Dsv4TensorEncoding::Plain) {
        const auto scale_rows = descriptor.scale_shape[0];
        const auto full_scale_columns = descriptor.scale_shape[1];
        const auto local_scale_columns = descriptor.local_scale_shape[1];
        const auto scale_row_bytes = full_scale_columns * scale_width;
        const auto local_scale_row_bytes = local_scale_columns * scale_width;
        for (std::uint64_t row = 0U; row < scale_rows; ++row) {
            const auto relative = row * scale_row_bytes +
                                  descriptor.rank * local_scale_row_bytes;
            descriptor.scale_slices.push_back(
                {descriptor.scale_source_offset + relative, relative,
                 row * local_scale_row_bytes, local_scale_row_bytes});
        }
    }
}

void append_descriptor_shape_errors(const Dsv4RankShardDescriptor& descriptor,
                                    ValidationResult& result) {
    const bool rank_one_plain =
        descriptor.encoding == Dsv4TensorEncoding::Plain &&
        shape1(descriptor.logical_shape) && shape1(descriptor.packed_shape);
    if (rank_one_plain) {
        if (descriptor.ownership != Dsv4ShardOwnership::Replicated ||
            descriptor.local_logical_shape != descriptor.logical_shape ||
            descriptor.local_packed_shape != descriptor.packed_shape) {
            result.errors.emplace_back(
                "rank-one plain rank-shard tensors must remain replicated");
        }
        return;
    }
    if (!shape2(descriptor.logical_shape) || !shape2(descriptor.packed_shape)) {
        result.errors.emplace_back(
            "rank-shard logical and packed shapes must be positive rank-two tensors");
        return;
    }
    if (descriptor.logical_shape[0] != descriptor.packed_shape[0]) {
        result.errors.emplace_back("rank-shard logical and packed row counts disagree");
    }
    if (descriptor.encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
        std::uint64_t logical_width = 0U;
        if (!checked_product(descriptor.packed_shape[1], 2U, logical_width) ||
            logical_width != descriptor.logical_shape[1]) {
            result.errors.emplace_back(
                "rank-shard FP4 logical width is not twice its packed width");
        }
    } else if (descriptor.logical_shape != descriptor.packed_shape) {
        result.errors.emplace_back(
            "rank-shard non-FP4 logical and packed shapes disagree");
    }
    if (!shape2(descriptor.local_logical_shape) ||
        !shape2(descriptor.local_packed_shape) ||
        descriptor.local_logical_shape[0] != descriptor.local_packed_shape[0]) {
        result.errors.emplace_back("rank-shard local shapes are malformed");
    } else if (descriptor.encoding == Dsv4TensorEncoding::Fp4E2m1Group32 &&
               (descriptor.local_packed_shape[1] >
                    std::numeric_limits<std::uint64_t>::max() / 2U ||
                descriptor.local_packed_shape[1] * 2U !=
                    descriptor.local_logical_shape[1])) {
        result.errors.emplace_back("rank-shard local FP4 shapes disagree");
    } else if (descriptor.encoding != Dsv4TensorEncoding::Fp4E2m1Group32 &&
               descriptor.local_logical_shape != descriptor.local_packed_shape) {
        result.errors.emplace_back("rank-shard local logical and packed shapes disagree");
    }
}

}  // namespace

std::string_view to_string(Dsv4ShardOwnership ownership) noexcept {
    switch (ownership) {
        case Dsv4ShardOwnership::Replicated: return "replicated";
        case Dsv4ShardOwnership::ContiguousRows: return "contiguous_rows";
        case Dsv4ShardOwnership::StridedColumns: return "strided_columns";
    }
    return "unknown";
}

ValidationResult validate_dsv4_rank_shard_descriptor(
    const Dsv4RankShardDescriptor& descriptor) {
    ValidationResult result;
    if (descriptor.base_name.empty() || descriptor.weight_name.empty()) {
        result.errors.emplace_back("rank-shard descriptor has no tensor name");
    }
    if (descriptor.weight_shard.empty()) {
        result.errors.emplace_back("rank-shard descriptor has no weight source shard");
    }
    if (descriptor.world_size == 0U || descriptor.rank >= descriptor.world_size) {
        result.errors.emplace_back("rank-shard rank/world size is invalid");
    }
    const bool replicated = descriptor.ownership == Dsv4ShardOwnership::Replicated;
    const bool rows = descriptor.ownership == Dsv4ShardOwnership::ContiguousRows;
    const bool columns = descriptor.ownership == Dsv4ShardOwnership::StridedColumns;
    if (!replicated && !rows && !columns) {
        result.errors.emplace_back("rank-shard ownership class is invalid");
    }
    if ((replicated && descriptor.shard_axis != kNoShardAxis) ||
        (rows && descriptor.shard_axis != 0) ||
        (columns && descriptor.shard_axis != 1)) {
        result.errors.emplace_back("rank-shard axis disagrees with ownership class");
    }
    append_descriptor_shape_errors(descriptor, result);
    const bool rank_one_plain =
        descriptor.encoding == Dsv4TensorEncoding::Plain &&
        shape1(descriptor.logical_shape) && shape1(descriptor.packed_shape);
    if ((!rank_one_plain &&
         (!shape2(descriptor.logical_shape) || !shape2(descriptor.packed_shape)))) {
        return result;
    }

    const auto weight_width = static_cast<std::uint64_t>(
        safetensors_dtype_bytes(descriptor.weight_dtype));
    std::uint64_t expected_weight_bytes = 0U;
    if (weight_width == 0U ||
        !shape_bytes(descriptor.packed_shape, descriptor.weight_dtype,
                     expected_weight_bytes) ||
        expected_weight_bytes != descriptor.weight_source_bytes) {
        result.errors.emplace_back(
            "rank-shard packed weight bytes disagree with dtype and shape");
    }
    if (descriptor.encoding == Dsv4TensorEncoding::Plain) {
        if (!descriptor.scale_shape.empty() || !descriptor.scale_shard.empty() ||
            descriptor.scale_source_bytes != 0U ||
            descriptor.local_scale_bytes != 0U || !descriptor.scale_slices.empty() ||
            descriptor.scale_dtype != SafetensorsDtype::Other) {
            result.errors.emplace_back("plain rank-shard descriptor has scale state");
        }
    } else {
        if (!shape2(descriptor.packed_shape) || !shape2(descriptor.scale_shape) ||
            descriptor.scale_shard.empty() ||
            descriptor.scale_dtype != SafetensorsDtype::F8E8M0) {
            result.errors.emplace_back("quantized rank-shard scale descriptor is malformed");
        } else {
            if (descriptor.scale_shard != descriptor.weight_shard) {
                result.errors.emplace_back(
                    "quantized rank-shard weight and scale cross source shards");
            }
            validate_scale_shape(descriptor, result);
            std::uint64_t expected_scale_bytes = 0U;
            if (!shape_bytes(descriptor.scale_shape, descriptor.scale_dtype,
                             expected_scale_bytes) ||
                expected_scale_bytes != descriptor.scale_source_bytes) {
                result.errors.emplace_back(
                    "rank-shard scale bytes disagree with dtype and shape");
            }
        }
    }
    if (descriptor.encoding == Dsv4TensorEncoding::Fp8E4m3Block128 &&
        descriptor.weight_dtype != SafetensorsDtype::F8E4M3) {
        result.errors.emplace_back("FP8 rank-shard weight must retain F8_E4M3");
    }
    if (descriptor.encoding == Dsv4TensorEncoding::Fp4E2m1Group32 &&
        descriptor.weight_dtype != SafetensorsDtype::I8) {
        result.errors.emplace_back("native FP4 rank-shard weight must retain source I8 packing");
    }
    const auto block = shard_alignment(descriptor.encoding);
    if (block == 0U || descriptor.logical_shard_alignment != block ||
        descriptor.weight_slice_alignment_bytes == 0U ||
        descriptor.scale_slice_alignment_bytes == 0U) {
        result.errors.emplace_back("rank-shard block/group alignment is invalid");
    }

    if (rank_one_plain && !replicated) {
        result.errors.emplace_back("rank-one plain tensor cannot be row/column sharded");
    } else if (replicated) {
        if (descriptor.local_logical_shape != descriptor.logical_shape ||
            descriptor.local_packed_shape != descriptor.packed_shape ||
            descriptor.local_scale_shape != descriptor.scale_shape ||
            descriptor.local_weight_bytes != descriptor.weight_source_bytes ||
            descriptor.local_scale_bytes != descriptor.scale_source_bytes) {
            result.errors.emplace_back("replicated rank-shard local shape is not full shape");
        }
    } else if (rows) {
        if (descriptor.logical_shape[0] % descriptor.world_size != 0U ||
            descriptor.local_logical_shape[1] != descriptor.logical_shape[1] ||
            descriptor.local_packed_shape[1] != descriptor.packed_shape[1]) {
            result.errors.emplace_back(
                "contiguous-row rank shard is not divisible or changes its input width");
        }
        if (descriptor.encoding == Dsv4TensorEncoding::Fp8E4m3Block128 &&
            descriptor.local_logical_shape[0] % 128U != 0U) {
            result.errors.emplace_back(
                "FP8 contiguous-row shard splits a 128-row scale block");
        }
    } else if (columns) {
        if (descriptor.logical_shape[1] % descriptor.world_size != 0U ||
            descriptor.local_logical_shape[0] != descriptor.logical_shape[0] ||
            descriptor.local_packed_shape[0] != descriptor.packed_shape[0]) {
            result.errors.emplace_back(
                "strided-column rank shard is not divisible or changes its output rows");
        }
        if (descriptor.logical_shard_alignment != 1U &&
            descriptor.local_logical_shape[1] % descriptor.logical_shard_alignment != 0U) {
            result.errors.emplace_back(
                "strided-column rank shard splits a block/group along its input width");
        }
    }
    if (descriptor.local_scale_shape.size() == 2U &&
        descriptor.encoding != Dsv4TensorEncoding::Plain) {
        if (rows && descriptor.scale_shape[0] % descriptor.world_size != 0U) {
            result.errors.emplace_back("row shard scale rows are not divisible by world size");
        }
        if (columns && descriptor.scale_shape[1] % descriptor.world_size != 0U) {
            result.errors.emplace_back("column shard scale columns are not divisible by world size");
        }
    }

    validate_slices(descriptor.weight_slices, descriptor.weight_source_offset,
                    descriptor.weight_source_bytes, descriptor.local_weight_bytes,
                    descriptor.weight_slice_alignment_bytes, "weight", result);
    if (descriptor.encoding != Dsv4TensorEncoding::Plain) {
        validate_slices(descriptor.scale_slices, descriptor.scale_source_offset,
                        descriptor.scale_source_bytes, descriptor.local_scale_bytes,
                        descriptor.scale_slice_alignment_bytes, "scale", result);
    }
    return result;
}

ParseResult<Dsv4RankShardDescriptor> describe_dsv4_rank_shard(
    const Dsv4CheckpointReader& checkpoint, std::string_view base_name,
    Dsv4ShardOwnership ownership, std::uint32_t rank,
    std::uint32_t world_size) {
    ParseResult<Dsv4RankShardDescriptor> result;
    result.value.base_name = std::string(base_name);
    result.value.weight_name = result.value.base_name + ".weight";
    result.value.ownership = ownership;
    result.value.rank = rank;
    result.value.world_size = world_size;
    result.value.shard_axis = ownership == Dsv4ShardOwnership::Replicated
                                  ? kNoShardAxis
                                  : ownership == Dsv4ShardOwnership::ContiguousRows ? 0 : 1;

    if (world_size == 0U || rank >= world_size) {
        result.errors.emplace_back("rank-shard rank/world size is invalid");
        return result;
    }
    if (ownership != Dsv4ShardOwnership::Replicated &&
        ownership != Dsv4ShardOwnership::ContiguousRows &&
        ownership != Dsv4ShardOwnership::StridedColumns) {
        result.errors.emplace_back("rank-shard ownership class is invalid");
        return result;
    }

    const auto* weight = checkpoint.find(result.value.weight_name);
    if (weight == nullptr) {
        result.value.weight_name = result.value.base_name;
        weight = checkpoint.find(result.value.weight_name);
    }
    if (weight == nullptr) {
        result.errors.emplace_back("rank-shard weight is missing: " +
                                   result.value.weight_name);
        return result;
    }
    result.value.encoding = weight->encoding;
    result.value.weight_dtype = weight->source_dtype;
    result.value.weight_shard = weight->shard;
    result.value.packed_shape = weight->source_shape;
    result.value.logical_shape = result.value.packed_shape;
    result.value.weight_source_offset = weight->source_offset;
    result.value.weight_source_bytes = weight->source_bytes;
    if (!shape2(result.value.packed_shape) &&
        !(result.value.encoding == Dsv4TensorEncoding::Plain &&
          shape1(result.value.packed_shape))) {
        result.errors.emplace_back(
            "rank-shard source weight must be a positive rank-two tensor, or a rank-one plain tensor");
        return result;
    }
    if (result.value.encoding == Dsv4TensorEncoding::Fp4E2m1Group32) {
        if (result.value.packed_shape.size() != 2U ||
            result.value.packed_shape[1] >
                std::numeric_limits<std::uint64_t>::max() / 2U) {
            result.errors.emplace_back("native FP4 packed shape cannot form a logical width");
            return result;
        }
        result.value.logical_shape[1] = result.value.packed_shape[1] * 2U;
    }

    if (result.value.encoding != Dsv4TensorEncoding::Plain) {
        result.value.scale_name = result.value.base_name + ".scale";
        const auto* scale = checkpoint.find(result.value.scale_name);
        if (scale == nullptr) {
            result.errors.emplace_back("rank-shard scale is missing: " +
                                       result.value.scale_name);
            return result;
        }
        result.value.scale_dtype = scale->source_dtype;
        result.value.scale_shard = scale->shard;
        result.value.scale_shape = scale->source_shape;
        result.value.scale_source_offset = scale->source_offset;
        result.value.scale_source_bytes = scale->source_bytes;
    }

    if (result.value.encoding == Dsv4TensorEncoding::Plain &&
        result.value.packed_shape.size() == 1U) {
        result.value.local_logical_shape = result.value.logical_shape;
        result.value.local_packed_shape = result.value.packed_shape;
    } else {
        const auto local_rows = ownership == Dsv4ShardOwnership::ContiguousRows
                                    ? result.value.logical_shape[0] / world_size
                                    : result.value.logical_shape[0];
        const auto local_logical_columns =
            ownership == Dsv4ShardOwnership::StridedColumns
                ? result.value.logical_shape[1] / world_size
                : result.value.logical_shape[1];
        const auto local_packed_columns =
            result.value.encoding == Dsv4TensorEncoding::Fp4E2m1Group32
                ? local_logical_columns / 2U
                : local_logical_columns;
        result.value.local_logical_shape = {local_rows, local_logical_columns};
        result.value.local_packed_shape = {local_rows, local_packed_columns};
    }
    if (result.value.encoding != Dsv4TensorEncoding::Plain) {
        const auto local_scale_rows = ownership == Dsv4ShardOwnership::ContiguousRows
                                          ? result.value.scale_shape[0] / world_size
                                          : result.value.scale_shape[0];
        const auto local_scale_columns = ownership == Dsv4ShardOwnership::StridedColumns
                                             ? result.value.scale_shape[1] / world_size
                                             : result.value.scale_shape[1];
        result.value.local_scale_shape = {local_scale_rows, local_scale_columns};
    }
    std::uint64_t local_weight_elements = 1U;
    for (const auto dimension : result.value.local_packed_shape) {
        if (!checked_product(local_weight_elements, dimension,
                             local_weight_elements)) {
            result.errors.emplace_back("rank-shard local weight byte count overflows");
            return result;
        }
    }
    if (!checked_product(local_weight_elements,
                         safetensors_dtype_bytes(result.value.weight_dtype),
                         result.value.local_weight_bytes)) {
        result.errors.emplace_back("rank-shard local weight byte count overflows");
        return result;
    }
    if (result.value.encoding != Dsv4TensorEncoding::Plain) {
        std::uint64_t local_scale_elements = 0U;
        if (!checked_product(result.value.local_scale_shape[0],
                             result.value.local_scale_shape[1], local_scale_elements) ||
            !checked_product(local_scale_elements,
                             safetensors_dtype_bytes(result.value.scale_dtype),
                             result.value.local_scale_bytes)) {
            result.errors.emplace_back("rank-shard local scale byte count overflows");
            return result;
        }
    }
    result.value.logical_shard_alignment = shard_alignment(result.value.encoding);
    result.value.weight_slice_alignment_bytes = weight_slice_alignment(
        result.value.encoding, result.value.weight_dtype);
    result.value.scale_slice_alignment_bytes = result.value.encoding ==
                                                       Dsv4TensorEncoding::Plain
                                                   ? 1U
                                                   : static_cast<std::uint64_t>(
                                                         safetensors_dtype_bytes(
                                                             result.value.scale_dtype));
    build_slices(result.value);
    const auto valid = validate_dsv4_rank_shard_descriptor(result.value);
    if (!valid.ok()) append_errors(result, std::move(valid.errors));
    return result;
}

ParseResult<Dsv4RankShardPayload> load_dsv4_rank_shard(
    const Dsv4CheckpointReader& checkpoint,
    const Dsv4RankShardDescriptor& descriptor) {
    ParseResult<Dsv4RankShardPayload> result;
    const auto valid = validate_dsv4_rank_shard_descriptor(descriptor);
    if (!valid.ok()) {
        append_errors(result, std::move(valid.errors));
        return result;
    }
    const auto* weight = checkpoint.find(descriptor.weight_name);
    const auto* scale = descriptor.encoding == Dsv4TensorEncoding::Plain
                            ? nullptr
                            : checkpoint.find(descriptor.scale_name);
    if (weight == nullptr) {
        result.errors.emplace_back("rank-shard weight disappeared: " +
                                   descriptor.weight_name);
        return result;
    }
    ValidationResult source_valid;
    if (!source_shapes_match(descriptor, *weight, scale, source_valid)) {
        append_errors(result, std::move(source_valid.errors));
        return result;
    }

    result.value.weight.resize(static_cast<std::size_t>(descriptor.local_weight_bytes));
    if (descriptor.encoding != Dsv4TensorEncoding::Plain) {
        result.value.scale.resize(static_cast<std::size_t>(descriptor.local_scale_bytes));
    }
    Dsv4CheckpointReadStats reader_stats;
    for (const auto& slice : descriptor.weight_slices) {
        auto loaded = checkpoint.read_slice_into(
            descriptor.weight_name, slice.relative_offset,
            std::span<std::byte>(result.value.weight.data() + slice.destination_offset,
                                 static_cast<std::size_t>(slice.bytes)),
            &reader_stats);
        if (!loaded.ok()) {
            append_errors(result, std::move(loaded.errors));
            result.value.weight.clear();
            result.value.scale.clear();
            return result;
        }
    }
    for (const auto& slice : descriptor.scale_slices) {
        auto loaded = checkpoint.read_slice_into(
            descriptor.scale_name, slice.relative_offset,
            std::span<std::byte>(result.value.scale.data() + slice.destination_offset,
                                 static_cast<std::size_t>(slice.bytes)),
            &reader_stats);
        if (!loaded.ok()) {
            append_errors(result, std::move(loaded.errors));
            result.value.weight.clear();
            result.value.scale.clear();
            return result;
        }
    }
    if (reader_stats.bytes != descriptor.local_weight_bytes +
                                  descriptor.local_scale_bytes) {
        result.errors.emplace_back("rank-shard read bytes do not match local payload bytes");
        result.value.weight.clear();
        result.value.scale.clear();
        return result;
    }
    result.value.stats.calls = reader_stats.calls;
    result.value.stats.bytes = reader_stats.bytes;
    return result;
}

}  // namespace strata
