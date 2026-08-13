#include "test.hpp"

#include "strata/deepseek_checkpoint.hpp"
#include "strata/deepseek_rank_shard.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct FixtureCase {
    std::string base;
    strata::Dsv4ShardOwnership ownership;
    strata::SafetensorsDtype weight_dtype;
    std::vector<std::uint64_t> packed_shape;
    std::vector<std::uint64_t> scale_shape;
};

std::filesystem::path dsv4_model_directory() {
    if (const auto* environment = std::getenv("STRATA_DSV4_MODEL_DIR");
        environment != nullptr && *environment != '\0') {
        return environment;
    }
    return std::filesystem::path(STRATA_SOURCE_DIR) / "models/dsv4f";
}

void require_disjoint(const std::vector<strata::Dsv4RankShardSlice>& left,
                      const std::vector<strata::Dsv4RankShardSlice>& right) {
    for (const auto& first : left) {
        for (const auto& second : right) {
            const auto first_end = first.relative_offset + first.bytes;
            const auto second_end = second.relative_offset + second.bytes;
            REQUIRE(first_end <= second.relative_offset ||
                    second_end <= first.relative_offset);
        }
    }
}

void require_slices_match_source(
    const strata::Dsv4CheckpointReader& checkpoint,
    std::string_view tensor_name,
    const strata::Dsv4RankShardDescriptor& rank0,
    const strata::Dsv4RankShardPayload& payload0,
    const strata::Dsv4RankShardDescriptor& rank1,
    const strata::Dsv4RankShardPayload& payload1,
    bool replicated) {
    constexpr std::uint64_t kSourceChunkBytes = 8ULL << 20U;
    if (!replicated) require_disjoint(rank0.weight_slices, rank1.weight_slices);
    const auto check_rank = [&](const auto& descriptor, const auto& payload) {
        std::uint64_t covered = 0U;
        for (const auto& slice : descriptor.weight_slices) {
            for (std::uint64_t offset = 0U; offset < slice.bytes;) {
                const auto chunk = std::min(kSourceChunkBytes, slice.bytes - offset);
                const auto source = checkpoint.read_slice(
                    tensor_name, slice.relative_offset + offset, chunk);
                REQUIRE(source.ok());
                REQUIRE(std::equal(
                    source.value.begin(), source.value.end(),
                    payload.weight.begin() +
                        static_cast<std::ptrdiff_t>(slice.destination_offset + offset)));
                offset += chunk;
            }
            covered += slice.bytes;
        }
        return covered;
    };
    const auto covered0 = check_rank(rank0, payload0);
    const auto covered1 = check_rank(rank1, payload1);
    if (replicated) REQUIRE(covered0 == rank0.weight_source_bytes);
    else REQUIRE(covered0 + covered1 == rank0.weight_source_bytes);
}

void require_scale_slices_match_source(
    const strata::Dsv4CheckpointReader& checkpoint,
    std::string_view tensor_name,
    const strata::Dsv4RankShardDescriptor& rank0,
    const strata::Dsv4RankShardPayload& payload0,
    const strata::Dsv4RankShardDescriptor& rank1,
    const strata::Dsv4RankShardPayload& payload1,
    bool replicated) {
    constexpr std::uint64_t kSourceChunkBytes = 8ULL << 20U;
    if (!replicated) require_disjoint(rank0.scale_slices, rank1.scale_slices);
    const auto check_rank = [&](const auto& descriptor, const auto& payload) {
        std::uint64_t covered = 0U;
        for (const auto& slice : descriptor.scale_slices) {
            for (std::uint64_t offset = 0U; offset < slice.bytes;) {
                const auto chunk = std::min(kSourceChunkBytes, slice.bytes - offset);
                const auto source = checkpoint.read_slice(
                    tensor_name, slice.relative_offset + offset, chunk);
                REQUIRE(source.ok());
                REQUIRE(std::equal(
                    source.value.begin(), source.value.end(),
                    payload.scale.begin() +
                        static_cast<std::ptrdiff_t>(slice.destination_offset + offset)));
                offset += chunk;
            }
            covered += slice.bytes;
        }
        return covered;
    };
    const auto covered0 = check_rank(rank0, payload0);
    const auto covered1 = check_rank(rank1, payload1);
    if (replicated) REQUIRE(covered0 == rank0.scale_source_bytes);
    else REQUIRE(covered0 + covered1 == rank0.scale_source_bytes);
}

void check_fixture(const strata::Dsv4CheckpointReader& checkpoint,
                   const FixtureCase& fixture) {
    const auto* source_weight = checkpoint.find(fixture.base + ".weight");
    const auto* source_direct = checkpoint.find(fixture.base);
    const auto* weight = source_weight != nullptr ? source_weight : source_direct;
    REQUIRE(weight != nullptr);
    REQUIRE(weight->source_dtype == fixture.weight_dtype);
    REQUIRE(weight->source_shape == fixture.packed_shape);

    const auto rank0 = strata::describe_dsv4_rank_shard(
        checkpoint, fixture.base, fixture.ownership, 0U, 2U);
    const auto rank1 = strata::describe_dsv4_rank_shard(
        checkpoint, fixture.base, fixture.ownership, 1U, 2U);
    REQUIRE(rank0.ok());
    REQUIRE(rank1.ok());
    REQUIRE(rank0.value.weight_source_offset == weight->source_offset);
    REQUIRE(rank1.value.weight_source_offset == weight->source_offset);
    REQUIRE(rank0.value.weight_source_bytes == weight->source_bytes);
    REQUIRE(rank1.value.weight_source_bytes == weight->source_bytes);
    REQUIRE(rank0.value.weight_shard == weight->shard);
    REQUIRE(rank1.value.weight_shard == weight->shard);
    REQUIRE(rank0.value.weight_dtype == weight->source_dtype);
    REQUIRE(rank1.value.weight_dtype == weight->source_dtype);
    REQUIRE(rank0.value.encoding == weight->encoding);
    REQUIRE(rank1.value.encoding == weight->encoding);
    REQUIRE(rank0.value.scale_shape == fixture.scale_shape);
    REQUIRE(rank1.value.scale_shape == fixture.scale_shape);
    REQUIRE(strata::validate_dsv4_rank_shard_descriptor(rank0.value).ok());
    REQUIRE(strata::validate_dsv4_rank_shard_descriptor(rank1.value).ok());

    const auto loaded0 = strata::load_dsv4_rank_shard(checkpoint, rank0.value);
    const auto loaded1 = strata::load_dsv4_rank_shard(checkpoint, rank1.value);
    REQUIRE(loaded0.ok());
    REQUIRE(loaded1.ok());
    REQUIRE(loaded0.value.weight.size() == rank0.value.local_weight_bytes);
    REQUIRE(loaded1.value.weight.size() == rank1.value.local_weight_bytes);
    REQUIRE(loaded0.value.scale.size() == rank0.value.local_scale_bytes);
    REQUIRE(loaded1.value.scale.size() == rank1.value.local_scale_bytes);
    REQUIRE(loaded0.value.stats.bytes == rank0.value.local_weight_bytes +
                                            rank0.value.local_scale_bytes);
    REQUIRE(loaded1.value.stats.bytes == rank1.value.local_weight_bytes +
                                            rank1.value.local_scale_bytes);
    REQUIRE(loaded0.value.stats.calls == rank0.value.weight_slices.size() +
                                            rank0.value.scale_slices.size());
    REQUIRE(loaded1.value.stats.calls == rank1.value.weight_slices.size() +
                                            rank1.value.scale_slices.size());

    require_slices_match_source(
        checkpoint, weight->name, rank0.value, loaded0.value, rank1.value,
        loaded1.value,
        fixture.ownership == strata::Dsv4ShardOwnership::Replicated);
    if (fixture.scale_shape.empty()) {
        REQUIRE(loaded0.value.scale.empty());
        REQUIRE(loaded1.value.scale.empty());
    } else {
        const auto* scale = checkpoint.find(fixture.base + ".scale");
        REQUIRE(scale != nullptr);
        REQUIRE(rank0.value.scale_shard == scale->shard);
        REQUIRE(rank1.value.scale_shard == scale->shard);
        REQUIRE(rank0.value.scale_source_offset == scale->source_offset);
        REQUIRE(rank1.value.scale_source_offset == scale->source_offset);
        REQUIRE(rank0.value.scale_source_bytes == scale->source_bytes);
        REQUIRE(rank1.value.scale_source_bytes == scale->source_bytes);
        REQUIRE(rank0.value.scale_dtype == scale->source_dtype);
        REQUIRE(rank1.value.scale_dtype == scale->source_dtype);
        REQUIRE(rank0.value.scale_shape == scale->source_shape);
        REQUIRE(rank1.value.scale_shape == scale->source_shape);
        REQUIRE(rank0.value.encoding == scale->encoding);
        REQUIRE(rank1.value.encoding == scale->encoding);
        require_scale_slices_match_source(
            checkpoint, scale->name, rank0.value, loaded0.value, rank1.value,
            loaded1.value,
            fixture.ownership == strata::Dsv4ShardOwnership::Replicated);
    }
}

strata::Dsv4RankShardDescriptor valid_descriptor() {
    strata::Dsv4RankShardDescriptor descriptor;
    descriptor.base_name = "synthetic";
    descriptor.weight_name = "synthetic.weight";
    descriptor.scale_name = "synthetic.scale";
    descriptor.weight_shard = "synthetic.safetensors";
    descriptor.scale_shard = "synthetic.safetensors";
    descriptor.ownership = strata::Dsv4ShardOwnership::ContiguousRows;
    descriptor.rank = 0U;
    descriptor.world_size = 2U;
    descriptor.shard_axis = 0;
    descriptor.encoding = strata::Dsv4TensorEncoding::Fp8E4m3Block128;
    descriptor.weight_dtype = strata::SafetensorsDtype::F8E4M3;
    descriptor.scale_dtype = strata::SafetensorsDtype::F8E8M0;
    descriptor.logical_shape = {256U, 256U};
    descriptor.packed_shape = {256U, 256U};
    descriptor.scale_shape = {2U, 2U};
    descriptor.local_logical_shape = {128U, 256U};
    descriptor.local_packed_shape = {128U, 256U};
    descriptor.local_scale_shape = {1U, 2U};
    descriptor.weight_source_offset = 1000U;
    descriptor.weight_source_bytes = 256U * 256U;
    descriptor.scale_source_offset = 2000U;
    descriptor.scale_source_bytes = 4U;
    descriptor.local_weight_bytes = 128U * 256U;
    descriptor.local_scale_bytes = 2U;
    descriptor.logical_shard_alignment = 128U;
    descriptor.weight_slice_alignment_bytes = 128U;
    descriptor.scale_slice_alignment_bytes = 1U;
    descriptor.weight_slices.push_back(
        {1000U, 0U, 0U, descriptor.local_weight_bytes});
    descriptor.scale_slices.push_back({2000U, 0U, 0U, 2U});
    return descriptor;
}

}  // namespace

TEST_CASE("DeepSeek TP2 descriptor rejects malformed rank and alignment contracts") {
    const auto valid = valid_descriptor();
    REQUIRE(strata::validate_dsv4_rank_shard_descriptor(valid).ok());

    auto bad_rank = valid;
    bad_rank.rank = 2U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_rank).ok());

    auto bad_world = valid;
    bad_world.world_size = 3U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_world).ok());

    auto bad_scale = valid;
    bad_scale.scale_shape = {2U, 1U};
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_scale).ok());

    auto bad_alignment = valid;
    bad_alignment.weight_slices[0].relative_offset = 1U;
    bad_alignment.weight_slices[0].source_offset = 1001U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_alignment).ok());

    auto bad_extent = valid;
    bad_extent.local_weight_bytes -= 128U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_extent).ok());

    auto bad_source_extent = valid;
    bad_source_extent.weight_source_bytes -= 128U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_source_extent).ok());

    auto bad_source_offset = valid;
    bad_source_offset.weight_slices[0].source_offset += 128U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_source_offset).ok());

    auto bad_scale_extent = valid;
    bad_scale_extent.scale_source_bytes -= 1U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_scale_extent).ok());

    auto bad_scale_alignment = valid;
    bad_scale_alignment.scale_slices[0].relative_offset = 1U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_scale_alignment).ok());

    auto bad_row_divisibility = valid;
    bad_row_divisibility.logical_shape[0] = 255U;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_row_divisibility).ok());

    auto bad_precision = valid;
    bad_precision.weight_dtype = strata::SafetensorsDtype::Bf16;
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_precision).ok());

    auto bad_weight_shard = valid;
    bad_weight_shard.weight_shard.clear();
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_weight_shard).ok());

    auto bad_scale_shard = valid;
    bad_scale_shard.scale_shard = "other.safetensors";
    REQUIRE(!strata::validate_dsv4_rank_shard_descriptor(bad_scale_shard).ok());
}

TEST_CASE("DeepSeek TP2 actual early middle late rank shards reconstruct byte exactly") {
    const auto model = dsv4_model_directory();
    if (!std::filesystem::exists(model / "model.safetensors.index.json")) {
        SKIP("actual DeepSeek V4 checkpoint fixture is absent");
    }
    const auto opened = strata::Dsv4CheckpointReader::open(model.string());
    REQUIRE(opened.ok());
    const auto& checkpoint = *opened.value;

    const auto fp8 = strata::SafetensorsDtype::F8E4M3;
    const auto fp4 = strata::SafetensorsDtype::I8;
    const auto bf16 = strata::SafetensorsDtype::Bf16;
    const auto f32 = strata::SafetensorsDtype::F32;
    const auto i64 = strata::SafetensorsDtype::I64;
    const auto rows = strata::Dsv4ShardOwnership::ContiguousRows;
    const auto columns = strata::Dsv4ShardOwnership::StridedColumns;
    const auto replicated = strata::Dsv4ShardOwnership::Replicated;
    const auto no_scale = std::vector<std::uint64_t>{};
    std::vector<FixtureCase> fixtures{
        {"embed", rows, bf16, {129280U, 4096U}, no_scale},
        {"head", rows, bf16, {129280U, 4096U}, no_scale},
        {"layers.2.attn.wq_a", replicated, fp8, {1024U, 4096U}, {8U, 32U}},
        {"layers.2.attn.wkv", replicated, fp8, {512U, 4096U}, {4U, 32U}},
        {"layers.2.attn.wq_b", rows, fp8, {32768U, 1024U}, {256U, 8U}},
        {"layers.2.attn.wo_a", rows, fp8, {8192U, 4096U}, {64U, 32U}},
        {"layers.2.attn.wo_b", columns, fp8, {4096U, 8192U}, {32U, 64U}},
        {"layers.2.attn.indexer.wq_b", replicated, fp8, {8192U, 1024U}, {64U, 8U}},
        {"layers.2.attn.compressor.wkv", replicated, bf16, {1024U, 4096U}, no_scale},
        {"layers.2.attn.indexer.compressor.wkv", replicated, bf16, {256U, 4096U}, no_scale},
        {"layers.2.ffn.gate", replicated, bf16, {256U, 4096U}, no_scale},
        {"layers.2.ffn.shared_experts.w1", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.2.ffn.shared_experts.w3", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.2.ffn.shared_experts.w2", columns, fp8, {4096U, 2048U}, {32U, 16U}},
        {"layers.2.ffn.experts.0.w1", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.2.ffn.experts.0.w3", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.2.ffn.experts.0.w2", columns, fp4, {4096U, 1024U}, {4096U, 64U}},
        {"layers.2.ffn.gate.tid2eid", replicated, i64, {129280U, 6U}, no_scale},
        {"layers.2.attn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.2.ffn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.2.hc_attn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"layers.2.hc_attn_scale", replicated, f32, {3U}, no_scale},
        {"layers.2.hc_attn_base", replicated, f32, {24U}, no_scale},
        {"layers.2.hc_ffn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"layers.2.hc_ffn_scale", replicated, f32, {3U}, no_scale},
        {"layers.2.hc_ffn_base", replicated, f32, {24U}, no_scale},
        {"layers.21.attn.wq_a", replicated, fp8, {1024U, 4096U}, {8U, 32U}},
        {"layers.21.attn.wkv", replicated, fp8, {512U, 4096U}, {4U, 32U}},
        {"layers.21.attn.compressor.wkv", replicated, bf16, {512U, 4096U}, no_scale},
        {"layers.21.attn.wq_b", rows, fp8, {32768U, 1024U}, {256U, 8U}},
        {"layers.21.attn.wo_a", rows, fp8, {8192U, 4096U}, {64U, 32U}},
        {"layers.21.attn.wo_b", columns, fp8, {4096U, 8192U}, {32U, 64U}},
        {"layers.21.ffn.gate", replicated, bf16, {256U, 4096U}, no_scale},
        {"layers.21.ffn.gate.bias", replicated, f32, {256U}, no_scale},
        {"layers.21.ffn.shared_experts.w1", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.21.ffn.shared_experts.w3", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.21.ffn.shared_experts.w2", columns, fp8, {4096U, 2048U}, {32U, 16U}},
        {"layers.21.ffn.experts.0.w1", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.21.ffn.experts.0.w3", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.21.ffn.experts.0.w2", columns, fp4, {4096U, 1024U}, {4096U, 64U}},
        {"layers.21.attn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.21.ffn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.21.hc_attn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"layers.21.hc_ffn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"layers.42.attn.wq_a", replicated, fp8, {1024U, 4096U}, {8U, 32U}},
        {"layers.42.attn.wkv", replicated, fp8, {512U, 4096U}, {4U, 32U}},
        {"layers.42.attn.compressor.wkv", replicated, bf16, {1024U, 4096U}, no_scale},
        {"layers.42.attn.indexer.compressor.wkv", replicated, bf16, {256U, 4096U}, no_scale},
        {"layers.42.attn.indexer.wq_b", replicated, fp8, {8192U, 1024U}, {64U, 8U}},
        {"layers.42.attn.wq_b", rows, fp8, {32768U, 1024U}, {256U, 8U}},
        {"layers.42.attn.wo_a", rows, fp8, {8192U, 4096U}, {64U, 32U}},
        {"layers.42.attn.wo_b", columns, fp8, {4096U, 8192U}, {32U, 64U}},
        {"layers.42.ffn.gate", replicated, bf16, {256U, 4096U}, no_scale},
        {"layers.42.ffn.gate.bias", replicated, f32, {256U}, no_scale},
        {"layers.42.ffn.shared_experts.w1", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.42.ffn.shared_experts.w3", rows, fp8, {2048U, 4096U}, {16U, 32U}},
        {"layers.42.ffn.shared_experts.w2", columns, fp8, {4096U, 2048U}, {32U, 16U}},
        {"layers.42.ffn.experts.0.w1", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.42.ffn.experts.0.w3", rows, fp4, {2048U, 2048U}, {2048U, 128U}},
        {"layers.42.ffn.experts.0.w2", columns, fp4, {4096U, 1024U}, {4096U, 64U}},
        {"layers.42.attn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.42.ffn_norm", replicated, bf16, {4096U}, no_scale},
        {"layers.42.hc_attn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"layers.42.hc_ffn_fn", replicated, f32, {24U, 16384U}, no_scale},
        {"norm", replicated, bf16, {4096U}, no_scale},
        {"hc_head_fn", replicated, f32, {4U, 16384U}, no_scale},
        {"hc_head_scale", replicated, f32, {1U}, no_scale},
        {"hc_head_base", replicated, f32, {4U}, no_scale},
    };
    for (const auto& fixture : fixtures) check_fixture(checkpoint, fixture);
}
