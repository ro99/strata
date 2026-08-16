#include "test.hpp"

#include "strata/deepseek_kv_cache.hpp"
#include "strata/deepseek_ops.hpp"
#include "strata/dsv4_attention_kv.hpp"
#include "strata/model_adapter.hpp"
#include "strata/numerics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

std::vector<float> row(std::size_t width, float value) {
    return std::vector<float>(width, value);
}

void require_bit_equal(std::span<const float> actual,
                       std::span<const float> expected) {
    REQUIRE(actual.size() == expected.size());
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        REQUIRE(std::bit_cast<std::uint32_t>(actual[index]) ==
                std::bit_cast<std::uint32_t>(expected[index]));
    }
}

}  // namespace

TEST_CASE("DeepSeek physical KV layout is block-major and admitted") {
    const auto main = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Sliding);
    REQUIRE(main.ok());
    REQUIRE(main.value.semantic_width == 512U);
    REQUIRE(main.value.block_rows == 256U);
    REQUIRE(main.value.token_data_bytes == 576U);
    REQUIRE(main.value.token_scale_bytes == 8U);
    REQUIRE(main.value.block_bytes == 149504U);
    REQUIRE(main.value.format ==
            strata::Dsv4PhysicalKvFormat::
                Fp8E4m3Group64Bf16RopeUe8m0);

    const auto index = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex);
    REQUIRE(index.ok());
    REQUIRE(index.value.semantic_width == 128U);
    REQUIRE(index.value.token_data_bytes == 128U);
    REQUIRE(index.value.token_scale_bytes == 4U);
    REQUIRE(index.value.block_bytes == 33792U);
    REQUIRE(index.value.format ==
            strata::Dsv4PhysicalKvFormat::Fp8E4m3PerTensorF32Scale);
    const auto compressed4 = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Sliding, 64U);
    const auto compressed128 = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Sliding, 2U);
    const auto compressed_index = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, 64U);
    REQUIRE(compressed4.ok());
    REQUIRE(compressed4.value.block_bytes == 37376U);
    REQUIRE(compressed128.ok());
    REQUIRE(compressed128.value.block_bytes == 1168U);
    REQUIRE(compressed_index.ok());
    REQUIRE(compressed_index.value.block_bytes == 8448U);
    REQUIRE(!strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, 2U).ok());
    REQUIRE(!strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Sliding, 32U).ok());

    constexpr std::uint32_t compressed_row = 63U;
    std::vector<std::byte> compressed_block(
        static_cast<std::size_t>(compressed4.value.block_bytes));
    const auto compressed_data =
        static_cast<std::size_t>(compressed_row) * 576U;
    const auto compressed_scales = 64U * 576U +
        static_cast<std::size_t>(compressed_row) * 8U;
    std::fill_n(compressed_block.begin() +
                    static_cast<std::ptrdiff_t>(compressed_data),
                448U, std::byte{0x38U});
    std::fill_n(compressed_block.begin() +
                    static_cast<std::ptrdiff_t>(compressed_scales),
                7U, std::byte{127U});
    std::vector<float> decoded_compressed(512U);
    REQUIRE(strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::Csa, compressed_block,
        compressed_row, decoded_compressed).ok());
    REQUIRE(decoded_compressed[0] == 1.0F);

    constexpr std::uint32_t row_index = 255U;
    std::vector<std::byte> main_block(
        static_cast<std::size_t>(main.value.block_bytes));
    const auto main_data = static_cast<std::size_t>(row_index) * 576U;
    const auto main_scales = 256U * 576U +
                             static_cast<std::size_t>(row_index) * 8U;
    for (std::size_t group = 0U; group < 7U; ++group) {
        main_block[main_scales + group] =
            static_cast<std::byte>(123U + group);
        for (std::size_t column = 0U; column < 64U; ++column) {
            main_block[main_data + group * 64U + column] = std::byte{0x38U};
        }
    }
    for (std::size_t column = 0U; column < 64U; ++column) {
        const auto encoded = strata::bf16_encode(
            static_cast<float>(column) / 64.0F);
        std::memcpy(main_block.data() + main_data + 448U +
                        column * sizeof(encoded),
                    &encoded, sizeof(encoded));
    }
    std::vector<float> decoded_main(512U);
    REQUIRE(strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, main_block,
        row_index, decoded_main).ok());
    for (std::size_t group = 0U; group < 7U; ++group) {
        const auto expected = strata::bf16_round_f32(
            strata::dsv4_fp8_e8m0_scale_f32(
                static_cast<std::uint8_t>(123U + group)));
        REQUIRE(decoded_main[group * 64U] == expected);
    }
    REQUIRE(decoded_main[448U + 63U] ==
            strata::bf16_round_f32(63.0F / 64.0F));
    main_block[main_scales + 7U] = std::byte{1U};
    REQUIRE(!strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, main_block,
        row_index, decoded_main).ok());

    std::vector<std::byte> index_block(
        static_cast<std::size_t>(index.value.block_bytes));
    const auto index_data = static_cast<std::size_t>(row_index) * 128U;
    const auto index_scale = 256U * 128U +
                             static_cast<std::size_t>(row_index) * 4U;
    std::fill_n(index_block.begin() +
                    static_cast<std::ptrdiff_t>(index_data),
                128U, std::byte{0x38U});
    constexpr float scale = 0.125F;
    std::memcpy(index_block.data() + index_scale, &scale, sizeof(scale));
    std::vector<float> decoded_index(128U);
    REQUIRE(strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_block,
        row_index, decoded_index).ok());
    REQUIRE(std::all_of(decoded_index.begin(), decoded_index.end(),
                        [](float value) { return value == 0.125F; }));

    const auto admission = strata::dsv4_physical_kv_admission(32768U);
    REQUIRE(admission.ok());
    REQUIRE(admission.value.sliding_bytes == 12857344U);
    REQUIRE(admission.value.compressed_bytes == 103456768U);
    REQUIRE(admission.value.index_bytes == 22708224U);
    REQUIRE(admission.value.payload_bytes == 139022336U);
    REQUIRE(admission.value.compressor_state_bytes == 11862016U);
    REQUIRE(admission.value.index_state_bytes == 344064U);
    REQUIRE(admission.value.total_bytes == 151228416U);
    // Physical allocator identities cover 256 source tokens, so compressed
    // pages contain 256 / ratio rows. Count those real cache blocks rather
    // than 256-row accounting aggregates.
    REQUIRE(admission.value.allocated_blocks == 8022U);

    const auto short_admission =
        strata::dsv4_physical_kv_admission(4096U);
    REQUIRE(short_admission.ok());
    REQUIRE(short_admission.value.sliding_bytes == 12'857'344ULL);
    REQUIRE(short_admission.value.compressed_bytes == 12'932'096ULL);
    REQUIRE(short_admission.value.index_bytes == 2'838'528ULL);
    REQUIRE(short_admission.value.payload_bytes == 28'627'968ULL);
}

TEST_CASE("DeepSeek physical KV encoder writes accepted physical planes") {
    const auto main_layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::Csa, 64U);
    REQUIRE(main_layout.ok());
    std::vector<std::byte> main_page(
        static_cast<std::size_t>(main_layout.value.block_bytes));
    std::vector<float> main_values(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    main_values[0] = -0.0F;
    for (std::size_t column = 448U; column < main_values.size(); ++column) {
        main_values[column] = strata::bf16_round_f32(
            static_cast<float>(static_cast<int>(column) - 480) / 64.0F);
    }
    constexpr std::uint32_t main_row = 63U;
    REQUIRE(strata::dsv4_physical_encode_kv_row(
        strata::Dsv4KvBlockKind::Csa, main_values,
        main_row, main_page).ok());
    const auto main_data = static_cast<std::size_t>(main_row) * 576U;
    const auto main_scales = 64U * 576U +
                             static_cast<std::size_t>(main_row) * 8U;
    REQUIRE(main_page[main_data] == std::byte{0U});
    REQUIRE(main_page[main_data + 1U] == std::byte{0x78U});
    for (std::size_t group = 0U; group < 7U; ++group) {
        REQUIRE(main_page[main_scales + group] == std::byte{119U});
    }
    REQUIRE(main_page[main_scales + 7U] == std::byte{0U});
    std::vector<float> decoded_main(main_values.size());
    REQUIRE(strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::Csa, main_page,
        main_row, decoded_main).ok());
    REQUIRE(decoded_main[0] == 0.0F);
    REQUIRE(!std::signbit(decoded_main[0]));
    require_bit_equal(std::span<const float>(decoded_main).subspan(1U),
                      std::span<const float>(main_values).subspan(1U));

    const auto index_layout = strata::dsv4_physical_kv_layout(
        strata::Dsv4KvBlockKind::LearnedIndex, 64U);
    REQUIRE(index_layout.ok());
    std::vector<std::byte> index_page(
        static_cast<std::size_t>(index_layout.value.block_bytes));
    std::vector<float> index_values(
        strata::kDeepSeekV4ExecutionContract.index_head_dim, 1.0F);
    REQUIRE(strata::dsv4_physical_encode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_values,
        main_row, index_page).ok());
    const auto index_data = static_cast<std::size_t>(main_row) * 128U;
    const auto index_scale = 64U * 128U +
                             static_cast<std::size_t>(main_row) * 4U;
    REQUIRE(index_page[index_data] == std::byte{0x78U});
    float stored_scale = 0.0F;
    std::memcpy(&stored_scale, index_page.data() + index_scale,
                sizeof(stored_scale));
    REQUIRE(stored_scale == std::ldexp(1.0F, -8));
    std::vector<float> decoded_index(index_values.size());
    REQUIRE(strata::dsv4_physical_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_page,
        main_row, decoded_index).ok());
    require_bit_equal(decoded_index, index_values);

    std::array<float, 4> query{-0.0F, 1.0F, -2.0F, 500.0F};
    REQUIRE(strata::dsv4_physical_quantize_query_e4m3_f32(query).ok());
    REQUIRE(query[0] == 0.0F && !std::signbit(query[0]));
    REQUIRE(query[1] == 1.0F);
    REQUIRE(query[2] == -2.0F);
    REQUIRE(query[3] == 448.0F);
}

TEST_CASE("DeepSeek physical prefill page keeps its earliest causal window") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = strata::kDsv4PhysicalKvBlockRows;
    config.sliding_window_rows = 128U;
    config.host_capacity_bytes = 1U << 20U;
    config.physical_layout = true;
    strata::Dsv4KvCache cache(config);
    REQUIRE(cache.validate().ok());
    const auto sequence = cache.create_sequence();
    REQUIRE(sequence.ok());
    const auto values = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);

    for (std::uint64_t position = 0U; position < 128U; ++position) {
        REQUIRE(cache.append(sequence.value,
                             strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values).ok());
    }
    // A page beginning at position 128 attends row 128 against rows 1..128,
    // even though appending the entire page would ordinarily advance the
    // sliding floor to 64 before the first attend call.
    for (std::uint64_t position = 128U; position < 192U; ++position) {
        REQUIRE(cache.append(sequence.value,
                             strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values, 1U).ok());
    }
    REQUIRE(!cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 0U).ok());
    REQUIRE(cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 1U).ok());
    REQUIRE(cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 128U).ok());

    // The next ordinary append advances the floor to the normal 128-row
    // window; the page-scoped retention does not leak into decode semantics.
    REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 192U, values).ok());
    REQUIRE(!cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 1U).ok());
    REQUIRE(cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 65U).ok());
}

TEST_CASE("DeepSeek live cache retains physical page boundaries") {
    strata::Dsv4KvCacheConfig invalid;
    invalid.physical_layout = true;
    invalid.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache rejected(invalid);
    REQUIRE(!rejected.validate().ok());

    strata::Dsv4KvCacheConfig config;
    config.block_rows = strata::kDsv4PhysicalKvBlockRows;
    config.sliding_window_rows = 512U;
    config.host_capacity_bytes = 1U << 20U;
    config.physical_layout = true;
    strata::Dsv4KvCache cache(config);
    REQUIRE(cache.validate().ok());
    const auto sequence = cache.create_sequence();
    REQUIRE(sequence.ok());
    const auto main = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    const auto index = row(
        strata::kDeepSeekV4ExecutionContract.index_head_dim, 1.0F);
    for (std::uint64_t position = 0U; position < 257U; ++position) {
        REQUIRE(cache.append(sequence.value,
                             strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, main).ok());
    }
    for (std::uint64_t position = 0U; position < 65U; ++position) {
        REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Csa,
                             1U, 4U, position, main).ok());
        REQUIRE(cache.append(sequence.value,
                             strata::Dsv4KvBlockKind::LearnedIndex,
                             1U, 4U, position, index).ok());
    }
    for (std::uint64_t position = 0U; position < 3U; ++position) {
        REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Hca,
                             2U, 128U, position, main).ok());
    }
    const auto sliding = cache.block_table(
        sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U);
    const auto csa = cache.block_table(
        sequence.value, strata::Dsv4KvBlockKind::Csa, 1U);
    const auto hca = cache.block_table(
        sequence.value, strata::Dsv4KvBlockKind::Hca, 2U);
    const auto learned = cache.block_table(
        sequence.value, strata::Dsv4KvBlockKind::LearnedIndex, 1U);
    REQUIRE(sliding.ok() && sliding.value.size() == 2U);
    REQUIRE(csa.ok() && csa.value.size() == 2U);
    REQUIRE(hca.ok() && hca.value.size() == 2U);
    REQUIRE(learned.ok() && learned.value.size() == 2U);
    REQUIRE(sliding.value[0].capacity_rows == 256U);
    REQUIRE(csa.value[0].capacity_rows == 64U);
    REQUIRE(hca.value[0].capacity_rows == 2U);
    REQUIRE(learned.value[0].capacity_rows == 64U);
    REQUIRE(sliding.value[0].payload_bytes == 149504U);
    REQUIRE(csa.value[0].payload_bytes == 37376U);
    REQUIRE(hca.value[0].payload_bytes == 1168U);
    REQUIRE(learned.value[0].payload_bytes == 8448U);
    REQUIRE(sliding.value[0].format ==
            strata::Dsv4KvFormat::PhysicalFp8E4m3Group64Bf16Rope);
    REQUIRE(learned.value[0].format ==
            strata::Dsv4KvFormat::PhysicalFp8E4m3PerTensor);
    const auto last = cache.row(
        sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U, 256U);
    REQUIRE(last.ok());
    require_bit_equal(last.value, main);
    REQUIRE(!cache.learned_index_segments(
        sequence.value, 1U, 65U).ok());
}

TEST_CASE("DeepSeek compact KV codecs are exact or reject the write") {
    constexpr auto head_dim = strata::kDeepSeekV4ExecutionContract.head_dim;
    constexpr auto rope_dim =
        strata::kDeepSeekV4ExecutionContract.rope_head_dim;
    std::vector<float> kv(head_dim);
    for (std::size_t group = 0U; group < (head_dim - rope_dim) / 64U; ++group) {
        const float scale = std::ldexp(1.0F, static_cast<int>(group) - 3);
        for (std::size_t column = 0U; column < 64U; ++column) {
            const auto code = column == 63U ? std::uint8_t{0x7EU}
                : static_cast<std::uint8_t>(column + 1U);
            kv[group * 64U + column] =
                strata::dsv4_fp8_e4m3_f32(code) * scale;
        }
    }
    for (std::size_t column = 0U; column < rope_dim; ++column) {
        kv[head_dim - rope_dim + column] = std::ldexp(
            static_cast<float>(static_cast<int>(column % 7U) - 3), -4);
    }
    const auto kv_format = strata::dsv4_kv_format(
        strata::Dsv4KvBlockKind::Sliding);
    std::vector<std::byte> encoded_kv(static_cast<std::size_t>(
        strata::dsv4_kv_row_bytes(strata::Dsv4KvBlockKind::Sliding,
                                  kv_format)));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format, kv, encoded_kv).ok());
    std::vector<float> decoded_kv(head_dim);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        encoded_kv, decoded_kv).ok());
    require_bit_equal(decoded_kv, kv);

    auto lossy = kv;
    lossy[0] = 1.1F;
    REQUIRE(!strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        lossy, encoded_kv).ok());
    encoded_kv[head_dim - rope_dim] = std::byte{0xFFU};
    REQUIRE(!strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, kv_format,
        encoded_kv, decoded_kv).ok());

    constexpr auto index_dim =
        strata::kDeepSeekV4ExecutionContract.index_head_dim;
    std::vector<float> index(index_dim);
    for (std::size_t group = 0U; group < index_dim / 32U; ++group) {
        const float scale = std::ldexp(1.0F, static_cast<int>(group) - 2);
        for (std::size_t column = 0U; column < 32U; ++column) {
            const auto code = column == 31U ? std::uint8_t{7U}
                : static_cast<std::uint8_t>(column % 8U);
            index[group * 32U + column] =
                strata::dsv4_fp4_e2m1_f32(code) * scale;
        }
    }
    const auto index_format = strata::dsv4_kv_format(
        strata::Dsv4KvBlockKind::LearnedIndex);
    std::vector<std::byte> encoded_index(static_cast<std::size_t>(
        strata::dsv4_kv_row_bytes(strata::Dsv4KvBlockKind::LearnedIndex,
                                  index_format)));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        index, encoded_index).ok());
    std::vector<float> decoded_index(index_dim);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        encoded_index, decoded_index).ok());
    require_bit_equal(decoded_index, index);
    encoded_index[index_dim / 2U] = std::byte{0xFFU};
    REQUIRE(!strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex, index_format,
        encoded_index, decoded_index).ok());

    std::vector<std::byte> oracle(kv.size() * sizeof(float));
    REQUIRE(strata::dsv4_encode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, strata::Dsv4KvFormat::F32,
        lossy, oracle).ok());
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::Sliding, strata::Dsv4KvFormat::F32,
        oracle, decoded_kv).ok());
    require_bit_equal(decoded_kv, lossy);
}

TEST_CASE("DeepSeek KV cache keeps typed block tables and masks stale rows") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    REQUIRE(cache.validate().ok());
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    REQUIRE(cache.stats().host_used_bytes == 0U);
    REQUIRE(cache.stats().used_blocks == 0U);

    const auto kv = row(strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    for (std::uint64_t position = 0U; position < 5U; ++position) {
        REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, kv).ok());
    }
    REQUIRE(!cache.row(created.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 0U).ok());
    REQUIRE(cache.row(created.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 1U).ok());

    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Csa,
                         1U, 4U, 0U, kv).ok());
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Hca,
                         2U, 128U, 0U, kv).ok());
    const auto index = row(
        strata::kDeepSeekV4ExecutionContract.index_head_dim, 2.0F);
    REQUIRE(cache.append(created.value,
                         strata::Dsv4KvBlockKind::LearnedIndex,
                         1U, 4U, 0U, index).ok());

    const auto csa = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::Csa, 1U);
    const auto hca = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::Hca, 2U);
    const auto learned = cache.block_table(
        created.value, strata::Dsv4KvBlockKind::LearnedIndex, 1U);
    REQUIRE(csa.ok() && csa.value.size() == 1U);
    REQUIRE(hca.ok() && hca.value.size() == 1U);
    REQUIRE(learned.ok() && learned.value.size() == 1U);
    REQUIRE(csa.value[0].compression_ratio == 4U);
    REQUIRE(hca.value[0].compression_ratio == 128U);
    REQUIRE(learned.value[0].row_width ==
            strata::kDeepSeekV4ExecutionContract.index_head_dim);
    REQUIRE(csa.value[0].format ==
            strata::Dsv4KvFormat::Fp8E4m3Group64Bf16Rope);
    REQUIRE(hca.value[0].format ==
            strata::Dsv4KvFormat::Fp8E4m3Group64Bf16Rope);
    REQUIRE(learned.value[0].format ==
            strata::Dsv4KvFormat::Fp4E2m1Group32);
    REQUIRE(learned.value[0].format_version == strata::kDsv4KvFormatVersion);
    REQUIRE(learned.value[0].physical_bytes > learned.value[0].payload_bytes);
    const auto compact = cache.learned_index_segments(
        created.value, 1U, 1U);
    REQUIRE(compact.ok() && compact.value.size() == 1U);
    REQUIRE(compact.value[0].rows == 1U);
    REQUIRE(compact.value[0].device_buffer == nullptr);
    std::vector<float> compact_decoded(index.size());
    const auto compact_row_bytes = strata::dsv4_kv_row_bytes(
        strata::Dsv4KvBlockKind::LearnedIndex,
        strata::Dsv4KvFormat::Fp4E2m1Group32);
    REQUIRE(strata::dsv4_decode_kv_row(
        strata::Dsv4KvBlockKind::LearnedIndex,
        strata::Dsv4KvFormat::Fp4E2m1Group32,
        compact.value[0].host_bytes.subspan(
            compact.value[0].byte_offset, compact_row_bytes),
        compact_decoded).ok());
    require_bit_equal(compact_decoded, index);

    REQUIRE(cache.truncate_sequence(created.value, 4U).ok());
    REQUIRE(!cache.row(created.value, strata::Dsv4KvBlockKind::Hca,
                       2U, 0U).ok());
    REQUIRE(cache.reset_sequence(created.value).ok());
    REQUIRE(cache.stats().used_blocks == 0U);
    REQUIRE(cache.release_sequence(created.value).ok());
    REQUIRE(!cache.release_sequence(created.value).ok());
}

TEST_CASE("DeepSeek KV forks copy partial blocks only on write") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 4U;
    config.sliding_window_rows = 8U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    const auto original = cache.create_sequence();
    REQUIRE(original.ok());
    const auto first = row(strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 0U, first).ok());
    const auto fork = cache.fork_sequence(original.value);
    REQUIRE(fork.ok());
    const auto shared = cache.block_table(
        original.value, strata::Dsv4KvBlockKind::Sliding, 0U);
    REQUIRE(shared.ok() && shared.value.size() == 1U);
    REQUIRE(shared.value[0].owner_sequence == original.value);
    REQUIRE(shared.value[0].refcount == 2U);

    const auto original_second = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 2.0F);
    const auto fork_second = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 3.0F);
    REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 1U, original_second).ok());
    REQUIRE(cache.append(fork.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 1U, fork_second).ok());
    const auto from_original = cache.row(
        original.value, strata::Dsv4KvBlockKind::Sliding, 0U, 1U);
    const auto from_fork = cache.row(
        fork.value, strata::Dsv4KvBlockKind::Sliding, 0U, 1U);
    REQUIRE(from_original.ok() && from_original.value.front() == 2.0F);
    REQUIRE(from_fork.ok() && from_fork.value.front() == 3.0F);
    REQUIRE(cache.stats().copy_on_write_blocks == 1U);

    REQUIRE(cache.release_sequence(original.value).ok());
    REQUIRE(cache.stats().used_blocks == 1U);
    REQUIRE(cache.release_sequence(fork.value).ok());
    REQUIRE(cache.stats().used_blocks == 0U);
}

TEST_CASE("DeepSeek KV truncation cannot expose stale shared rows") {
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 4U;
    config.sliding_window_rows = 8U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    const auto original = cache.create_sequence();
    REQUIRE(original.ok());
    for (std::uint64_t position = 0U; position < 4U; ++position) {
        const auto values = row(
            strata::kDeepSeekV4ExecutionContract.head_dim,
            static_cast<float>(position));
        REQUIRE(cache.append(original.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values).ok());
    }
    const auto fork = cache.fork_sequence(original.value);
    REQUIRE(fork.ok());
    REQUIRE(cache.truncate_sequence(fork.value, 2U).ok());
    REQUIRE(!cache.row(fork.value, strata::Dsv4KvBlockKind::Sliding,
                       0U, 2U).ok());
    const auto replacement = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 9.0F);
    REQUIRE(cache.append(fork.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 2U, replacement).ok());
    REQUIRE(cache.row(fork.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 2U).value.front() == 9.0F);
    REQUIRE(cache.row(original.value, strata::Dsv4KvBlockKind::Sliding,
                      0U, 2U).value.front() == 2.0F);
}

TEST_CASE("DeepSeek KV host allocation fails at its own ceiling") {
    const auto block_bytes = strata::dsv4_kv_block_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding), 2U);
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = block_bytes;
    strata::Dsv4KvCache cache(config);
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    const auto values = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 0U, values).ok());
    REQUIRE(!cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                          1U, 1U, 0U, values).ok());
    REQUIRE(cache.stats().host_used_bytes == block_bytes);
    REQUIRE(cache.stats().host_write_bytes == strata::dsv4_kv_row_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding)));
    REQUIRE(cache.stats().used_blocks == 1U);
}

TEST_CASE("DeepSeek KV device eviction protects in-flight blocks") {
    const auto devices = strata::CudaBackend::available_devices();
    if (!strata::CudaBackend::compiled() || devices.empty()) return;
    strata::CudaBackend backend;
    const std::array<int, 1> selected{devices.front()};
    REQUIRE(backend.initialize(selected, false).ok());

    const auto block_bytes = strata::dsv4_kv_block_bytes(
        strata::Dsv4KvBlockKind::Sliding,
        strata::dsv4_kv_format(strata::Dsv4KvBlockKind::Sliding), 2U);
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 2U;
    config.sliding_window_rows = 4U;
    config.host_capacity_bytes = block_bytes * 2U;
    config.devices = {devices.front()};
    config.device_capacity_bytes = {block_bytes};
    strata::Dsv4KvCache cache(config, &backend);
    const auto created = cache.create_sequence();
    REQUIRE(created.ok());
    const auto values = row(
        strata::kDeepSeekV4ExecutionContract.head_dim, 1.0F);
    for (std::uint64_t position = 0U; position < 3U; ++position) {
        REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position, values).ok());
    }

    auto first = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 0U, 0U);
    REQUIRE(first.ok() && first.value.valid());
    REQUIRE(!cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U).ok());
    first.value = {};
    auto second = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U);
    REQUIRE(second.ok() && second.value.valid());
    second.value = {};
    REQUIRE(cache.append(created.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, 3U, values).ok());
    auto refreshed = cache.acquire_device(
        created.value, strata::Dsv4KvBlockKind::Sliding, 0U, 2U, 0U);
    REQUIRE(refreshed.ok() && refreshed.value.valid());
    REQUIRE(cache.stats().promotions == 3U);
    REQUIRE(cache.stats().evictions == 1U);
    REQUIRE(backend.stats().activation_h2d_bytes == block_bytes * 3U);
    REQUIRE(backend.stats().synchronization_calls == 3U);
}

TEST_CASE("DeepSeek KV truncation restores the state a lookahead started from") {
    // The future-entropy lookahead decodes one speculative token per candidate
    // and then rolls the sequence back. What it rolls back to has to be
    // bit-identical to never having decoded it, because the token the sampler
    // goes on to emit is decoded from exactly this state.
    constexpr auto head_dim = strata::kDeepSeekV4ExecutionContract.head_dim;
    strata::Dsv4KvCacheConfig config;
    config.block_rows = 4U;
    config.sliding_window_rows = 16U;
    config.host_capacity_bytes = 1U << 20U;
    strata::Dsv4KvCache cache(config);
    const auto sequence = cache.create_sequence();
    REQUIRE(sequence.ok());

    // Six accepted tokens, spanning a block boundary so the rollback has to
    // release a block rather than only move an end marker.
    constexpr std::uint64_t accepted = 6U;
    for (std::uint64_t position = 0U; position < accepted; ++position) {
        REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, position,
                             row(head_dim, static_cast<float>(position + 1U))).ok());
    }
    const auto accepted_blocks = cache.stats().used_blocks;
    const auto before = cache.block_table(
        sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U);
    REQUIRE(before.ok());

    // Three candidates, each decoded at the same position and then undone.
    for (std::uint32_t candidate = 0U; candidate < 3U; ++candidate) {
        REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                             0U, 1U, accepted,
                             row(head_dim, -static_cast<float>(candidate + 1U))).ok());
        REQUIRE(cache.truncate_sequence(sequence.value, accepted).ok());

        const auto after = cache.block_table(
            sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U);
        REQUIRE(after.ok());
        REQUIRE(after.value.size() == before.value.size());
        REQUIRE(cache.stats().used_blocks == accepted_blocks);
        // The speculative row is gone, not merely hidden behind an end marker.
        REQUIRE(!cache.row(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                           0U, accepted).ok());
        // Every accepted row still reads back exactly as it was written.
        for (std::uint64_t position = 0U; position < accepted; ++position) {
            const auto stored = cache.row(
                sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U, position);
            REQUIRE(stored.ok());
            require_bit_equal(stored.value,
                              row(head_dim, static_cast<float>(position + 1U)));
        }
    }

    // And the sequence still accepts the real token at the position the
    // speculative ones occupied, with no contiguity complaint.
    REQUIRE(cache.append(sequence.value, strata::Dsv4KvBlockKind::Sliding,
                         0U, 1U, accepted, row(head_dim, 12.0F)).ok());
    const auto emitted = cache.row(
        sequence.value, strata::Dsv4KvBlockKind::Sliding, 0U, accepted);
    REQUIRE(emitted.ok());
    require_bit_equal(emitted.value, row(head_dim, 12.0F));
}
