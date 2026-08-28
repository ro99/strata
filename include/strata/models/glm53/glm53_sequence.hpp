#pragma once

#include "strata/platform/result.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace strata {

// 45 target layers plus the checkpoint's exact next-token prediction layer.
inline constexpr std::uint32_t kGlm53LayerCount = 46U;
inline constexpr std::uint32_t kGlm53KdaHeads = 64U;
inline constexpr std::uint32_t kGlm53KdaHeadWidth = 128U;
inline constexpr std::uint32_t kGlm53KdaWidth =
    kGlm53KdaHeads * kGlm53KdaHeadWidth;
inline constexpr std::uint32_t kGlm53MlaRank = 512U;

// Physical, copy-on-write rows used by the sparse-MLA cache. A logical
// sequence owns a page table; prefix snapshots share immutable pages and only
// the page receiving the next append is copied. State is F32 exactly as the
// reference runtime -- this is paging, not cache quantization.
class Glm53PagedRows {
public:
    Glm53PagedRows() = default;
    Glm53PagedRows(std::uint32_t columns, std::uint32_t page_rows);

    [[nodiscard]] ValidationResult reset(
        std::uint32_t columns, std::uint32_t page_rows);
    [[nodiscard]] ValidationResult append(std::span<const float> row);
    [[nodiscard]] ValidationResult append_rows(
        std::span<const float> rows, std::uint32_t row_count);
    [[nodiscard]] ValidationResult truncate(std::uint32_t rows);
    [[nodiscard]] std::span<const float> row(std::uint32_t index) const noexcept;
    [[nodiscard]] std::vector<float> materialize() const;

    [[nodiscard]] std::uint32_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::uint32_t page_rows() const noexcept { return page_rows_; }
    [[nodiscard]] std::uint32_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t physical_pages() const noexcept {
        return pages_.size();
    }
    [[nodiscard]] std::uint64_t private_bytes() const noexcept;

private:
    struct Page {
        explicit Page(std::size_t elements) : values(elements) {}
        std::vector<float> values;
    };

    [[nodiscard]] ValidationResult ensure_append_page();

    std::uint32_t columns_{};
    std::uint32_t page_rows_{};
    std::uint32_t rows_{};
    std::vector<std::shared_ptr<Page>> pages_;
};

// All mutable state for one GLM text sequence. Large KDA matrices and short
// convolution histories are allocated lazily and copied on first write after
// a prefix fork. Sparse MLA uses the physical page table above.
class Glm53SequenceState {
public:
    Glm53SequenceState() = default;

    [[nodiscard]] ValidationResult reset(
        std::uint32_t maximum_context_tokens,
        std::uint32_t mla_page_rows = 64U);
    [[nodiscard]] std::span<float> recurrent(std::uint32_t layer);
    [[nodiscard]] std::span<float> convolution(
        std::uint32_t layer, std::uint32_t projection);
    [[nodiscard]] Glm53PagedRows& mla(std::uint32_t layer);
    [[nodiscard]] const Glm53PagedRows& mla(std::uint32_t layer) const;
    void copy_mla_from(std::uint32_t layer,
                       const Glm53SequenceState& source);

    [[nodiscard]] std::uint32_t token_count() const noexcept {
        return token_count_;
    }
    void set_token_count(std::uint32_t value) noexcept { token_count_ = value; }
    [[nodiscard]] std::uint32_t maximum_context_tokens() const noexcept {
        return maximum_context_tokens_;
    }
    [[nodiscard]] std::uint64_t private_bytes() const noexcept;

private:
    using Buffer = std::shared_ptr<std::vector<float>>;
    [[nodiscard]] static std::span<float> writable(
        Buffer& buffer, std::size_t elements);

    std::array<Buffer, kGlm53LayerCount> recurrent_{};
    std::array<std::array<Buffer, 3U>, kGlm53LayerCount> convolution_{};
    std::array<Glm53PagedRows, kGlm53LayerCount> mla_{};
    std::uint32_t maximum_context_tokens_{};
    std::uint32_t mla_page_rows_{64U};
    std::uint32_t token_count_{};
};

}  // namespace strata
