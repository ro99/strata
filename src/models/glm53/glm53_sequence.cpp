#include "strata/models/glm53/glm53_sequence.hpp"

#include "strata/models/glm53/glm53_manifest.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace strata {

Glm53PagedRows::Glm53PagedRows(std::uint32_t columns,
                               std::uint32_t page_rows) {
    static_cast<void>(reset(columns, page_rows));
}

ValidationResult Glm53PagedRows::reset(std::uint32_t columns,
                                       std::uint32_t page_rows) {
    if (columns == 0U || page_rows == 0U ||
        static_cast<std::uint64_t>(columns) * page_rows >
            std::numeric_limits<std::size_t>::max()) {
        return {{"GLM-5.3 physical page geometry is invalid"}};
    }
    columns_ = columns;
    page_rows_ = page_rows;
    rows_ = 0U;
    pages_.clear();
    return {};
}

ValidationResult Glm53PagedRows::ensure_append_page() {
    if (columns_ == 0U || page_rows_ == 0U) {
        return {{"GLM-5.3 physical page table is not initialized"}};
    }
    const auto page_index = rows_ / page_rows_;
    try {
        if (page_index == pages_.size()) {
            pages_.push_back(std::make_shared<Page>(
                static_cast<std::size_t>(columns_) * page_rows_));
        } else if (!pages_[page_index].unique()) {
            pages_[page_index] = std::make_shared<Page>(*pages_[page_index]);
        }
    } catch (const std::bad_alloc&) {
        return {{"GLM-5.3 could not allocate a physical MLA page"}};
    }
    return {};
}

ValidationResult Glm53PagedRows::append(std::span<const float> row_values) {
    if (row_values.size() != columns_) {
        return {{"GLM-5.3 physical MLA row has an invalid width"}};
    }
    auto result = ensure_append_page();
    if (!result.ok()) return result;
    const auto page_index = rows_ / page_rows_;
    const auto page_row = rows_ % page_rows_;
    std::copy(row_values.begin(), row_values.end(),
              pages_[page_index]->values.begin() +
                  static_cast<std::ptrdiff_t>(page_row * columns_));
    ++rows_;
    return {};
}

ValidationResult Glm53PagedRows::append_rows(std::span<const float> values,
                                             std::uint32_t row_count) {
    if (values.size() != static_cast<std::size_t>(row_count) * columns_) {
        return {{"GLM-5.3 physical MLA page append has an invalid shape: "
                 "values=" + std::to_string(values.size()) +
                 " rows=" + std::to_string(row_count) +
                 " columns=" + std::to_string(columns_)}};
    }
    for (std::uint32_t row_index = 0U; row_index < row_count; ++row_index) {
        auto result = append(values.subspan(
            static_cast<std::size_t>(row_index) * columns_, columns_));
        if (!result.ok()) return result;
    }
    return {};
}

ValidationResult Glm53PagedRows::truncate(std::uint32_t rows) {
    if (rows > rows_) return {{"GLM-5.3 physical MLA truncate grows state"}};
    rows_ = rows;
    const auto keep = rows == 0U ? 0U : (rows + page_rows_ - 1U) / page_rows_;
    pages_.resize(keep);
    return {};
}

std::span<const float> Glm53PagedRows::row(std::uint32_t index) const noexcept {
    if (index >= rows_ || columns_ == 0U || page_rows_ == 0U) return {};
    const auto page_index = index / page_rows_;
    const auto page_row = index % page_rows_;
    return std::span<const float>(pages_[page_index]->values)
        .subspan(static_cast<std::size_t>(page_row) * columns_, columns_);
}

std::vector<float> Glm53PagedRows::materialize() const {
    std::vector<float> result(static_cast<std::size_t>(rows_) * columns_);
    for (std::uint32_t index = 0U; index < rows_; ++index) {
        const auto source = row(index);
        std::copy(source.begin(), source.end(),
                  result.begin() + static_cast<std::ptrdiff_t>(
                      static_cast<std::size_t>(index) * columns_));
    }
    return result;
}

std::uint64_t Glm53PagedRows::private_bytes() const noexcept {
    std::uint64_t result = 0U;
    for (const auto& page : pages_) {
        if (page.unique()) result += page->values.size() * sizeof(float);
    }
    return result;
}

ValidationResult Glm53SequenceState::reset(
    std::uint32_t maximum_context_tokens, std::uint32_t mla_page_rows) {
    if (maximum_context_tokens == 0U || mla_page_rows == 0U) {
        return {{"GLM-5.3 sequence geometry is invalid"}};
    }
    maximum_context_tokens_ = maximum_context_tokens;
    mla_page_rows_ = std::min(maximum_context_tokens, mla_page_rows);
    token_count_ = 0U;
    recurrent_.fill({});
    for (auto& layer : convolution_) layer.fill({});
    for (std::uint32_t layer = 0U; layer < kGlm53LayerCount; ++layer) {
        auto result = mla_[layer].reset(kGlm53MlaRank, mla_page_rows_);
        if (!result.ok()) return result;
        // 128 indexer key channels followed by 128 k-pool gate channels.
        result = indexer_[layer].reset(256U, mla_page_rows_);
        if (!result.ok()) return result;
    }
    return {};
}

std::span<float> Glm53SequenceState::writable(Buffer& buffer,
                                              std::size_t elements) {
    if (buffer == nullptr) {
        buffer = std::make_shared<std::vector<float>>(elements, 0.0F);
    } else if (!buffer.unique()) {
        buffer = std::make_shared<std::vector<float>>(*buffer);
    }
    return *buffer;
}

std::span<float> Glm53SequenceState::recurrent(std::uint32_t layer) {
    if (layer >= kGlm53LayerCount || !glm53_kda_layer(layer)) return {};
    return writable(recurrent_[layer],
        static_cast<std::size_t>(kGlm53KdaHeads) * kGlm53KdaHeadWidth *
            kGlm53KdaHeadWidth);
}

std::span<const float> Glm53SequenceState::recurrent(
    std::uint32_t layer) const noexcept {
    if (layer >= kGlm53LayerCount || !glm53_kda_layer(layer) ||
        recurrent_[layer] == nullptr) {
        return {};
    }
    return *recurrent_[layer];
}

std::span<float> Glm53SequenceState::convolution(
    std::uint32_t layer, std::uint32_t projection) {
    if (layer >= kGlm53LayerCount || projection >= 3U ||
        !glm53_kda_layer(layer)) return {};
    return writable(convolution_[layer][projection],
                    static_cast<std::size_t>(kGlm53KdaWidth) * 3U);
}

std::span<const float> Glm53SequenceState::convolution(
    std::uint32_t layer, std::uint32_t projection) const noexcept {
    if (layer >= kGlm53LayerCount || projection >= 3U ||
        !glm53_kda_layer(layer) || convolution_[layer][projection] == nullptr) {
        return {};
    }
    return *convolution_[layer][projection];
}

Glm53PagedRows& Glm53SequenceState::mla(std::uint32_t layer) {
    return mla_.at(layer);
}

Glm53PagedRows& Glm53SequenceState::indexer(std::uint32_t layer) {
    return indexer_.at(layer);
}

const Glm53PagedRows& Glm53SequenceState::mla(std::uint32_t layer) const {
    return mla_.at(layer);
}

const Glm53PagedRows& Glm53SequenceState::indexer(std::uint32_t layer) const {
    return indexer_.at(layer);
}

void Glm53SequenceState::copy_mla_from(
    std::uint32_t layer, const Glm53SequenceState& source) {
    mla_.at(layer) = source.mla_.at(layer);
}

std::uint64_t Glm53SequenceState::private_bytes() const noexcept {
    std::uint64_t result = 0U;
    for (const auto& buffer : recurrent_) {
        if (buffer != nullptr && buffer.unique())
            result += buffer->size() * sizeof(float);
    }
    for (const auto& layer : convolution_) {
        for (const auto& buffer : layer) {
            if (buffer != nullptr && buffer.unique())
                result += buffer->size() * sizeof(float);
        }
    }
    for (const auto& pages : mla_) result += pages.private_bytes();
    for (const auto& pages : indexer_) result += pages.private_bytes();
    return result;
}

}  // namespace strata
