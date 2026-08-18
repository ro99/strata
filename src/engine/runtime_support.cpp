#include "strata/runtime_support.hpp"

#include "strata/cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <unordered_set>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace strata {

void StopSequenceBuffer::emit(std::size_t end, std::uint32_t token,
                              const TokenStreamCallback& callback) {
    if (end <= emitted_) return;
    if (callback &&
        !callback(token, std::string_view(text_).substr(emitted_, end - emitted_))) {
        cancelled_ = true;
    }
    emitted_ = end;
}

void StopSequenceBuffer::append(std::uint32_t token, std::string_view piece,
                                const TokenStreamCallback& callback) {
    if (stopped_ || cancelled_) return;
    text_.append(piece);
    std::size_t stop_at = std::string::npos;
    for (const auto& stop : stops_) {
        const auto found = text_.find(stop, emitted_);
        if (!stop.empty() && found < stop_at) stop_at = found;
    }
    if (stop_at != std::string::npos) {
        emit(stop_at, token, callback);
        text_.resize(stop_at);
        stopped_ = true;
        return;
    }
    std::size_t held = 0U;
    for (const auto& stop : stops_) {
        const auto maximum = std::min(stop.size() - (stop.empty() ? 0U : 1U),
                                      text_.size());
        for (std::size_t length = maximum; length > held; --length) {
            if (text_.compare(text_.size() - length, length, stop, 0U, length) == 0) {
                held = length;
                break;
            }
        }
    }
    emit(text_.size() - held, token, callback);
}

void StopSequenceBuffer::finish(const TokenStreamCallback& callback) {
    if (!stopped_ && !cancelled_) emit(text_.size(), 0U, callback);
}

ValidationResult validate_common_runtime_config(
    std::span<const int> devices, double vram_fraction,
    double sampling_temperature, std::string_view model_label) {
    ValidationResult result;
    if (devices.empty()) {
        result.errors.emplace_back(std::string(model_label) +
                                   " runtime requires at least one CUDA device");
    }
    std::unordered_set<int> unique;
    for (const int device : devices) {
        if (device < 0 || !unique.insert(device).second) {
            result.errors.emplace_back(std::string(model_label) +
                                       " runtime devices must be unique non-negative ids");
            break;
        }
    }
    if (!std::isfinite(vram_fraction) || vram_fraction <= 0.0 ||
        vram_fraction > 0.95) {
        result.errors.emplace_back("VRAM cache fraction must be in (0, 0.95]");
    }
    if (!std::isfinite(sampling_temperature) || sampling_temperature < 0.0 ||
        sampling_temperature > 10.0) {
        result.errors.emplace_back(std::string(model_label) +
                                   " sampling temperature must be within [0, 10]");
    }
    return result;
}

RuntimeDevicePlanResult plan_runtime_devices(
    std::span<const int> devices, double vram_fraction,
    std::uint64_t per_device_workspace_reserve,
    std::uint64_t minimum_device_budget,
    std::string_view model_label,
    std::uint64_t explicit_device_budget_bytes) {
    RuntimeDevicePlanResult result;
    std::vector<std::uint64_t> totals;
    result.value.devices.assign(devices.begin(), devices.end());
    for (const int device : devices) {
        auto memory = CudaBackend::device_memory(device);
        if (!memory.ok()) {
            result.errors = std::move(memory.errors);
            return result;
        }
        const auto computed = compute_runtime_device_budget(
            memory.value.free_bytes, vram_fraction,
            explicit_device_budget_bytes);
        const auto budget = computed.applied_budget_bytes;
        if (budget < minimum_device_budget ||
            budget <= per_device_workspace_reserve) {
            result.errors.emplace_back(
                std::string(model_label) + " CUDA device " +
                std::to_string(device) +
                " does not meet the admitted VRAM budget");
            return result;
        }
        result.value.budgets.push_back(budget);
        result.value.fractional_budgets.push_back(
            computed.fractional_budget_bytes);
        result.value.explicit_budgets.push_back(
            computed.explicit_budget_bytes);
        result.value.weight_capacities.push_back(
            budget - per_device_workspace_reserve);
        totals.push_back(memory.value.total_bytes);
    }
    const auto smallest = *std::min_element(totals.begin(), totals.end());
    if (smallest == 0U) {
        result.errors.emplace_back(std::string(model_label) +
                                   " CUDA device reports zero total memory");
        return result;
    }
    for (std::size_t slot = 0U; slot < totals.size(); ++slot) {
        if (totals[slot] > (std::numeric_limits<std::uint64_t>::max() -
                            smallest / 2U) / 2U) {
            result.errors.emplace_back("CUDA device schedule weight overflows");
            return result;
        }
        const auto shares = std::max<std::uint64_t>(
            1U, (totals[slot] * 2U + smallest / 2U) / smallest);
        for (std::uint64_t count = 0U; count < shares; ++count) {
            result.value.weighted_schedule.push_back(slot);
        }
    }
    return result;
}

RuntimeDeviceBudget compute_runtime_device_budget(
    std::uint64_t free_bytes, double vram_fraction,
    std::uint64_t explicit_budget_bytes) noexcept {
    RuntimeDeviceBudget result;
    // Keep the historical double product so the byte budget remains stable
    // with the pre-existing fractional admission calculation.  The widened
    // comparison below prevents an out-of-range integer conversion.
    const double scaled = static_cast<double>(free_bytes) * vram_fraction;
    if (std::isfinite(vram_fraction) && vram_fraction > 0.0 &&
        std::isfinite(scaled) && scaled > 0.0) {
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        result.fractional_budget_bytes = static_cast<long double>(scaled) >=
                static_cast<long double>(maximum)
            ? maximum
            : static_cast<std::uint64_t>(scaled);
    }
    result.explicit_budget_bytes = explicit_budget_bytes;
    result.applied_budget_bytes = result.fractional_budget_bytes;
    if (explicit_budget_bytes != 0U &&
        explicit_budget_bytes < result.applied_budget_bytes) {
        result.applied_budget_bytes = explicit_budget_bytes;
    }
    return result;
}

std::string_view runtime_budget_bound_name(
    std::uint64_t fractional_budget_bytes,
    std::uint64_t explicit_budget_bytes) noexcept {
    if (explicit_budget_bytes != 0U &&
        explicit_budget_bytes < fractional_budget_bytes) {
        return "explicit";
    }
    if (explicit_budget_bytes != 0U &&
        explicit_budget_bytes == fractional_budget_bytes) {
        return "equal";
    }
    return "fractional";
}

std::uint64_t process_resident_set_bytes() noexcept {
#if defined(__linux__)
    std::ifstream input("/proc/self/statm");
    std::uint64_t virtual_pages = 0U;
    std::uint64_t resident_pages = 0U;
    input >> virtual_pages >> resident_pages;
    static_cast<void>(virtual_pages);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (!input || page_bytes <= 0 ||
        resident_pages > std::numeric_limits<std::uint64_t>::max() /
                             static_cast<std::uint64_t>(page_bytes)) return 0U;
    return resident_pages * static_cast<std::uint64_t>(page_bytes);
#else
    return 0U;
#endif
}

std::vector<std::uint64_t> device_vram_used_bytes(
    std::span<const int> devices) {
    std::vector<std::uint64_t> result;
    result.reserve(devices.size());
    for (const int device : devices) {
        const auto memory = CudaBackend::device_memory(device);
        result.push_back(memory.ok() && memory.value.total_bytes >=
                                           memory.value.free_bytes
                             ? memory.value.total_bytes - memory.value.free_bytes
                             : 0U);
    }
    return result;
}

std::size_t incremental_kv_prefix_tokens(
    std::span<const std::uint32_t> cached_tokens,
    std::span<const std::uint32_t> prompt_tokens) noexcept {
    if (cached_tokens.empty() || cached_tokens.size() >= prompt_tokens.size() ||
        !std::equal(cached_tokens.begin(), cached_tokens.end(),
                    prompt_tokens.begin())) {
        return 0U;
    }
    return cached_tokens.size();
}

bool trim_oldest_chat_turn(std::vector<ChatMessage>& messages) {
    const auto first = std::find_if(messages.begin(), messages.end(),
        [](const ChatMessage& message) { return message.role != ChatRole::System; });
    if (first == messages.end()) return false;
    const auto next = std::find_if(std::next(first), messages.end(),
        [](const ChatMessage& message) { return message.role == ChatRole::User; });
    if (next == messages.end()) return false;
    messages.erase(first, next);
    return true;
}

std::size_t complete_utf8_prefix(std::string_view text) noexcept {
    std::size_t offset = 0U;
    while (offset < text.size()) {
        const auto lead = static_cast<unsigned char>(text[offset]);
        std::size_t length = 0U;
        if (lead <= 0x7FU) length = 1U;
        else if (lead >= 0xC2U && lead <= 0xDFU) length = 2U;
        else if (lead >= 0xE0U && lead <= 0xEFU) length = 3U;
        else if (lead >= 0xF0U && lead <= 0xF4U) length = 4U;
        else return std::string_view::npos;
        if (text.size() - offset < length) return offset;
        for (std::size_t index = 1U; index < length; ++index) {
            if ((static_cast<unsigned char>(text[offset + index]) & 0xC0U) != 0x80U) {
                return std::string_view::npos;
            }
        }
        const auto second = length == 1U
            ? 0U : static_cast<unsigned char>(text[offset + 1U]);
        if ((lead == 0xE0U && second < 0xA0U) ||
            (lead == 0xEDU && second > 0x9FU) ||
            (lead == 0xF0U && second < 0x90U) ||
            (lead == 0xF4U && second > 0x8FU)) {
            return std::string_view::npos;
        }
        offset += length;
    }
    return offset;
}


ChatPrompt prepare_chat_prompt(const ChatPromptRequest& request) {
    ChatPrompt prompt;
    if (request.render == nullptr || request.encode == nullptr) {
        prompt.errors.emplace_back(
            "chat prompt preparation needs a template and a tokenizer");
        return prompt;
    }
    std::vector<ChatMessage> active(request.messages.begin(),
                                   request.messages.end());
    for (;;) {
        auto encoded = request.encode(request.render(active));
        if (!encoded.ok()) {
            prompt.errors = std::move(encoded.errors);
            return prompt;
        }
        if (encoded.value.size() + request.maximum_new_tokens <=
            request.maximum_context_tokens) {
            prompt.token_ids = std::move(encoded.value);
            return prompt;
        }
        if (!request.trim_oldest_turn_to_fit || !trim_oldest_chat_turn(active)) {
            prompt.errors.emplace_back(
                "prompt and requested generation exceed the context ceiling");
            return prompt;
        }
    }
}

}  // namespace strata
