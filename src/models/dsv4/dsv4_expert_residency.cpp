#include "strata/dsv4_expert_residency.hpp"

#include <charconv>
#include <fstream>
#include <sstream>

namespace strata {
namespace {

[[nodiscard]] bool parse_u32(std::string_view text, std::uint32_t& out) {
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), out);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

}  // namespace

ParseResult<Dsv4ExpertResidencyPlan> Dsv4ExpertResidencyPlan::parse(
    std::string_view text, std::uint32_t layers, std::uint32_t experts) {
    ParseResult<Dsv4ExpertResidencyPlan> result;
    if (layers == 0U || experts == 0U) {
        result.errors.emplace_back(
            "expert residency plan needs a positive layer and expert count");
        return result;
    }
    std::istringstream stream{std::string(text)};
    std::string line;
    if (!std::getline(stream, line) ||
        line.rfind("strata.dsv4_expert_residency ", 0U) != 0U) {
        result.errors.emplace_back(
            "expert residency plan is missing its schema header");
        return result;
    }
    if (line != "strata.dsv4_expert_residency 1") {
        result.errors.emplace_back(
            "unsupported expert residency plan version: " + line);
        return result;
    }

    // The declared geometry must match the model actually loaded. A plan built
    // for a different layer or expert count would index the wrong triplets and
    // is a hard failure, not something to reconcile.
    std::uint32_t plan_layers = 0U;
    std::uint32_t plan_experts = 0U;
    std::uint64_t declared_pairs = 0U;
    bool saw_geometry = false;
    bool saw_pairs = false;
    while (std::getline(stream, line)) {
        std::istringstream fields{line};
        std::string key;
        fields >> key;
        if (key == "layers") {
            std::string experts_key;
            if (!(fields >> plan_layers >> experts_key >> plan_experts)) {
                result.errors.emplace_back(
                    "expert residency plan has a malformed geometry line");
                return result;
            }
            saw_geometry = true;
        } else if (key == "pairs") {
            if (!(fields >> declared_pairs)) {
                result.errors.emplace_back(
                    "expert residency plan has a malformed pair count");
                return result;
            }
            saw_pairs = true;
            break;
        }
        // Other header lines are diagnostic and intentionally ignored, so the
        // planner can add counters without breaking existing plans.
    }
    if (!saw_geometry || !saw_pairs) {
        result.errors.emplace_back(
            "expert residency plan is missing its geometry or pair count");
        return result;
    }
    if (plan_layers != layers || plan_experts != experts) {
        result.errors.emplace_back(
            "expert residency plan declares " + std::to_string(plan_layers) +
            " layers and " + std::to_string(plan_experts) +
            " experts, model has " + std::to_string(layers) + " and " +
            std::to_string(experts));
        return result;
    }

    Dsv4ExpertResidencyPlan plan;
    plan.layers_ = layers;
    plan.experts_ = experts;
    plan.bitmap_.assign(static_cast<std::size_t>(layers) * experts, 0U);
    plan.pairs_.reserve(static_cast<std::size_t>(declared_pairs));
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        std::istringstream fields{line};
        std::uint32_t layer = 0U;
        std::uint32_t expert = 0U;
        if (!(fields >> layer >> expert)) {
            result.errors.emplace_back(
                "expert residency plan has a malformed entry: " + line);
            return result;
        }
        if (layer >= layers || expert >= experts) {
            result.errors.emplace_back(
                "expert residency plan names layer " + std::to_string(layer) +
                " expert " + std::to_string(expert) + ", outside the model");
            return result;
        }
        auto& bit = plan.bitmap_[static_cast<std::size_t>(layer) * experts + expert];
        if (bit != 0U) {
            result.errors.emplace_back(
                "expert residency plan repeats layer " + std::to_string(layer) +
                " expert " + std::to_string(expert) +
                "; a repeated triplet would be counted twice");
            return result;
        }
        bit = 1U;
        plan.pairs_.emplace_back(layer, expert);
    }
    if (plan.pairs_.size() != declared_pairs) {
        result.errors.emplace_back(
            "expert residency plan declares " + std::to_string(declared_pairs) +
            " pairs but lists " + std::to_string(plan.pairs_.size()));
        return result;
    }
    result.value = std::move(plan);
    return result;
}

ParseResult<Dsv4ExpertResidencyPlan> Dsv4ExpertResidencyPlan::load(
    const std::string& path, std::uint32_t layers, std::uint32_t experts) {
    ParseResult<Dsv4ExpertResidencyPlan> result;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        result.errors.emplace_back(
            "cannot open expert residency plan " + path);
        return result;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if (!file.good() && !file.eof()) {
        result.errors.emplace_back(
            "cannot read expert residency plan " + path);
        return result;
    }
    return parse(buffer.str(), layers, experts);
}

std::size_t Dsv4ExpertResidencyPlan::slice(std::size_t offset,
                                           std::size_t stride) {
    if (stride <= 1U) return pairs_.size();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kept;
    kept.reserve(pairs_.size() / stride + 1U);
    for (std::size_t index = 0U; index < pairs_.size(); ++index) {
        const auto [layer, expert] = pairs_[index];
        if (index % stride == offset % stride) {
            kept.emplace_back(layer, expert);
            continue;
        }
        // Not ours: clear the bit so this tier never claims an expert another
        // tier owns. Exactly one engine must owe each triplet.
        bitmap_[static_cast<std::size_t>(layer) * experts_ + expert] = 0U;
    }
    pairs_ = std::move(kept);
    return pairs_.size();
}

std::size_t Dsv4ExpertResidencyPlan::truncate(std::size_t limit) {
    if (limit >= pairs_.size()) return pairs_.size();
    for (std::size_t index = limit; index < pairs_.size(); ++index) {
        const auto [layer, expert] = pairs_[index];
        bitmap_[static_cast<std::size_t>(layer) * experts_ + expert] = 0U;
    }
    pairs_.resize(limit);
    return pairs_.size();
}

}  // namespace strata
