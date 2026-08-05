#include "test.hpp"

#include "strata/chat_protocol.hpp"
#include "strata/inkling_runtime.hpp"
#include "strata/model_adapter.hpp"
#include "strata/cuda_backend.hpp"
#include "strata/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path inkling_model_path() {
    return std::filesystem::path(STRATA_SOURCE_DIR) / "models/inkling-s";
}

bool inkling_checkpoint_present() {
    return std::filesystem::exists(inkling_model_path() /
                                   "model.safetensors.index.json");
}

}  // namespace

TEST_CASE("Inkling chat template emits the effort block once, before the turn") {
    const std::array messages{
        strata::ChatMessage{strata::ChatRole::User, "hi"}};
    const auto rendered = strata::render_inkling_chat_prompt(messages);
    // The template opens with the thinking-effort system message, then the
    // user turn, and ends primed for the model to speak.
    REQUIRE(rendered ==
            "<|message_system|><|content_text|>Thinking effort level: 0.9"
            "<|end_message|><|message_user|><|content_text|>hi<|end_message|>"
            "<|message_model|>");

    // A leading system message comes first and does not trigger the effort
    // block until a non-system turn appears.
    const std::array with_system{
        strata::ChatMessage{strata::ChatRole::System, "be terse"},
        strata::ChatMessage{strata::ChatRole::User, "hi"}};
    const auto second = strata::render_inkling_chat_prompt(with_system);
    REQUIRE(second.starts_with(
        "<|message_system|><|content_text|>be terse<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.9"));
    // Exactly one effort block, whatever the turn count.
    std::size_t occurrences = 0U;
    for (std::size_t at = second.find("Thinking effort level");
         at != std::string::npos;
         at = second.find("Thinking effort level", at + 1U)) {
        ++occurrences;
    }
    REQUIRE(occurrences == 1U);

    // An assistant turn closes with the end-of-sampling marker.
    const std::array multi{
        strata::ChatMessage{strata::ChatRole::User, "hi"},
        strata::ChatMessage{strata::ChatRole::Assistant, "hello"},
        strata::ChatMessage{strata::ChatRole::User, "again"}};
    const auto third = strata::render_inkling_chat_prompt(multi);
    REQUIRE(third.find("<|message_model|><|content_text|>hello<|end_message|>"
                       "<|content_model_end_sampling|>") != std::string::npos);
}

TEST_CASE("Inkling runtime rejects use before initialization") {
    strata::InklingRuntime runtime;
    const std::array<std::uint32_t, 2> tokens{1U, 2U};
    std::vector<std::vector<float>> logits;
    REQUIRE(!runtime.forward_logits(tokens, logits).ok());
    REQUIRE(!runtime.generate_stream("hello", 1U).ok());
}

TEST_CASE("Inkling runtime generates the expected continuation") {
    if (!inkling_checkpoint_present()) {
        SKIP("pinned Inkling-Small-NVFP4 checkpoint is absent");
    }
    strata::InklingRuntimeConfig config;
    config.maximum_context_tokens = 64U;
    // Faulting 154 GiB into page cache is a production load-time cost, not a
    // test cost; the device path is exercised either way.
    config.warm_expert_pages = false;
    strata::InklingRuntime runtime;
    const auto initialized =
        runtime.initialize(inkling_model_path().string(), config);
    for (const auto& error : initialized.errors) {
        std::cerr << "initialize: " << error << '\n';
    }
    REQUIRE(initialized.ok());

    // Greedy decoding on a fact the model certainly knows. This is the
    // end-to-end gate: an error anywhere in the relative attention, the short
    // convolutions, the expert-sink routing, the NVFP4 dequantization or the
    // muP logit scale turns this into noise rather than a near miss.
    const auto result = runtime.generate_stream("The capital of France is", 2U);
    for (const auto& error : result.errors) {
        std::cerr << "generate: " << error << '\n';
    }
    REQUIRE(result.ok());
    REQUIRE(result.text.find("Paris") != std::string::npos);
    REQUIRE(result.metrics.prompt_tokens > 0U);
    REQUIRE(result.metrics.decode_tokens > 0U);
    // Six of 256 experts per layer: a step must touch far less than the
    // 154 GiB the routed experts occupy on disk.
    REQUIRE(result.metrics.graph.routed_expert_bytes > 0U);
    REQUIRE(result.metrics.graph.routed_expert_bytes <
            static_cast<std::uint64_t>(result.metrics.graph.forward_tokens) *
                8ULL * (1ULL << 30U));
}

TEST_CASE("Inkling teacher forcing is deterministic and ranks the known token") {
    if (!inkling_checkpoint_present()) {
        SKIP("pinned Inkling-Small-NVFP4 checkpoint is absent");
    }
    strata::InklingRuntimeConfig config;
    config.maximum_context_tokens = 64U;
    // Faulting 154 GiB into page cache is a production load-time cost, not a
    // test cost; the device path is exercised either way.
    config.warm_expert_pages = false;
    strata::InklingRuntime runtime;
    REQUIRE(runtime.initialize(inkling_model_path().string(), config).ok());

    auto tokenizer = strata::ModelTokenizer::load(
        (inkling_model_path() / "tokenizer.json").string());
    REQUIRE(tokenizer.ok());
    const auto prompt = tokenizer.value.encode("The capital of France is");
    REQUIRE(prompt.ok());

    std::vector<std::vector<float>> first;
    auto status = runtime.forward_logits(prompt.value, first);
    for (const auto& error : status.errors) {
        std::cerr << "forward: " << error << '\n';
    }
    REQUIRE(status.ok());
    REQUIRE(first.size() == prompt.value.size());
    REQUIRE(first.back().size() ==
            strata::kInklingExecutionContract.vocabulary_size);

    // The argmax at the final position must decode to Paris.
    const auto best = static_cast<std::uint32_t>(std::distance(
        first.back().begin(),
        std::max_element(first.back().begin(), first.back().end())));
    const auto piece = tokenizer.value.decode_token(best);
    REQUIRE(piece.ok());
    REQUIRE(piece.value.find("Paris") != std::string::npos);

    // Re-running the same tokens must reproduce the logits bit for bit. A
    // stale KV row or an unreset convolution history shows up here and nowhere
    // else, because generation would still look plausible.
    std::vector<std::vector<float>> second;
    status = runtime.forward_logits(prompt.value, second);
    REQUIRE(status.ok());
    REQUIRE(second.size() == first.size());
    for (std::size_t row = 0U; row < first.size(); ++row) {
        for (std::size_t index = 0U; index < first[row].size(); ++index) {
            REQUIRE(first[row][index] == second[row][index]);
        }
    }

    // An inverted NVFP4 scale or a lost softmax shift surfaces as a non-finite
    // logit long before it surfaces as bad text.
    for (const auto& value : first.back()) REQUIRE(std::isfinite(value));
}

TEST_CASE("Inkling device logits match the host oracle") {
    if (!inkling_checkpoint_present()) {
        SKIP("pinned Inkling-Small-NVFP4 checkpoint is absent");
    }
    if (strata::CudaBackend::available_devices().empty()) {
        SKIP("no CUDA device is available");
    }
    auto tokenizer = strata::ModelTokenizer::load(
        (inkling_model_path() / "tokenizer.json").string());
    REQUIRE(tokenizer.ok());
    const auto prompt = tokenizer.value.encode("The capital of France is");
    REQUIRE(prompt.ok());

    const auto run = [&](bool cuda, std::vector<std::vector<float>>& logits) {
        strata::InklingRuntimeConfig config;
        config.maximum_context_tokens = 64U;
        config.warm_expert_pages = false;
        config.enable_cuda = cuda;
        strata::InklingRuntime runtime;
        const auto initialized =
            runtime.initialize(inkling_model_path().string(), config);
        for (const auto& error : initialized.errors) {
            std::cerr << "initialize: " << error << '\n';
        }
        REQUIRE(initialized.ok());
        const auto status = runtime.forward_logits(prompt.value, logits);
        for (const auto& error : status.errors) {
            std::cerr << "forward: " << error << '\n';
        }
        REQUIRE(status.ok());
    };

    std::vector<std::vector<float>> host;
    std::vector<std::vector<float>> device;
    run(false, host);
    run(true, device);
    REQUIRE(host.size() == device.size());

    // The device path reassociates every reduction and the NVFP4 kernel
    // accumulates in a different order, so the contract is agreement on the
    // decision and a bounded relative deviation, not bit equality.
    for (std::size_t row = 0U; row < host.size(); ++row) {
        REQUIRE(host[row].size() == device[row].size());
        const auto host_best = static_cast<std::uint32_t>(std::distance(
            host[row].begin(),
            std::max_element(host[row].begin(), host[row].end())));
        const auto device_best = static_cast<std::uint32_t>(std::distance(
            device[row].begin(),
            std::max_element(device[row].begin(), device[row].end())));
        REQUIRE(host_best == device_best);

        double squared = 0.0;
        double reference = 0.0;
        for (std::size_t index = 0U; index < host[row].size(); ++index) {
            REQUIRE(std::isfinite(device[row][index]));
            const double delta = static_cast<double>(device[row][index]) -
                                 host[row][index];
            squared += delta * delta;
            reference += static_cast<double>(host[row][index]) * host[row][index];
        }
        const double relative =
            reference > 0.0 ? std::sqrt(squared / reference) : 0.0;
        if (relative > 2.0e-2) {
            std::cerr << "row " << row << " relative deviation " << relative
                      << '\n';
        }
        REQUIRE(relative <= 2.0e-2);
    }
}
