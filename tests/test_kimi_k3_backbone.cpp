#include "kimi_fixture.hpp"
#include "test.hpp"

#include "strata/kimi_k3_runtime.hpp"
#include "strata/model_adapter.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

// Gates 5 and 6: full-model teacher forcing and greedy generation. The whole
// 93-layer backbone runs in the runtime and is compared against the checkpoint's
// own reference at *every* layer, not only at the end. The fixture comes from
// `scripts/run_kimi_k3_backbone_fixture.sh`, which builds each real
// `KimiDecoderLayer` with real weights, calls it, and frees it — the 1.45 TiB
// model does not fit in torch on this machine and disk offload is forbidden.
//
// Two passes, because prefill and decode are different code on both sides:
// a four-token prompt page (chunkwise KDA, causal MLA) and then one decode step
// against the state the page left (recurrent KDA, MLA against committed rows).
// The decode token is the reference's own greedy choice, so the second pass is a
// generation oracle: the runtime has to pick the same token from its own logits
// and then agree about what that token produces.
//
// `generate_from_tokens` is not re-run here. It is a loop over `evaluate` and
// the shared `sample_logits`, both of which this gate and the sampler's own
// tests already cover; a second pass to re-test the loop would cost another ten
// minutes of SATA for nothing.
//
// Opt-in. The runtime loads a 106.55 GiB dense spine and sweeps the routed
// experts of all 93 layers twice, about ten minutes and roughly 250 GiB of SATA
// reads. `make check` must not carry that, so the gate runs when
// `STRATA_KIMI_BACKBONE=1` is set and is skipped otherwise.
//
// Predicted shape, stated before the run:
//
//  * Per-layer relative L2 sits in the 5e-3 to 2e-2 band — the same bfloat16
//    op-boundary rounding gate 4 measured (5e-3 to 9e-3 for one layer) — and
//    is **flat in depth**, because the attention-residual stream is additive:
//    an error that enters at layer k is carried forward, but the prefix it
//    rides on grows with it, so the *relative* figure should not compound.
//  * A rising trend, or a step at one layer, is a defect to localize, not a
//    tolerance to widen. The one benign step this model can produce is a
//    router boundary flip: 896 experts at top-16, and a token whose 16th and
//    17th scores are within the rounding is free to route differently. That
//    would show as a jump at a single MoE layer that then persists, and the
//    test prints every off-shape layer so it can be told apart from a real bug.
//  * The decode pass should agree at least as well as the prompt pass. It
//    carries the prompt's error in through the cache but adds only one token's
//    worth of its own.
//  * Both argmaxes must match. That is the only part of this gate that is not a
//    tolerance: a backbone that ranks a different token first has not
//    reproduced the model, however small its L2 is.
//
// Gate: every layer of both passes within 3.0e-2 relative L2 and 0.9995 cosine,
// no upward trend across depth, both logit vectors within 3.0e-2, and both
// argmaxes equal to the reference's.

namespace {

using kimi_test::Agreement;
using kimi_test::compare;
using kimi_test::Fixture;
using kimi_test::kimi_directory;
using kimi_test::kimi_present;

constexpr float kRelativeL2Gate = 3.0e-2F;
constexpr float kCosineGate = 0.9995F;
// Not a gate: the band the prediction above says the measurement should fall
// in. Exceeding it inside the gate is reported, because a number that is passing
// but off-shape is still something to explain.
constexpr float kPredictedBand = 2.0e-2F;

std::string fixture_path() {
    return kimi_test::fixture_path("kimi-k3-backbone.fixture");
}

// One pass's per-layer agreement, filled by the runtime's layer observer and
// checked after the pass so the whole depth profile is printed before anything
// fails: which layer diverged is the finding, and stopping at the first would
// hide whether it is a step or a trend.
struct Depth {
    std::vector<Agreement> layers;
    std::vector<bool> seen;
    std::vector<std::string> errors;
    std::string tag;

    explicit Depth(std::string name, std::uint32_t count)
        : layers(count), seen(count, false), tag(std::move(name)) {}

    void observe(const Fixture& fixture, std::uint32_t layer,
                 std::span<const float> hidden) {
        if (layer >= layers.size()) return;
        const auto* reference =
            fixture.find(tag + ".layer." + std::to_string(layer));
        if (reference == nullptr) {
            errors.push_back(tag + " layer " + std::to_string(layer) +
                             " is missing from the fixture");
            return;
        }
        if (reference->values.size() != hidden.size()) {
            errors.push_back(tag + " layer " + std::to_string(layer) +
                             " shape disagrees with the fixture");
            return;
        }
        layers[layer] = compare(hidden, reference->values);
        seen[layer] = true;
    }

    [[nodiscard]] std::uint32_t report() const {
        float worst = 0.0F;
        std::uint32_t worst_layer = 0U;
        std::uint32_t outside_band = 0U;
        std::uint32_t failures = 0U;
        for (std::uint32_t layer = 0U; layer < layers.size(); ++layer) {
            if (!seen[layer]) {
                std::cout << "  [gate 5] " << tag << " layer " << layer
                          << " was never observed\n";
                ++failures;
                continue;
            }
            const auto& agreement = layers[layer];
            if (agreement.relative_l2 > worst) {
                worst = agreement.relative_l2;
                worst_layer = layer;
            }
            const bool off_shape = agreement.relative_l2 > kPredictedBand;
            if (off_shape) ++outside_band;
            if (agreement.relative_l2 >= kRelativeL2Gate ||
                agreement.cosine <= kCosineGate) {
                ++failures;
            }
            if (off_shape || layer % 8U == 0U || layer + 1U == layers.size()) {
                std::cout << "  [gate 5] " << tag << " layer " << layer
                          << " relative L2 " << agreement.relative_l2
                          << ", cosine " << agreement.cosine << '\n';
            }
        }
        std::cout << "  [gate 5] " << tag << " worst layer " << worst_layer
                  << " at " << worst << ", layers outside the predicted "
                  << kPredictedBand << " band: " << outside_band << '\n';
        return failures;
    }

    // Flat in depth, not compounding: the mean over the last eight layers must
    // not exceed twice the mean over the first eight. A backbone whose error
    // grows with depth is a different failure from one that is uniformly noisy,
    // and only the first is fatal to long-context decode.
    [[nodiscard]] bool flat_in_depth() const {
        double head = 0.0;
        double tail = 0.0;
        for (std::uint32_t index = 0U; index < 8U; ++index) {
            head += layers[index].relative_l2;
            tail += layers[layers.size() - 1U - index].relative_l2;
        }
        std::cout << "  [gate 5] " << tag << " mean relative L2, first eight "
                  << (head / 8.0) << ", last eight " << (tail / 8.0) << '\n';
        return tail <= 2.0 * head + 1.0e-6;
    }
};

std::uint32_t argmax(std::span<const float> values) {
    return static_cast<std::uint32_t>(
        std::distance(values.begin(),
                      std::max_element(values.begin(), values.end())));
}

}  // namespace

TEST_CASE("Kimi-K3 backbone matches the reference at every layer") {
    if (!kimi_present()) SKIP("pinned Kimi-K3 checkpoint is absent");
    if (std::getenv("STRATA_KIMI_BACKBONE") == nullptr) {
        SKIP("full-backbone teacher forcing is opt-in: it loads 106.55 GiB and "
             "sweeps the routed experts twice, about ten minutes; set "
             "STRATA_KIMI_BACKBONE=1 to run it");
    }
    Fixture fixture;
    if (!fixture.load(fixture_path())) {
        SKIP("backbone fixture is absent; run "
             "scripts/run_kimi_k3_backbone_fixture.sh");
    }

    const auto* ids = fixture.find("prompt.token_ids");
    const auto* prompt_logits = fixture.find("prompt.last_logits");
    const auto* decode_id = fixture.find("decode.token_id");
    const auto* decode_logits = fixture.find("decode.last_logits");
    REQUIRE(ids != nullptr);
    REQUIRE(prompt_logits != nullptr);
    REQUIRE(decode_id != nullptr);
    REQUIRE(decode_logits != nullptr);
    std::vector<std::uint32_t> tokens;
    tokens.reserve(ids->values.size());
    for (const auto value : ids->values) {
        tokens.push_back(static_cast<std::uint32_t>(value));
    }
    REQUIRE(!tokens.empty());

    const auto& c = strata::kKimiK3ExecutionContract;
    Depth prompt("prompt", c.layer_count);
    Depth decode("decode", c.layer_count);
    Depth* active = &prompt;

    strata::KimiK3RuntimeConfig config;
    config.maximum_context_tokens = 64U;
    config.prefill_page_tokens = 64U;
    config.layer_observer = [&](std::uint32_t layer,
                                std::span<const float> hidden) {
        active->observe(fixture, layer, hidden);
    };

    strata::KimiK3Runtime runtime;
    auto initialized = runtime.initialize(kimi_directory(), config);
    for (const auto& error : initialized.errors) std::cout << "  " << error << '\n';
    REQUIRE(initialized.ok());

    std::vector<float> logits(runtime.vocabulary_size());
    auto step = runtime.evaluate(tokens, 0U, logits);
    for (const auto& error : step.errors) std::cout << "  " << error << '\n';
    REQUIRE(step.ok());

    const auto prompt_failures = prompt.report();
    const auto prompt_agreement = compare(logits, prompt_logits->values);
    std::cout << "  [gate 5] prompt last-token logits relative L2 "
              << prompt_agreement.relative_l2 << ", cosine "
              << prompt_agreement.cosine << '\n';

    // Gate 6: the runtime must pick the token the reference picked, from its own
    // logits, and then be fed that token — exactly what greedy generation does.
    const auto chosen = argmax(logits);
    const auto expected_next = static_cast<std::uint32_t>(decode_id->values.at(0));
    std::cout << "  [gate 6] greedy next token " << chosen << " against "
              << expected_next << '\n';

    active = &decode;
    const std::array<std::uint32_t, 1U> next{expected_next};
    auto second = runtime.evaluate(next, static_cast<std::uint32_t>(tokens.size()),
                                   logits);
    for (const auto& error : second.errors) std::cout << "  " << error << '\n';
    REQUIRE(second.ok());

    const auto decode_failures = decode.report();
    const auto decode_agreement = compare(logits, decode_logits->values);
    std::cout << "  [gate 5] decode logits relative L2 "
              << decode_agreement.relative_l2 << ", cosine "
              << decode_agreement.cosine << '\n';
    const auto decode_choice = argmax(logits);
    const auto reference_choice = argmax(decode_logits->values);
    std::cout << "  [gate 6] second greedy token " << decode_choice
              << " against " << reference_choice << '\n';

    for (const auto& error : prompt.errors) std::cout << "  " << error << '\n';
    for (const auto& error : decode.errors) std::cout << "  " << error << '\n';
    REQUIRE(prompt.errors.empty());
    REQUIRE(decode.errors.empty());
    REQUIRE(prompt_failures == 0U);
    REQUIRE(decode_failures == 0U);
    REQUIRE(prompt.flat_in_depth());
    REQUIRE(decode.flat_in_depth());
    REQUIRE(prompt_agreement.relative_l2 < kRelativeL2Gate);
    REQUIRE(prompt_agreement.cosine > kCosineGate);
    REQUIRE(decode_agreement.relative_l2 < kRelativeL2Gate);
    REQUIRE(decode_agreement.cosine > kCosineGate);
    REQUIRE(chosen == expected_next);
    REQUIRE(decode_choice == reference_choice);
}
