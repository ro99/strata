#include "test.hpp"

#include "strata/app/openai_protocol.hpp"

#include <string>

TEST_CASE("OpenAI chat requests preserve messages and generation controls") {
    strata::OpenAiChatRequest request;
    std::string error;
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"system","content":"brief"},{"role":"user","name":"alice","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}],"temperature":0.7,"top_p":0.8,"n":2,"stream":true,"stop":["END"],"max_tokens":42,"presence_penalty":0.2,"frequency_penalty":-0.3,"logit_bias":{"7":4.5},"seed":9,"user":"u","response_format":{"type":"text"},"logprobs":true,"top_logprobs":3})",
        request, error));
    REQUIRE(error.empty());
    REQUIRE(request.model == "local");
    REQUIRE(request.messages.size() == 2U);
    REQUIRE(request.messages[0].role == strata::ChatRole::System);
    REQUIRE(request.messages[1].content == "hello world");
    REQUIRE(request.messages[1].name == "alice");
    REQUIRE(request.generation.maximum_new_tokens == 42U);
    REQUIRE(request.generation.sampling.temperature == 0.7);
    REQUIRE(request.generation.sampling.top_p == 0.8);
    REQUIRE(request.has_temperature);
    REQUIRE(request.has_top_p);
    REQUIRE(request.generation.sampling.seed == 9U);
    REQUIRE(request.generation.sampling.logit_bias[0].first == 7U);
    REQUIRE(request.generation.sampling.top_logprobs == 3U);
    REQUIRE(request.generation.stop == std::vector<std::string>{"END"});
    REQUIRE(request.n == 2U);
    REQUIRE(request.stream);
    REQUIRE(request.logprobs);
}

TEST_CASE("OpenAI router extraction preserves the original body contract") {
    std::string model;
    std::string error;
    REQUIRE(strata::parse_openai_model_field(
        R"({"messages":[],"model":"creative-large","vendor":{"x":1}})",
        model, error));
    REQUIRE(model == "creative-large");
    REQUIRE(!strata::parse_openai_model_field(
        R"({"model":"a","model":"b"})", model, error));
    REQUIRE(error.find("exactly once") != std::string::npos);
}

TEST_CASE("sampler extensions reach both OpenAI endpoints") {
    const std::string knobs =
        R"("top_k":40,"min_p":0.05,"typical_p":0.9,"xtc_probability":0.5,)"
        R"("xtc_threshold":0.15,"repetition_penalty":1.1,"penalty_window":256,)"
        R"("dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":3,)"
        R"("dry_window":512,"no_repeat_ngram":4,)"
        R"("future_entropy_candidates":20,"future_entropy_top_n":30,)"
        R"("alpha":0.5,"future_entropy_curve":"crossfade",)"
        R"("alpha_wave_amplitude":0.6,"alpha_wave_period":50)";
    const auto check = [](const strata::SamplingOptions& sampling) {
        REQUIRE(sampling.top_k == 40U);
        REQUIRE(sampling.min_p == 0.05);
        REQUIRE(sampling.typical_p == 0.9);
        REQUIRE(sampling.xtc_probability == 0.5);
        REQUIRE(sampling.xtc_threshold == 0.15);
        REQUIRE(sampling.repetition_penalty == 1.1);
        REQUIRE(sampling.penalty_window == 256U);
        REQUIRE(sampling.dry_multiplier == 0.8);
        REQUIRE(sampling.dry_allowed_length == 3U);
        REQUIRE(sampling.dry_window == 512U);
        REQUIRE(sampling.no_repeat_ngram == 4U);
        REQUIRE(sampling.future_entropy_candidates == 20U);
        REQUIRE(sampling.future_entropy_top_n == 30U);
        REQUIRE(sampling.future_entropy_alpha == 0.5);
        REQUIRE(sampling.future_entropy_curve ==
                strata::FutureEntropyCurve::Crossfade);
        REQUIRE(sampling.future_entropy_wave_amplitude == 0.6);
        REQUIRE(sampling.future_entropy_wave_period == 50.0);
    };

    strata::OpenAiChatRequest chat;
    std::string error;
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],)" + knobs + "}",
        chat, error));
    check(chat.generation.sampling);

    strata::OpenAiChatRequest completion;
    REQUIRE(strata::parse_openai_completion_request(
        R"({"model":"local","prompt":"x",)" + knobs + "}", completion, error));
    check(completion.generation.sampling);

    // The shared validator gates the extensions on both endpoints too.
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],"min_p":1.5})",
        chat, error));
    REQUIRE(error.find("min_p") != std::string::npos);
    REQUIRE(!strata::parse_openai_completion_request(
        R"({"model":"local","prompt":"x","dry_base":0.5})", completion, error));
    REQUIRE(error.find("dry_base") != std::string::npos);
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],)"
        R"("alpha":2.0})", chat, error));
    REQUIRE(error.find("future_entropy_alpha") != std::string::npos);
    // An unknown curve must be named, not silently taken as the default: the
    // two curves are different samplers at every alpha but the endpoints.
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],)"
        R"("future_entropy_curve":"balanced"})", chat, error));
    REQUIRE(error.find("future_entropy_curve") != std::string::npos);
}

TEST_CASE("OpenAI protocol rejects silent semantic fallbacks") {
    strata::OpenAiChatRequest request;
    std::string error;
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})",
        request, error));
    REQUIRE(error.find("image") != std::string::npos);
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],"top_logprobs":2})",
        request, error));
    REQUIRE(error.find("logprobs") != std::string::npos);
    REQUIRE(!strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"x"}],"temperature":3})",
        request, error));
}

TEST_CASE("OpenAI protocol preserves base64 image parts for Gemma 4") {
    strata::OpenAiChatRequest request;
    std::string error;
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw=="}}]}]})",
        request, error));
    REQUIRE(request.messages.front().content == "look");
    REQUIRE(request.messages.front().parts.size() == 2U);
    REQUIRE(request.messages.front().parts.back().kind ==
            strata::ChatContentKind::Image);
    REQUIRE(request.messages.front().parts.back().data.size() == 4U);
}

TEST_CASE("OpenAI usage and timings render llama-swap-compatible metrics") {
    const auto usage = strata::render_openai_usage(120U, 40U, 80U);
    REQUIRE(strata::is_json_object(usage));
    REQUIRE(usage.find("\"prompt_tokens\":120") != std::string::npos);
    REQUIRE(usage.find("\"completion_tokens\":40") != std::string::npos);
    REQUIRE(usage.find("\"total_tokens\":160") != std::string::npos);
    REQUIRE(usage.find("\"cached_tokens\":80") != std::string::npos);

    strata::GenerationMetrics metrics;
    metrics.prompt_tokens = 120U;
    metrics.prefill_tokens = 40U;
    metrics.reused_prompt_tokens = 80U;
    metrics.decode_tokens = 30U;
    metrics.prefill_seconds = 2.0;
    metrics.decode_seconds = 3.0;
    const auto timings = strata::render_openai_timings(metrics);
    REQUIRE(strata::is_json_object(timings));
    REQUIRE(timings.find("\"prompt_n\":40") != std::string::npos);
    REQUIRE(timings.find("\"cache_n\":80") != std::string::npos);
    REQUIRE(timings.find("\"predicted_n\":30") != std::string::npos);
    REQUIRE(timings.find("\"prompt_per_second\":20") != std::string::npos);
    REQUIRE(timings.find("\"predicted_per_second\":10") != std::string::npos);
    REQUIRE(timings.find("\"prompt_per_token_ms\":50") != std::string::npos);
    REQUIRE(timings.find("\"predicted_per_token_ms\":100") != std::string::npos);
}

TEST_CASE("usage and timings are finite when no phase ran") {
    const auto timings = strata::render_openai_timings(strata::GenerationMetrics{});
    REQUIRE(strata::is_json_object(timings));
    REQUIRE(timings.find("nan") == std::string::npos);
    REQUIRE(timings.find("inf") == std::string::npos);
    REQUIRE(timings.find("\"prompt_n\":0") != std::string::npos);
    REQUIRE(timings.find("\"prompt_per_second\":0") != std::string::npos);
}

TEST_CASE("legacy completions and tokenize requests parse") {
    strata::OpenAiChatRequest completion;
    std::string error;
    REQUIRE(strata::parse_openai_completion_request(
        R"({"model":"local","prompt":"hello","max_tokens":7,"seed":4})",
        completion, error));
    REQUIRE(completion.messages.size() == 1U);
    REQUIRE(completion.messages[0].content == "hello");
    REQUIRE(completion.generation.maximum_new_tokens == 7U);

    std::string model;
    std::string text;
    REQUIRE(strata::parse_openai_tokenize_request(
        R"({"model":"local","text":"hello"})", model, text, error));
    REQUIRE(model == "local");
    REQUIRE(text == "hello");
    REQUIRE(strata::is_json_object("{\"ok\":true}"));
    REQUIRE(!strata::is_json_object("[1,2]"));
}

namespace {

// GLM-5.3's shape: the prompt already opened the block, so generation starts
// inside it and carries only the closing tag.
constexpr strata::ReasoningFormat kGlm53{"<think>", "</think>", true,
                                         "low,high,max"};

}  // namespace

TEST_CASE("reasoning splits at the closing tag when the prompt opened it") {
    const auto whole = strata::split_reasoning(
        kGlm53, "weighing it up</think>The answer is 555.");
    REQUIRE(whole.reasoning == "weighing it up");
    REQUIRE(whole.content == "The answer is 555.");
}

TEST_CASE("reasoning splitter holds a tag that straddles two pieces") {
    // The tokenizer puts the boundary wherever it likes, so the closing tag
    // arrives in fragments. Nothing may be emitted as content until the tag is
    // resolved, and nothing may be lost.
    strata::ReasoningSplitter splitter(kGlm53);
    std::string reasoning;
    std::string content;
    for (const auto piece : {"think", "ing</", "thi", "nk>ans", "wer"}) {
        const auto delta = splitter.consume(piece);
        reasoning += delta.reasoning;
        content += delta.content;
    }
    const auto tail = splitter.finish();
    reasoning += tail.reasoning;
    content += tail.content;
    REQUIRE(reasoning == "thinking");
    REQUIRE(content == "answer");
}

TEST_CASE("reasoning splitter releases a stream that ended mid-tag") {
    // Truncation inside the scratchpad is the documented failure when --max-new
    // is too small to reach </think>. The held bytes are real output.
    strata::ReasoningSplitter splitter(kGlm53);
    auto delta = splitter.consume("still thinking</thin");
    REQUIRE(delta.reasoning == "still thinking");
    REQUIRE(delta.content.empty());
    const auto tail = splitter.finish();
    REQUIRE(tail.reasoning == "</thin");
    REQUIRE(tail.content.empty());
    REQUIRE(splitter.reasoning_open());
}

TEST_CASE("reasoning splitter stops scanning after the block closes") {
    // One block per generation: an answer may quote a tag's literal text
    // without being re-parsed as reasoning.
    const auto whole = strata::split_reasoning(
        kGlm53, "brief</think>write <think> to open a block");
    REQUIRE(whole.reasoning == "brief");
    REQUIRE(whole.content == "write <think> to open a block");
}

TEST_CASE("an unannotated model passes its whole output through as content") {
    const auto whole = strata::split_reasoning(
        strata::ReasoningFormat{}, "plain <think> output</think>");
    REQUIRE(whole.reasoning.empty());
    REQUIRE(whole.content == "plain <think> output</think>");
}

TEST_CASE("a model that emits its own opening tag is split on both") {
    constexpr strata::ReasoningFormat self_opening{"<think>", "</think>", false,
                                                   ""};
    const auto whole = strata::split_reasoning(
        self_opening, "lead<think>why</think>answer");
    REQUIRE(whole.reasoning == "why");
    REQUIRE(whole.content == "leadanswer");
}

TEST_CASE("chat requests carry a reasoning budget and reasoning visibility") {
    strata::OpenAiChatRequest request;
    std::string error;
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"hi"}],)"
        R"("reasoning_effort":"low","include_reasoning":false})",
        request, error));
    REQUIRE(error.empty());
    REQUIRE(request.generation.reasoning_effort == "low");
    REQUIRE(!request.include_reasoning);
}

TEST_CASE("chat_template_kwargs supplies the budget vLLM clients already send") {
    strata::OpenAiChatRequest request;
    std::string error;
    // A client that also talks to vLLM sends kwargs meant for other models.
    // The ones this server cannot honour are skipped, not rejected.
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"hi"}],)"
        R"("chat_template_kwargs":{"enable_thinking":false,)"
        R"("reasoning_effort":"high","clear_thinking":true}})",
        request, error));
    REQUIRE(error.empty());
    REQUIRE(request.generation.reasoning_effort == "high");
    REQUIRE(request.include_reasoning);
}

TEST_CASE("a request naming no budget leaves the model's own default") {
    strata::OpenAiChatRequest request;
    std::string error;
    REQUIRE(strata::parse_openai_chat_request(
        R"({"model":"local","messages":[{"role":"user","content":"hi"}]})",
        request, error));
    REQUIRE(request.generation.reasoning_effort.empty());
    REQUIRE(request.include_reasoning);
}
