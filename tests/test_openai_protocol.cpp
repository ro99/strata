#include "test.hpp"

#include "strata/openai_protocol.hpp"

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
    REQUIRE(request.generation.sampling.seed == 9U);
    REQUIRE(request.generation.sampling.logit_bias[0].first == 7U);
    REQUIRE(request.generation.sampling.top_logprobs == 3U);
    REQUIRE(request.generation.stop == std::vector<std::string>{"END"});
    REQUIRE(request.n == 2U);
    REQUIRE(request.stream);
    REQUIRE(request.logprobs);
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
