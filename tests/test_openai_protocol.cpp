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
