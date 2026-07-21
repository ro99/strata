#include "test.hpp"

#include "strata/chat_protocol.hpp"

#include <string>

TEST_CASE("chat protocol parses prompt records and extensible fields") {
    strata::ChatRequest request;
    std::string error;
    REQUIRE(strata::parse_chat_request(
        R"({"version":1,"command":"prompt","text":"hello\n\u2603","future":true})",
        request, error));
    REQUIRE(request.prompt == "hello\n\xE2\x98\x83");
    REQUIRE(request.messages.size() == 1U);
    REQUIRE(request.messages[0].role == strata::ChatRole::User);
    REQUIRE(!request.includes_history);
    REQUIRE(error.empty());
}

TEST_CASE("chat protocol parses a complete alternating conversation") {
    strata::ChatRequest request;
    std::string error;
    REQUIRE(strata::parse_chat_request(
        R"({"command":"prompt","text":"And its population?","messages":[{"role":"user","content":"Capital of France?"},{"role":"assistant","content":"Paris"},{"role":"user","content":"And its population?"}]})",
        request, error));
    REQUIRE(request.includes_history);
    REQUIRE(request.messages.size() == 3U);
    REQUIRE(request.messages[1].role == strata::ChatRole::Assistant);
    REQUIRE(request.messages[1].content == "Paris");
    REQUIRE(request.prompt == "And its population?");
}

TEST_CASE("chat protocol rejects malformed and ambiguous prompt records") {
    strata::ChatRequest request;
    std::string error;
    REQUIRE(!strata::parse_chat_request(
        R"({"command":"prompt","text":"one","text":"two"})", request, error));
    REQUIRE(error.find("duplicate") != std::string::npos);

    REQUIRE(!strata::parse_chat_request(
        R"({"command":"shutdown","text":"hello"})", request, error));
    REQUIRE(error.find("prompt") != std::string::npos);

    REQUIRE(!strata::parse_chat_request(
        R"({"command":"prompt","text":""})", request, error));
    REQUIRE(error.find("empty") != std::string::npos);

    REQUIRE(!strata::parse_chat_request("not json", request, error));
    REQUIRE(!error.empty());

    REQUIRE(!strata::parse_chat_request(
        R"({"command":"prompt","messages":[{"role":"assistant","content":"orphan"},{"role":"user","content":"hello"}]})",
        request, error));
    REQUIRE(error.find("alternate") != std::string::npos);

    REQUIRE(!strata::parse_chat_request(
        R"({"command":"prompt","text":"different","messages":[{"role":"user","content":"hello"}]})",
        request, error));
    REQUIRE(error.find("match") != std::string::npos);
}
