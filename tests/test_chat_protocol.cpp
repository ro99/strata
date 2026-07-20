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
    REQUIRE(error.empty());
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
}
