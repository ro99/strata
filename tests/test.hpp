#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::test {

using TestFunction = void (*)();

struct TestCase {
    std::string name;
    TestFunction function{};
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(std::string name, TestFunction function) {
        registry().push_back(TestCase{std::move(name), function});
    }
};

[[noreturn]] inline void fail(std::string_view expression, std::string_view file, int line) {
    std::ostringstream message;
    message << file << ':' << line << ": assertion failed: " << expression;
    throw std::runtime_error(message.str());
}

}  // namespace strata::test

#define STRATA_TEST_JOIN_INNER(a, b) a##b
#define STRATA_TEST_JOIN(a, b) STRATA_TEST_JOIN_INNER(a, b)
#define TEST_CASE(name)                                                                  \
    static void STRATA_TEST_JOIN(strata_test_, __LINE__)();                              \
    static ::strata::test::Registrar STRATA_TEST_JOIN(strata_registrar_, __LINE__)(      \
        name, &STRATA_TEST_JOIN(strata_test_, __LINE__));                                \
    static void STRATA_TEST_JOIN(strata_test_, __LINE__)()

#define REQUIRE(expression)                                                              \
    do {                                                                                 \
        if (!(expression)) {                                                             \
            ::strata::test::fail(#expression, __FILE__, __LINE__);                       \
        }                                                                                \
    } while (false)

#define REQUIRE_NEAR(actual, expected, epsilon)                                          \
    do {                                                                                 \
        if (std::fabs((actual) - (expected)) > (epsilon)) {                              \
            ::strata::test::fail(#actual " ~= " #expected, __FILE__, __LINE__);          \
        }                                                                                \
    } while (false)
