#include "test.hpp"

int main() {
    int failures = 0;
    std::size_t skipped = 0U;
    for (const auto& test : strata::test::registry()) {
        try {
            test.function();
            std::cout << "PASS " << test.name << '\n';
        } catch (const strata::test::Skip& skip) {
            ++skipped;
            std::cout << "SKIP " << test.name << ": " << skip.what() << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": unknown exception\n";
        }
    }
    std::cout << strata::test::registry().size() -
                     static_cast<std::size_t>(failures) - skipped
              << '/' << strata::test::registry().size() << " tests passed, "
              << skipped << " skipped\n";
    return failures == 0 ? 0 : 1;
}
