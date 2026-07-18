#include "strata/tokenizer.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string tokenizer_path;
    std::string prompt;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--tokenizer" && index + 1 < argc) {
            tokenizer_path = argv[++index];
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else {
            std::cerr << "usage: strata-tokenize --tokenizer FILE --prompt TEXT\n";
            return 2;
        }
    }
    if (tokenizer_path.empty() || prompt.empty()) {
        std::cerr << "usage: strata-tokenize --tokenizer FILE --prompt TEXT\n";
        return 2;
    }
    const auto loaded = strata::BpeTokenizer::load(tokenizer_path);
    if (!loaded.ok()) {
        for (const auto& error : loaded.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    const auto rendered = strata::render_glm52_user_prompt(prompt);
    const auto encoded = loaded.value.encode(rendered);
    if (!encoded.ok()) {
        for (const auto& error : encoded.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    std::cout << "rendered=" << rendered << '\n'
              << "token_count=" << encoded.value.size() << '\n'
              << "tokens=";
    for (std::size_t index = 0; index < encoded.value.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << encoded.value[index];
    }
    std::cout << '\n';
    return 0;
}
