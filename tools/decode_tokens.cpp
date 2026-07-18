#include "strata/tokenizer.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: decode_tokens TOKENIZER_PATH TOKEN_ID [TOKEN_ID ...]\n";
        return 1;
    }
    const auto loaded = strata::BpeTokenizer::load(argv[1]);
    if (!loaded.ok()) {
        for (const auto& error : loaded.errors) std::cerr << "error: " << error << '\n';
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        std::uint32_t id = static_cast<std::uint32_t>(std::stoul(argv[i]));
        const auto piece = loaded.value.decode_token(id);
        if (!piece.ok()) {
            std::cout << "[" << id << "] ERROR: " << piece.errors[0] << '\n';
        } else {
            std::cout << "[" << id << "] ";
            for (unsigned char c : piece.value) {
                if (c >= 32 && c < 127) std::cout << c;
                else std::cout << "\\x" << std::hex << (int)c << std::dec;
            }
            std::cout << '\n';
        }
    }
    return 0;
}
