#include "test.hpp"

#include "strata/gemma4_image.hpp"

#include <string_view>

TEST_CASE("Gemma 4 image sizing preserves aspect ratio and patch groups") {
    const auto square = strata::gemma4_target_image_size(1024U, 1024U);
    REQUIRE(square.ok());
    REQUIRE(square.value.first == 768U);
    REQUIRE(square.value.second == 768U);
    REQUIRE(square.value.first % 48U == 0U);
    REQUIRE(square.value.second % 48U == 0U);

    const auto wide = strata::gemma4_target_image_size(1920U, 1080U);
    REQUIRE(wide.ok());
    REQUIRE(wide.value.first % 48U == 0U);
    REQUIRE(wide.value.second % 48U == 0U);
    REQUIRE(wide.value.first > wide.value.second);
}

TEST_CASE("Gemma 4 image decoder produces channel-last BF16 patches") {
    static constexpr char png[] =
        "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
        "\x00\x00\x00\x30\x00\x00\x00\x30\x08\x02\x00\x00\x00\xd8\x60\x6e"
        "\xd0\x00\x00\x00\x3e\x49\x44\x41\x54\x58\xc3\xed\xce\x31\x01\x00"
        "\x20\x10\x00\x21\xb5\x7f\xe7\xb7\x85\xe7\x00\x09\xd8\xb3\xfe\x72"
        "\xea\x80\x90\x90\x90\x50\x1d\x10\x12\x12\x12\xaa\x03\x42\x42\x42"
        "\x42\x75\x40\x48\x48\x48\xa8\x0e\x08\x09\x09\x09\xd5\x01\x21\x21"
        "\xa1\xd7\x2e\x49\xe1\x01\x5f\x6b\x6a\xca\xde\x00\x00\x00\x00\x49"
        "\x45\x4e\x44\xae\x42\x60\x82";
    const auto image = strata::prepare_gemma4_image(
        std::string_view(png, sizeof(png) - 1U), "image/png");
    REQUIRE(image.ok());
    REQUIRE(image.value.soft_tokens == 256U);
    REQUIRE(image.value.patch_width == 48U);
    REQUIRE(image.value.patch_height == 48U);
    REQUIRE(image.value.patches[0] == 1.0F);
    REQUIRE(image.value.patches[1] == -1.0F);
    REQUIRE(image.value.patches[2] == -1.0F);
}
