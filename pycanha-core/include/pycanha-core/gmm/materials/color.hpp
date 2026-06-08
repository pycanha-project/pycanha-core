#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace pycanha::gmm {

using ColorRGB = std::array<std::uint8_t, 3>;

/**
 * @brief A 3-channel (RGB) color.
 *
 * Restored from the legacy gmm. Construct from explicit channels, from a
 * ColorRGB triple, or from a named palette entry (see the palette in the
 * implementation file).
 */
class Color {
  public:
    Color(std::uint8_t red, std::uint8_t green, std::uint8_t blue) noexcept
        : _rgb{red, green, blue} {}

    explicit Color(ColorRGB rgb) noexcept : _rgb(rgb) {}

    /// @throws std::invalid_argument if @p color_name is not in the palette.
    explicit Color(const std::string& color_name);

    [[nodiscard]] const ColorRGB& get_rgb() const noexcept { return _rgb; }

    [[nodiscard]] std::uint8_t red() const noexcept { return _rgb[0]; }
    [[nodiscard]] std::uint8_t green() const noexcept { return _rgb[1]; }
    [[nodiscard]] std::uint8_t blue() const noexcept { return _rgb[2]; }

    /// @throws std::invalid_argument if @p color_name is not in the palette.
    [[nodiscard]] static ColorRGB get_rgb_from_color_palette(
        const std::string& color_name);

  private:
    ColorRGB _rgb;
};

}  // namespace pycanha::gmm
