#include "pycanha-core/gmm/materials/color.hpp"

#include <map>
#include <stdexcept>
#include <string>

namespace pycanha::gmm {

Color::Color(const std::string& color_name)
    : _rgb(get_rgb_from_color_palette(color_name)) {}

ColorRGB Color::get_rgb_from_color_palette(const std::string& color_name) {
    static const std::map<std::string, ColorRGB> color_palette = {
        {"BLUE_CYAN", {0, 127, 255}},
        {"CYAN", {0, 255, 255}},
        {"RED", {255, 0, 0}},
        {"GREEN", {0, 255, 0}},
        {"BLUE", {0, 0, 255}},
        {"BLACK", {0, 0, 0}},
        {"MAGENTA", {255, 0, 255}},
        {"YELLOW", {255, 255, 0}},
        {"ORANGE", {255, 127, 0}},
        {"YELLOW_GREEN", {127, 255, 0}},
        {"TURQUOISE", {0, 255, 127}},
        {"VIOLET", {127, 0, 255}},
        {"PURPLE", {255, 0, 127}},
        {"VERY_DARK_GREY", {84, 84, 84}},
        {"LIGHT_GREY", {168, 168, 168}},
        {"REDDISH_BROWN", {191, 63, 63}},
        {"ABSINTH", {191, 191, 63}},
        {"GREY_GREEN", {63, 191, 63}},
        {"METAL_GREY", {63, 191, 191}},
        {"LAVENDER", {63, 63, 191}},
        {"MAGENTA_GREY", {191, 63, 191}},
        {"DARK_RED", {127, 0, 0}},
        {"DARK_GREEN", {0, 127, 0}},
        {"DARK_BLUE", {0, 0, 127}},
        {"PALE_RED", {255, 127, 127}},
        {"PALE_GREEN", {127, 255, 127}},
        {"PALE_BLUE", {127, 127, 255}},
        {"GREY_BLACK", {36, 36, 36}},
        {"DARK_GREY", {112, 112, 112}},
        {"GREY", {140, 140, 140}},
        {"VERY_LIGHT_GREY", {219, 219, 219}},
        {"WHITE", {255, 255, 255}}};

    const auto it = color_palette.find(color_name);
    if (it == color_palette.end()) {
        throw std::invalid_argument("Color name not found in palette: " +
                                    color_name);
    }
    return it->second;
}

}  // namespace pycanha::gmm
