#pragma once

#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "./id.hpp"

namespace pycanha::gmm {

using ColorRGB = std::array<uint8_t, 3>;

class NamedMaterial : public UniqueID {
    // Private members
    std::string _name;

  public:
    NamedMaterial() { _name = "Material_" + std::to_string(get_id()); }

    explicit NamedMaterial(std::string name) : _name(std::move(name)) {}

    [[nodiscard]] const std::string& get_name() const { return _name; }

    void set_name(std::string name) { _name = std::move(name); }
};

class BulkMaterial : public NamedMaterial {
    // Private members
    double _density;
    double _specific_heat;
    double _thermal_conductivity;

  public:
    /**
     * @brief Default constructor. Properties are set to zero.
     */
    BulkMaterial()
        : _density(0.0), _specific_heat(0.0), _thermal_conductivity(0.0) {}

    BulkMaterial(double density, double specific_heat,
                 double thermal_conductivity)
        : _density(density),
          _specific_heat(specific_heat),
          _thermal_conductivity(thermal_conductivity) {}

    [[nodiscard]] double get_density() const { return _density; }

    [[nodiscard]] double get_specific_heat() const { return _specific_heat; }

    [[nodiscard]] double get_thermal_conductivity() const {
        return _thermal_conductivity;
    }

    void set_density(double density) { _density = density; }

    void set_specific_heat(double specific_heat) {
        _specific_heat = specific_heat;
    }

    void set_thermal_conductivity(double thermal_conductivity) {
        _thermal_conductivity = thermal_conductivity;
    }
};

class OpticalMaterial : public NamedMaterial {
    // Private members
    std::array<double, 6> _th_optical_properties;

  public:
    /**
     * @brief Default constructor. Black body.
     */
    OpticalMaterial()
        : _th_optical_properties({1.0, 0.0, 0.0, 1.0, 0.0, 0.0}) {}

    explicit OpticalMaterial(std::array<double, 6> th_optical_properties)
        : _th_optical_properties(th_optical_properties) {}

    [[nodiscard]] std::array<double, 6> get_th_optical_properties() const {
        return _th_optical_properties;
    }

    void set_th_optical_properties(
        std::array<double, 6> th_optical_properties) {
        _th_optical_properties = th_optical_properties;
    }
};

class Color {
    // Private members
    ColorRGB _rgb;

  public:
    /**
     * @brief Construct a new Color object given the RGB values.
     */
    explicit Color(ColorRGB rgb) : _rgb(rgb) {}

    /**
     * @brief Construct a new Color object given the name.
     */
    explicit Color(const std::string& color_name)
        : _rgb(get_rgb_from_color_palette(color_name)) {}

    /**
     * @brief Get the RGB values.
     */
    ColorRGB get_rgb() { return _rgb; }

    static ColorRGB get_rgb_from_color_palette(const std::string& color_name) {
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
        try {
            return color_palette.at(color_name);
        } catch (const std::out_of_range&) {
            throw std::invalid_argument("Color name not found in palette.");
        }
    }
};

}  // namespace pycanha::gmm
