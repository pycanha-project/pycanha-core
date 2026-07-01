#pragma once

#include <string>
#include <utility>

namespace pycanha::gmm {

/**
 * @brief Structural / thermal bulk properties of a material.
 *
 * Restored from the legacy gmm with a name and constructor/setter validation.
 * Density, conductivity and specific heat must all be non-negative; any
 * violation throws std::invalid_argument.
 */
class BulkMaterial {
  public:
    /// Default: unnamed black-hole-free material with all properties zero.
    BulkMaterial() = default;

    BulkMaterial(std::string name, double density, double conductivity,
                 double specific_heat);

    [[nodiscard]] const std::string& get_name() const noexcept { return _name; }
    [[nodiscard]] double get_density() const noexcept { return _density; }
    [[nodiscard]] double get_conductivity() const noexcept {
        return _conductivity;
    }
    [[nodiscard]] double get_specific_heat() const noexcept {
        return _specific_heat;
    }

    void set_name(std::string name) noexcept { _name = std::move(name); }
    void set_density(double density);
    void set_conductivity(double conductivity);
    void set_specific_heat(double specific_heat);

  private:
    static void validate(double density, double conductivity,
                         double specific_heat);

    std::string _name;
    double _density = 0.0;
    double _conductivity = 0.0;
    double _specific_heat = 0.0;
};

}  // namespace pycanha::gmm
