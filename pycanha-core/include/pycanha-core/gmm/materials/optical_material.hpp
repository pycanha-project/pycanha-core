#pragma once

#include <array>
#include <string>
#include <utility>

namespace pycanha::gmm {

/**
 * @brief Thermo-optical surface properties of a material.
 *
 * Restored from the legacy gmm with full fidelity: six optical degrees of
 * freedom held as an array, ordered as
 *   [0] emissivity_ir
 *   [1] specular_reflectivity_ir
 *   [2] transmissivity_ir
 *   [3] absorptivity_solar
 *   [4] specular_reflectivity_solar
 *   [5] transmissivity_solar
 * The default is a black body {1, 0, 0, 1, 0, 0}. Every value must lie in
 * [0, 1]; violations throw std::invalid_argument.
 *
 * Named convenience accessors (emissivity_ir / absorptivity_solar) and the
 * two-argument constructor address indices [0] and [3]; the remaining four
 * degrees of freedom default to 0 unless set via the full array API.
 */
class OpticalMaterial {
  public:
    using Properties = std::array<double, 6>;

    /// Default: black body.
    OpticalMaterial() = default;

    /// Two-property convenience: sets [0] = emissivity_ir, [3] =
    /// absorptivity_solar; all other DOF are 0.
    OpticalMaterial(std::string name, double emissivity_ir,
                    double absorptivity_solar);

    /// Full six-DOF constructor.
    OpticalMaterial(std::string name, Properties properties);

    [[nodiscard]] const std::string& get_name() const noexcept { return _name; }
    [[nodiscard]] const Properties& get_th_optical_properties() const noexcept {
        return _properties;
    }

    [[nodiscard]] double emissivity_ir() const noexcept {
        return _properties[0];
    }
    [[nodiscard]] double absorptivity_solar() const noexcept {
        return _properties[3];
    }

    void set_name(std::string name) noexcept { _name = std::move(name); }
    void set_th_optical_properties(Properties properties);
    void set_emissivity_ir(double emissivity_ir);
    void set_absorptivity_solar(double absorptivity_solar);

  private:
    static void validate(const Properties& properties);

    std::string _name;
    Properties _properties{1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
};

}  // namespace pycanha::gmm
