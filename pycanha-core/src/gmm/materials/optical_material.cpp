#include "pycanha-core/gmm/materials/optical_material.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace pycanha::gmm {

OpticalMaterial::OpticalMaterial(std::string name, double emissivity_ir,
                                 double absorptivity_solar)
    : _name(std::move(name)),
      _properties{emissivity_ir, 0.0, 0.0, absorptivity_solar, 0.0, 0.0} {
    validate(_properties);
}

OpticalMaterial::OpticalMaterial(std::string name, Properties properties)
    : _name(std::move(name)), _properties(properties) {
    validate(_properties);
}

void OpticalMaterial::set_th_optical_properties(Properties properties) {
    validate(properties);
    _properties = properties;
}

void OpticalMaterial::set_emissivity_ir(double emissivity_ir) {
    Properties candidate = _properties;
    candidate[0] = emissivity_ir;
    validate(candidate);
    _properties = candidate;
}

void OpticalMaterial::set_absorptivity_solar(double absorptivity_solar) {
    Properties candidate = _properties;
    candidate[3] = absorptivity_solar;
    validate(candidate);
    _properties = candidate;
}

void OpticalMaterial::validate(const Properties& properties) {
    for (const double value : properties) {
        if (value < 0.0 || value > 1.0) {
            throw std::invalid_argument(
                "OpticalMaterial: every optical property must be in [0, 1]");
        }
    }
}

}  // namespace pycanha::gmm
