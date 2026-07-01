#include "pycanha-core/gmm/materials/bulk_material.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace pycanha::gmm {

BulkMaterial::BulkMaterial(std::string name, double density,
                           double conductivity, double specific_heat)
    : _name(std::move(name)),
      _density(density),
      _conductivity(conductivity),
      _specific_heat(specific_heat) {
    validate(_density, _conductivity, _specific_heat);
}

void BulkMaterial::set_density(double density) {
    validate(density, _conductivity, _specific_heat);
    _density = density;
}

void BulkMaterial::set_conductivity(double conductivity) {
    validate(_density, conductivity, _specific_heat);
    _conductivity = conductivity;
}

void BulkMaterial::set_specific_heat(double specific_heat) {
    validate(_density, _conductivity, specific_heat);
    _specific_heat = specific_heat;
}

void BulkMaterial::validate(double density, double conductivity,
                            double specific_heat) {
    if (density < 0.0) {
        throw std::invalid_argument("BulkMaterial: density must be >= 0");
    }
    if (conductivity < 0.0) {
        throw std::invalid_argument("BulkMaterial: conductivity must be >= 0");
    }
    if (specific_heat < 0.0) {
        throw std::invalid_argument("BulkMaterial: specific_heat must be >= 0");
    }
}

}  // namespace pycanha::gmm
