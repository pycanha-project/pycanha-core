#include "pycanha-core/gmm/geometrymodel.hpp"

#include <cstdint>
#include <string>
#include <utility>

namespace pycanha::gmm {

GeometryModel::GeometryModel(std::string name) : _name(std::move(name)) {}

const std::string& GeometryModel::name() const noexcept { return _name; }

std::uint64_t GeometryModel::structure_version() const noexcept {
    return _structure_version;
}

}  // namespace pycanha::gmm
