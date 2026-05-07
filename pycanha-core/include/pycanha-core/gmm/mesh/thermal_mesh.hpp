#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "pycanha-core/gmm/ids.hpp"

namespace pycanha::gmm {

enum class Side : unsigned char { Front = 0, Back = 1 };

class ThermalMesh {
  public:
    ThermalMesh();
    ThermalMesh(std::vector<double> dir1_cuts, std::vector<double> dir2_cuts);

    [[nodiscard]] std::span<const double> dir1_cuts() const noexcept;
    [[nodiscard]] std::span<const double> dir2_cuts() const noexcept;
    [[nodiscard]] std::size_t num_faces_per_side() const noexcept;
    [[nodiscard]] FaceId face_id(std::size_t i, std::size_t j,
                                 Side side) const noexcept;

  private:
    static void validate_cuts(std::span<const double> cuts,
                              const char* axis_name);

    std::vector<double> _dir1_cuts{0.0, 1.0};
    std::vector<double> _dir2_cuts{0.0, 1.0};
};

}  // namespace pycanha::gmm
