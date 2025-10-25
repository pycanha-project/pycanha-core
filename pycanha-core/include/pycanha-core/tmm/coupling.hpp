
#pragma once

#include <Eigen/Sparse>

#include "pycanha-core/globals.hpp"

namespace pycanha {

/// Representation of a thermal coupling between two nodes.
class Coupling {
  public:
    /// Construct a coupling between two user nodes with a given conductance.
    Coupling(Index node_1, Index node_2, double value) noexcept;

    [[nodiscard]] Index get_node_1() const noexcept;
    [[nodiscard]] Index get_node_2() const noexcept;
    [[nodiscard]] double get_value() const noexcept;

    void set_node_1(Index node_1) noexcept;
    void set_node_2(Index node_2) noexcept;
    void set_value(double value) noexcept;

  private:
    Eigen::Triplet<double, Index> _conductor;
};

}  // namespace pycanha
