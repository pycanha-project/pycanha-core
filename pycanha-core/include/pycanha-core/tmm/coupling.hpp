
#pragma once

#include <Eigen/Sparse>

namespace pycanha {

/// Representation of a thermal coupling between two nodes.
class Coupling {
  public:
    /// Construct a coupling between two user nodes with a given conductance.
    Coupling(int node_1, int node_2, double value) noexcept;

    [[nodiscard]] int get_node_1() const noexcept;
    [[nodiscard]] int get_node_2() const noexcept;
    [[nodiscard]] double get_value() const noexcept;

    void set_node_1(int node_1) noexcept;
    void set_node_2(int node_2) noexcept;
    void set_value(double value) noexcept;

  private:
    Eigen::Triplet<double> _conductor;
};

}  // namespace pycanha
