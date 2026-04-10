#pragma once

#include <Eigen/Core>

#include "pycanha-core/globals.hpp"

namespace pycanha {

class DenseTimeSeries {
  public:
    using MatrixType =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    DenseTimeSeries() = default;
    DenseTimeSeries(Index num_timesteps, Index num_columns);

    [[nodiscard]] Eigen::VectorXd& times() noexcept;
    [[nodiscard]] const Eigen::VectorXd& times() const noexcept;

    [[nodiscard]] MatrixType& values() noexcept;
    [[nodiscard]] const MatrixType& values() const noexcept;

    void set_row(Index row_idx, double time,
                 const Eigen::Ref<const Eigen::VectorXd>& row_values);

    [[nodiscard]] Eigen::VectorXd interpolate(double query_time) const;

    [[nodiscard]] Index num_timesteps() const noexcept;
    [[nodiscard]] Index num_columns() const noexcept;

    void resize(Index num_timesteps, Index num_columns);
    void reset() noexcept;

  private:
    Eigen::VectorXd _times;
    MatrixType _values;
};

}  // namespace pycanha
