#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha {

class DenseMatrixTimeSeries {
  public:
    using MatrixType =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    DenseMatrixTimeSeries() = default;
    DenseMatrixTimeSeries(Index rows, Index cols);

    void push_back(double time, MatrixType matrix);
    void reserve(std::size_t n);

    [[nodiscard]] const Eigen::VectorXd& times() const noexcept;
    [[nodiscard]] const MatrixType& at(Index i) const;
    [[nodiscard]] MatrixType& at(Index i);
    [[nodiscard]] double time_at(Index i) const;

    [[nodiscard]] MatrixType interpolate(double query_time) const;

    [[nodiscard]] Index num_timesteps() const noexcept;
    [[nodiscard]] Index rows() const noexcept;
    [[nodiscard]] Index cols() const noexcept;

  private:
    Eigen::VectorXd _times;
    std::vector<MatrixType> _matrices;
    Index _rows = 0;
    Index _cols = 0;
};

}  // namespace pycanha
