#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha {

class SparseTimeSeries {
  public:
    using SparseMatrixType = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    SparseTimeSeries() = default;

    void push_back(double time, SparseMatrixType matrix);
    void reserve(std::size_t n);

    [[nodiscard]] const Eigen::VectorXd& times() const noexcept;
    [[nodiscard]] const SparseMatrixType& at(Index i) const;
    [[nodiscard]] SparseMatrixType& at(Index i);
    [[nodiscard]] double time_at(Index i) const;

    [[nodiscard]] SparseMatrixType interpolate(double query_time) const;

    [[nodiscard]] Index num_timesteps() const noexcept;
    [[nodiscard]] Index rows() const noexcept;
    [[nodiscard]] Index cols() const noexcept;
    [[nodiscard]] Index nnz() const noexcept;

  private:
    Eigen::VectorXd _times;
    std::vector<SparseMatrixType> _matrices;
    std::vector<SparseMatrixType::StorageIndex> _ref_outer;
    std::vector<SparseMatrixType::StorageIndex> _ref_inner;
};

}  // namespace pycanha