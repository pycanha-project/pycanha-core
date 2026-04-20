#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/interpolation_utils.hpp"

namespace pycanha {

DenseMatrixTimeSeries::DenseMatrixTimeSeries(Index rows, Index cols)
    : _rows(rows), _cols(cols) {
    if ((rows < 0) || (cols < 0)) {
        throw std::invalid_argument(
            "DenseMatrixTimeSeries dimensions must be non-negative");
    }
}

void DenseMatrixTimeSeries::push_back(double time, MatrixType matrix) {
    if (_times.size() > 0) {
        if (time <= _times(_times.size() - 1)) {
            throw std::invalid_argument(
                "DenseMatrixTimeSeries times must be strictly increasing");
        }
        if ((matrix.rows() != _rows) || (matrix.cols() != _cols)) {
            throw std::invalid_argument(
                "DenseMatrixTimeSeries matrix dimensions must remain "
                "constant");
        }
    } else {
        if ((_rows == 0) && (_cols == 0)) {
            _rows = matrix.rows();
            _cols = matrix.cols();
        } else if ((matrix.rows() != _rows) || (matrix.cols() != _cols)) {
            throw std::invalid_argument(
                "DenseMatrixTimeSeries matrix dimensions do not match the "
                "expected shape");
        }
    }

    _times.conservativeResize(_times.size() + 1);
    _times(_times.size() - 1) = time;
    _matrices.push_back(std::move(matrix));
}

void DenseMatrixTimeSeries::reserve(std::size_t n) { _matrices.reserve(n); }

const Eigen::VectorXd& DenseMatrixTimeSeries::times() const noexcept {
    return _times;
}

const DenseMatrixTimeSeries::MatrixType& DenseMatrixTimeSeries::at(
    Index i) const {
    return _matrices.at(to_sizet(i));
}

DenseMatrixTimeSeries::MatrixType& DenseMatrixTimeSeries::at(Index i) {
    return _matrices.at(to_sizet(i));
}

double DenseMatrixTimeSeries::time_at(Index i) const { return _times(i); }

DenseMatrixTimeSeries::MatrixType DenseMatrixTimeSeries::interpolate(
    double query_time) const {
    if (_matrices.empty()) {
        throw std::out_of_range("DenseMatrixTimeSeries has no data");
    }
    if (_matrices.size() == 1U) {
        return _matrices.front();
    }

    const detail::InterpLocation location =
        detail::locate(_times.data(), _times.size(), query_time);
    const Index lower_index = location.lower_index;
    const Index upper_index = lower_index + 1;

    return ((1.0 - location.fraction) * _matrices.at(to_sizet(lower_index))) +
           (location.fraction * _matrices.at(to_sizet(upper_index)));
}

Index DenseMatrixTimeSeries::num_timesteps() const noexcept {
    return to_idx(_matrices.size());
}

Index DenseMatrixTimeSeries::rows() const noexcept { return _rows; }

Index DenseMatrixTimeSeries::cols() const noexcept { return _cols; }

}  // namespace pycanha
