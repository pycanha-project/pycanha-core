#include "pycanha-core/thermaldata/dense_time_series.hpp"

#include <stdexcept>

#include "pycanha-core/thermaldata/interpolation_utils.hpp"

namespace pycanha {

DenseTimeSeries::DenseTimeSeries(Index num_timesteps, Index num_columns) {
    resize(num_timesteps, num_columns);
}

Eigen::VectorXd& DenseTimeSeries::times() noexcept { return _times; }

const Eigen::VectorXd& DenseTimeSeries::times() const noexcept {
    return _times;
}

DenseTimeSeries::MatrixType& DenseTimeSeries::values() noexcept {
    return _values;
}

const DenseTimeSeries::MatrixType& DenseTimeSeries::values() const noexcept {
    return _values;
}

void DenseTimeSeries::set_row(
    Index row_idx, double time,
    const Eigen::Ref<const Eigen::VectorXd>& row_values) {
    if ((row_idx < 0) || (row_idx >= num_timesteps())) {
        throw std::out_of_range("DenseTimeSeries row index out of range");
    }
    if (row_values.size() != num_columns()) {
        throw std::invalid_argument(
            "DenseTimeSeries row size does not match column count");
    }

    _times(row_idx) = time;
    _values.row(row_idx) = row_values.transpose();
}

Eigen::VectorXd DenseTimeSeries::interpolate(double query_time) const {
    if (num_timesteps() == 0) {
        throw std::out_of_range("DenseTimeSeries has no data");
    }
    if (num_timesteps() == 1) {
        return _values.row(0).transpose();
    }

    const detail::InterpLocation location =
        detail::locate(_times.data(), num_timesteps(), query_time);
    const Index lower_index = location.lower_index;
    const Index upper_index = lower_index + 1;

    return ((1.0 - location.fraction) * _values.row(lower_index).transpose()) +
           (location.fraction * _values.row(upper_index).transpose());
}

Index DenseTimeSeries::num_timesteps() const noexcept { return _times.size(); }

Index DenseTimeSeries::num_columns() const noexcept { return _values.cols(); }

void DenseTimeSeries::resize(Index num_timesteps, Index num_columns) {
    if ((num_timesteps < 0) || (num_columns < 0)) {
        throw std::invalid_argument(
            "DenseTimeSeries dimensions must be non-negative");
    }

    _times = Eigen::VectorXd::Zero(num_timesteps);
    _values = MatrixType::Zero(num_timesteps, num_columns);
}

void DenseTimeSeries::reset() noexcept {
    _times.setZero();
    _values.setZero();
}

}  // namespace pycanha