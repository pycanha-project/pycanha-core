#include "pycanha-core/thermaldata/sparse_time_series.hpp"

#include <algorithm>
#include <stdexcept>

#include "pycanha-core/thermaldata/interpolation_utils.hpp"

namespace pycanha {

namespace {

void validate_same_pattern(
    const SparseTimeSeries::SparseMatrixType& matrix,
    const std::vector<SparseTimeSeries::SparseMatrixType::StorageIndex>&
        ref_outer,
    const std::vector<SparseTimeSeries::SparseMatrixType::StorageIndex>&
        ref_inner) {
    using StorageIndex = SparseTimeSeries::SparseMatrixType::StorageIndex;

    const auto* outer_ptr = matrix.outerIndexPtr();
    const auto* inner_ptr = matrix.innerIndexPtr();
    const auto outer_size = static_cast<std::size_t>(matrix.outerSize()) + 1U;
    const auto inner_size = static_cast<std::size_t>(matrix.nonZeros());

    const std::vector<StorageIndex> outer(outer_ptr, outer_ptr + outer_size);
    const std::vector<StorageIndex> inner(inner_ptr, inner_ptr + inner_size);

    if ((outer != ref_outer) || (inner != ref_inner)) {
        throw std::invalid_argument(
            "SparseTimeSeries matrices must share the same sparsity pattern");
    }
}

}  // namespace

void SparseTimeSeries::push_back(double time, SparseMatrixType matrix) {
    if (_times.size() > 0) {
        if (time <= _times(_times.size() - 1)) {
            throw std::invalid_argument(
                "SparseTimeSeries times must be strictly increasing");
        }
        if ((matrix.rows() != rows()) || (matrix.cols() != cols())) {
            throw std::invalid_argument(
                "SparseTimeSeries matrix dimensions must remain constant");
        }
    }

    matrix.makeCompressed();

    if (_matrices.empty()) {
        const auto* outer_ptr = matrix.outerIndexPtr();
        const auto* inner_ptr = matrix.innerIndexPtr();
        const auto outer_size =
            static_cast<std::size_t>(matrix.outerSize()) + 1U;
        const auto inner_size = static_cast<std::size_t>(matrix.nonZeros());
        _ref_outer.assign(outer_ptr, outer_ptr + outer_size);
        _ref_inner.assign(inner_ptr, inner_ptr + inner_size);
    } else {
        validate_same_pattern(matrix, _ref_outer, _ref_inner);
    }

    _times.conservativeResize(_times.size() + 1);
    _times(_times.size() - 1) = time;
    _matrices.push_back(std::move(matrix));
}

void SparseTimeSeries::reserve(std::size_t n) { _matrices.reserve(n); }

const Eigen::VectorXd& SparseTimeSeries::times() const noexcept {
    return _times;
}

const SparseTimeSeries::SparseMatrixType& SparseTimeSeries::at(Index i) const {
    return _matrices.at(to_sizet(i));
}

SparseTimeSeries::SparseMatrixType& SparseTimeSeries::at(Index i) {
    return _matrices.at(to_sizet(i));
}

double SparseTimeSeries::time_at(Index i) const { return _times(i); }

SparseTimeSeries::SparseMatrixType SparseTimeSeries::interpolate(
    double query_time) const {
    if (_matrices.empty()) {
        throw std::out_of_range("SparseTimeSeries has no data");
    }
    if (_matrices.size() == 1U) {
        return _matrices.front();
    }

    const detail::InterpLocation location =
        detail::locate(_times.data(), _times.size(), query_time);
    const Index lower_index = location.lower_index;
    const Index upper_index = lower_index + 1;

    SparseMatrixType result = _matrices.at(to_sizet(lower_index));
    double* result_values = result.valuePtr();
    const double* lower_values = _matrices.at(to_sizet(lower_index)).valuePtr();
    const double* upper_values = _matrices.at(to_sizet(upper_index)).valuePtr();

    for (Index value_index = 0; value_index < result.nonZeros();
         ++value_index) {
        result_values[value_index] =
            ((1.0 - location.fraction) * lower_values[value_index]) +
            (location.fraction * upper_values[value_index]);
    }

    return result;
}

Index SparseTimeSeries::num_timesteps() const noexcept {
    return static_cast<Index>(_matrices.size());
}

Index SparseTimeSeries::rows() const noexcept {
    return _matrices.empty() ? 0 : _matrices.front().rows();
}

Index SparseTimeSeries::cols() const noexcept {
    return _matrices.empty() ? 0 : _matrices.front().cols();
}

Index SparseTimeSeries::nnz() const noexcept {
    return _matrices.empty() ? 0 : _matrices.front().nonZeros();
}

}  // namespace pycanha