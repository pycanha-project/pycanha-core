#include "pycanha-core/thermaldata/sparse_time_series.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/interpolation_utils.hpp"

namespace pycanha {

namespace {

template <typename Scalar>
using DynamicVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

constexpr std::string_view k_pattern_error_message =
    "SparseTimeSeries matrices must share the same sparsity pattern";

void validate_same_pattern(
    const SparseTimeSeries::SparseMatrixType& matrix,
    const std::vector<SparseTimeSeries::SparseMatrixType::StorageIndex>&
        ref_outer,
    const std::vector<SparseTimeSeries::SparseMatrixType::StorageIndex>&
        ref_inner) {
    using StorageIndex = SparseTimeSeries::SparseMatrixType::StorageIndex;

    const auto outer_size = static_cast<std::size_t>(matrix.outerSize()) + 1U;
    const auto inner_size = static_cast<std::size_t>(matrix.nonZeros());
    const Eigen::Map<const DynamicVector<StorageIndex>> outer_map(
        matrix.outerIndexPtr(), to_idx(outer_size));
    const Eigen::Map<const DynamicVector<StorageIndex>> inner_map(
        matrix.innerIndexPtr(), to_idx(inner_size));

    if ((ref_outer.size() != outer_size) || (ref_inner.size() != inner_size)) {
        throw std::invalid_argument(std::string(k_pattern_error_message));
    }

    for (Index index = 0; index < outer_map.size(); ++index) {
        if (ref_outer.at(to_sizet(index)) != outer_map(index)) {
            throw std::invalid_argument(std::string(k_pattern_error_message));
        }
    }

    for (Index index = 0; index < inner_map.size(); ++index) {
        if (ref_inner.at(to_sizet(index)) != inner_map(index)) {
            throw std::invalid_argument(std::string(k_pattern_error_message));
        }
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

    if (_matrices.empty()) {
        using StorageIndex = SparseMatrixType::StorageIndex;

        const auto outer_size =
            static_cast<std::size_t>(matrix.outerSize()) + 1U;
        const auto inner_size = static_cast<std::size_t>(matrix.nonZeros());
        const Eigen::Map<const DynamicVector<StorageIndex>> outer_map(
            matrix.outerIndexPtr(), to_idx(outer_size));
        const Eigen::Map<const DynamicVector<StorageIndex>> inner_map(
            matrix.innerIndexPtr(), to_idx(inner_size));

        _ref_outer.resize(outer_size);
        _ref_inner.resize(inner_size);

        for (Index index = 0; index < outer_map.size(); ++index) {
            _ref_outer[to_sizet(index)] = outer_map(index);
        }

        for (Index index = 0; index < inner_map.size(); ++index) {
            _ref_inner[to_sizet(index)] = inner_map(index);
        }
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
    Eigen::Map<Eigen::VectorXd> result_values(result.valuePtr(),
                                              result.nonZeros());
    const Eigen::Map<const Eigen::VectorXd> lower_values(
        _matrices.at(to_sizet(lower_index)).valuePtr(), result.nonZeros());
    const Eigen::Map<const Eigen::VectorXd> upper_values(
        _matrices.at(to_sizet(upper_index)).valuePtr(), result.nonZeros());

    result_values = ((1.0 - location.fraction) * lower_values) +
                    (location.fraction * upper_values);

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
