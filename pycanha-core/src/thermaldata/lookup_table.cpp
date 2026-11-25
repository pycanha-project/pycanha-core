#include "pycanha-core/thermaldata/lookup_table.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/interpolation_utils.hpp"

namespace pycanha {

namespace {

void validate_x_data(const Eigen::VectorXd& x) {
    if (x.size() == 0) {
        throw std::invalid_argument("Lookup table x data must not be empty");
    }

    for (Index i = 1; i < x.size(); ++i) {
        if (x(i) <= x(i - 1)) {
            throw std::invalid_argument(
                "Lookup table x data must be strictly increasing");
        }
    }
}

// Throws when data of this size and ends cannot be periodic. A table without
// data yet is checked when it gets some.
void validate_periodic(ExtrapolationMethod extrap, Index size, bool closed) {
    if ((extrap != ExtrapolationMethod::Periodic) || (size == 0)) {
        return;
    }
    if (size == 1) {
        throw std::invalid_argument(
            "A periodic lookup table needs at least two points");
    }
    if (!closed) {
        throw std::invalid_argument(
            "The first and last values of a periodic lookup table must be "
            "equal");
    }
}

[[nodiscard]] bool ends_equal(const Eigen::VectorXd& y) {
    return (y.size() > 0) && (y(0) == y(y.size() - 1));
}

[[nodiscard]] bool ends_equal(const LookupTableVec1D::MatrixType& y) {
    return (y.rows() > 0) && (y.row(0) == y.row(y.rows() - 1));
}

// The query a periodic table is looked up at; other tables keep theirs.
[[nodiscard]] double wrap_query(const Eigen::VectorXd& x_data, double query,
                                ExtrapolationMethod extrap) {
    return extrap == ExtrapolationMethod::Periodic
               ? detail::wrap_periodic(query, x_data(0),
                                       x_data(x_data.size() - 1))
               : query;
}

double evaluate_scalar_impl(const Eigen::VectorXd& x_data,
                            const Eigen::VectorXd& y_data, double query,
                            InterpolationMethod interp,
                            ExtrapolationMethod extrap) {
    if (x_data.size() != y_data.size()) {
        throw std::invalid_argument("Lookup table x/y size mismatch");
    }
    if (x_data.size() == 0) {
        throw std::out_of_range("Lookup table has no data");
    }
    if (x_data.size() == 1) {
        if ((extrap == ExtrapolationMethod::Throw) && (query != x_data(0))) {
            throw std::out_of_range(
                "Lookup table query is outside the table range");
        }
        return y_data(0);
    }

    const double wrapped = wrap_query(x_data, query, extrap);
    if ((extrap == ExtrapolationMethod::Periodic) && std::isnan(wrapped)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const detail::InterpLocation location =
        detail::locate(x_data.data(), x_data.size(), wrapped);
    const Index lower_index = location.lower_index;
    const Index upper_index = lower_index + 1;

    if (location.below_range || location.above_range) {
        switch (extrap) {
            case ExtrapolationMethod::Constant:
                return location.below_range ? y_data(0)
                                            : y_data(y_data.size() - 1);
            case ExtrapolationMethod::Linear:
                return ((1.0 - location.fraction) * y_data(lower_index)) +
                       (location.fraction * y_data(upper_index));
            case ExtrapolationMethod::Throw:
                throw std::out_of_range(
                    "Lookup table query is outside the table range");
            case ExtrapolationMethod::Periodic:
                // Wrapped above, so never outside the range.
                break;
        }
    }

    if (location.fraction <= 0.0) {
        return y_data(lower_index);
    }
    if (location.fraction >= 1.0) {
        return y_data(upper_index);
    }

    switch (interp) {
        case InterpolationMethod::Linear:
            return ((1.0 - location.fraction) * y_data(lower_index)) +
                   (location.fraction * y_data(upper_index));
        case InterpolationMethod::NearestLower:
        case InterpolationMethod::Step:
            return y_data(lower_index);
        case InterpolationMethod::NearestUpper:
            return y_data(upper_index);
    }

    throw std::invalid_argument("Unsupported interpolation method");
}

Eigen::VectorXd evaluate_vector_impl(const Eigen::VectorXd& x_data,
                                     const LookupTableVec1D::MatrixType& y_data,
                                     double query, InterpolationMethod interp,
                                     ExtrapolationMethod extrap) {
    if (x_data.size() != y_data.rows()) {
        throw std::invalid_argument("Lookup table x/y size mismatch");
    }
    if (x_data.size() == 0) {
        throw std::out_of_range("Lookup table has no data");
    }
    if (x_data.size() == 1) {
        if ((extrap == ExtrapolationMethod::Throw) && (query != x_data(0))) {
            throw std::out_of_range(
                "Lookup table query is outside the table range");
        }
        return y_data.row(0).transpose();
    }

    const double wrapped = wrap_query(x_data, query, extrap);
    if ((extrap == ExtrapolationMethod::Periodic) && std::isnan(wrapped)) {
        return Eigen::VectorXd::Constant(
            y_data.cols(), std::numeric_limits<double>::quiet_NaN());
    }
    const detail::InterpLocation location =
        detail::locate(x_data.data(), x_data.size(), wrapped);
    const Index lower_index = location.lower_index;
    const Index upper_index = lower_index + 1;

    if (location.below_range || location.above_range) {
        switch (extrap) {
            case ExtrapolationMethod::Constant:
                return location.below_range
                           ? y_data.row(0).transpose()
                           : y_data.row(y_data.rows() - 1).transpose();
            case ExtrapolationMethod::Linear:
                return ((1.0 - location.fraction) *
                        y_data.row(lower_index).transpose()) +
                       (location.fraction *
                        y_data.row(upper_index).transpose());
            case ExtrapolationMethod::Throw:
                throw std::out_of_range(
                    "Lookup table query is outside the table range");
            case ExtrapolationMethod::Periodic:
                // Wrapped above, so never outside the range.
                break;
        }
    }

    if (location.fraction <= 0.0) {
        return y_data.row(lower_index).transpose();
    }
    if (location.fraction >= 1.0) {
        return y_data.row(upper_index).transpose();
    }

    switch (interp) {
        case InterpolationMethod::Linear:
            return ((1.0 - location.fraction) *
                    y_data.row(lower_index).transpose()) +
                   (location.fraction * y_data.row(upper_index).transpose());
        case InterpolationMethod::NearestLower:
        case InterpolationMethod::Step:
            return y_data.row(lower_index).transpose();
        case InterpolationMethod::NearestUpper:
            return y_data.row(upper_index).transpose();
    }

    throw std::invalid_argument("Unsupported interpolation method");
}

}  // namespace

LookupTable1D::LookupTable1D(Eigen::VectorXd x, Eigen::VectorXd y,
                             InterpolationMethod interp,
                             ExtrapolationMethod extrap)
    : _interp(interp), _extrap(extrap) {
    set_data(std::move(x), std::move(y));
}

double LookupTable1D::operator()(double x_value) const {
    return evaluate(x_value);
}

double LookupTable1D::evaluate(double x_value) const {
    return evaluate_scalar_impl(_x, _y, x_value, _interp, _extrap);
}

Eigen::VectorXd LookupTable1D::evaluate(const Eigen::VectorXd& x_query) const {
    Eigen::VectorXd result(x_query.size());
    for (Index i = 0; i < x_query.size(); ++i) {
        result(i) = evaluate(x_query(i));
    }
    return result;
}

const Eigen::VectorXd& LookupTable1D::x() const noexcept { return _x; }

const Eigen::VectorXd& LookupTable1D::y() const noexcept { return _y; }

Index LookupTable1D::size() const noexcept { return _x.size(); }

InterpolationMethod LookupTable1D::interpolation_method() const noexcept {
    return _interp;
}

ExtrapolationMethod LookupTable1D::extrapolation_method() const noexcept {
    return _extrap;
}

void LookupTable1D::set_interpolation_method(
    InterpolationMethod method) noexcept {
    _interp = method;
}

void LookupTable1D::set_extrapolation_method(ExtrapolationMethod method) {
    validate_periodic(method, size(), ends_equal(_y));
    _extrap = method;
}

void LookupTable1D::set_data(Eigen::VectorXd x, Eigen::VectorXd y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Lookup table x/y size mismatch");
    }

    validate_x_data(x);
    validate_periodic(_extrap, x.size(), ends_equal(y));
    _x = std::move(x);
    _y = std::move(y);
}

void LookupTable1D::assign_x(std::span<const double> x) {
    if (std::cmp_not_equal(x.size(), _x.size())) {
        throw std::invalid_argument(
            "Lookup table x must keep its size when assigned in place");
    }
    if (std::ranges::adjacent_find(x, std::greater_equal<>()) != x.end()) {
        throw std::invalid_argument(
            "Lookup table x data must be strictly increasing");
    }
    std::ranges::copy(x, _x.begin());
}

void LookupTable1D::assign_y(std::span<const double> y) {
    if (std::cmp_not_equal(y.size(), _y.size())) {
        throw std::invalid_argument(
            "Lookup table y must keep its size when assigned in place");
    }
    validate_periodic(_extrap, size(), !y.empty() && (y.front() == y.back()));
    std::ranges::copy(y, _y.begin());
}

LookupTableVec1D::LookupTableVec1D(Eigen::VectorXd x, MatrixType y,
                                   InterpolationMethod interp,
                                   ExtrapolationMethod extrap)
    : _interp(interp), _extrap(extrap) {
    set_data(std::move(x), std::move(y));
}

Eigen::VectorXd LookupTableVec1D::evaluate(double x_value) const {
    return evaluate_vector_impl(_x, _y, x_value, _interp, _extrap);
}

LookupTableVec1D::MatrixType LookupTableVec1D::evaluate(
    const Eigen::VectorXd& x_query) const {
    MatrixType result(x_query.size(), num_values());
    for (Index i = 0; i < x_query.size(); ++i) {
        result.row(i) = evaluate(x_query(i)).transpose();
    }
    return result;
}

const Eigen::VectorXd& LookupTableVec1D::x() const noexcept { return _x; }

const LookupTableVec1D::MatrixType& LookupTableVec1D::y() const noexcept {
    return _y;
}

Index LookupTableVec1D::size() const noexcept { return _x.size(); }

Index LookupTableVec1D::num_values() const noexcept { return _y.cols(); }

InterpolationMethod LookupTableVec1D::interpolation_method() const noexcept {
    return _interp;
}

ExtrapolationMethod LookupTableVec1D::extrapolation_method() const noexcept {
    return _extrap;
}

void LookupTableVec1D::set_interpolation_method(
    InterpolationMethod method) noexcept {
    _interp = method;
}

void LookupTableVec1D::set_extrapolation_method(ExtrapolationMethod method) {
    validate_periodic(method, size(), ends_equal(_y));
    _extrap = method;
}

void LookupTableVec1D::set_data(Eigen::VectorXd x, MatrixType y) {
    if (x.size() != y.rows()) {
        throw std::invalid_argument("Lookup table x/y size mismatch");
    }

    validate_x_data(x);
    validate_periodic(_extrap, x.size(), ends_equal(y));
    _x = std::move(x);
    _y = std::move(y);
}

}  // namespace pycanha
