#pragma once

#include <Eigen/Core>
#include <cstdint>

#include "pycanha-core/globals.hpp"

namespace pycanha {

enum class InterpolationMethod : std::uint8_t {
    Linear,
    NearestLower,
    NearestUpper,
    Step,
};

enum class ExtrapolationMethod : std::uint8_t {
    Constant,
    Linear,
    Throw,
};

class LookupTable1D {
  public:
    LookupTable1D() = default;
    LookupTable1D(Eigen::VectorXd x, Eigen::VectorXd y,
                  InterpolationMethod interp = InterpolationMethod::Linear,
                  ExtrapolationMethod extrap = ExtrapolationMethod::Constant);

    [[nodiscard]] double operator()(double x_value) const;
    [[nodiscard]] double evaluate(double x_value) const;
    [[nodiscard]] Eigen::VectorXd evaluate(
        const Eigen::VectorXd& x_query) const;

    [[nodiscard]] const Eigen::VectorXd& x() const noexcept;
    [[nodiscard]] const Eigen::VectorXd& y() const noexcept;
    [[nodiscard]] Index size() const noexcept;

    [[nodiscard]] InterpolationMethod interpolation_method() const noexcept;
    [[nodiscard]] ExtrapolationMethod extrapolation_method() const noexcept;
    void set_interpolation_method(InterpolationMethod method) noexcept;
    void set_extrapolation_method(ExtrapolationMethod method) noexcept;

    void set_data(Eigen::VectorXd x, Eigen::VectorXd y);

  private:
    Eigen::VectorXd _x;
    Eigen::VectorXd _y;
    InterpolationMethod _interp{InterpolationMethod::Linear};
    ExtrapolationMethod _extrap{ExtrapolationMethod::Constant};
};

class LookupTableVec1D {
  public:
    using MatrixType =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    LookupTableVec1D() = default;
    LookupTableVec1D(
        Eigen::VectorXd x, MatrixType y,
        InterpolationMethod interp = InterpolationMethod::Linear,
        ExtrapolationMethod extrap = ExtrapolationMethod::Constant);

    [[nodiscard]] Eigen::VectorXd evaluate(double x_value) const;
    [[nodiscard]] MatrixType evaluate(const Eigen::VectorXd& x_query) const;

    [[nodiscard]] const Eigen::VectorXd& x() const noexcept;
    [[nodiscard]] const MatrixType& y() const noexcept;
    [[nodiscard]] Index size() const noexcept;
    [[nodiscard]] Index num_values() const noexcept;

    [[nodiscard]] InterpolationMethod interpolation_method() const noexcept;
    [[nodiscard]] ExtrapolationMethod extrapolation_method() const noexcept;
    void set_interpolation_method(InterpolationMethod method) noexcept;
    void set_extrapolation_method(ExtrapolationMethod method) noexcept;

    void set_data(Eigen::VectorXd x, MatrixType y);

  private:
    Eigen::VectorXd _x;
    MatrixType _y;
    InterpolationMethod _interp{InterpolationMethod::Linear};
    ExtrapolationMethod _extrap{ExtrapolationMethod::Constant};
};

}  // namespace pycanha
