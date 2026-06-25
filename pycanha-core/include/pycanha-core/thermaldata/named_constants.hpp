#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"

namespace pycanha {

// Raw storage for ESATAN user-defined named constants read from a .TMD file.
//
// Each constant is a transient series sharing a single time axis (the
// DataGroup1/times dataset). Steady-state data is simply a single timestep.
// Values are stored faithfully per ESATAN type ($REAL -> double,
// $INTEGER -> std::int64_t, $CHARACTER -> contiguous fixed-width chars).
//
// This is a passive container: values are never interpolated. Callers index
// the raw rows/columns directly.
class NamedConstants {
  public:
    using RealMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using IntMatrix = Eigen::Matrix<std::int64_t, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>;

    NamedConstants() = default;

    // Shared time axis (== DataGroup1/times); one entry per timestep.
    [[nodiscard]] Eigen::VectorXd& times() noexcept;
    [[nodiscard]] const Eigen::VectorXd& times() const noexcept;

    // $REAL constants. real_values() is (num_timesteps x num_real), row-major.
    [[nodiscard]] std::vector<std::string>& real_names() noexcept;
    [[nodiscard]] const std::vector<std::string>& real_names() const noexcept;
    [[nodiscard]] RealMatrix& real_values() noexcept;
    [[nodiscard]] const RealMatrix& real_values() const noexcept;

    // $INTEGER constants. values() is a (num_timesteps x num_int) row-major
    // matrix.
    [[nodiscard]] std::vector<std::string>& int_names() noexcept;
    [[nodiscard]] const std::vector<std::string>& int_names() const noexcept;
    [[nodiscard]] IntMatrix& int_values() noexcept;
    [[nodiscard]] const IntMatrix& int_values() const noexcept;

    // $CHARACTER constants. Stored as a contiguous fixed-width char buffer,
    // laid out row-major by (timestep, constant): the value at (t, c) occupies
    // char_width() bytes starting at offset (t * num_char + c) * char_width().
    [[nodiscard]] std::vector<std::string>& char_names() noexcept;
    [[nodiscard]] const std::vector<std::string>& char_names() const noexcept;
    [[nodiscard]] std::size_t char_width() const noexcept;

    // Replaces the character storage. buffer must have size
    // num_timesteps * char_names().size() * width.
    void set_char_storage(std::size_t width, std::vector<char> buffer);

    // Raw fixed-width view of the character constant c at timestep t. The view
    // has length char_width(); ESATAN pads with trailing spaces, so callers
    // that want a trimmed value should strip them.
    [[nodiscard]] std::string_view char_value(Index t, Index c) const;

    [[nodiscard]] Index num_timesteps() const noexcept;

  private:
    Eigen::VectorXd _times;

    std::vector<std::string> _real_names;
    RealMatrix _real_values;

    std::vector<std::string> _int_names;
    IntMatrix _int_values;

    std::vector<std::string> _char_names;
    std::size_t _char_width = 0;
    std::vector<char> _char_values;
};

}  // namespace pycanha
