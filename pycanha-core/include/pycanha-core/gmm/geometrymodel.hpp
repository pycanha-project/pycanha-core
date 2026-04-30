#pragma once

#include <cstdint>
#include <string>

namespace pycanha::gmm {

class GeometryModel {
  public:
    explicit GeometryModel(std::string name);
    ~GeometryModel() = default;

    GeometryModel(const GeometryModel&) = delete;
    GeometryModel& operator=(const GeometryModel&) = delete;
    GeometryModel(GeometryModel&&) noexcept = default;
    GeometryModel& operator=(GeometryModel&&) noexcept = default;

    [[nodiscard]] const std::string& name() const noexcept;

    [[deprecated(
        "Phase 0 placeholder; full GeometryModel API lands in later rewrite "
        "phases")]] [[nodiscard]] std::uint64_t
    structure_version() const noexcept;

  private:
    std::string _name;
    std::uint64_t _structure_version = 0;
};

}  // namespace pycanha::gmm
