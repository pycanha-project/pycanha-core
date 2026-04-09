#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

class Formulas {
  public:
    Formulas();
    Formulas(std::shared_ptr<ThermalNetwork> network,
             std::shared_ptr<Parameters> parameters);

    Formulas(const Formulas&) = delete;
    Formulas& operator=(const Formulas&) = delete;
    Formulas(Formulas&&) noexcept = default;
    Formulas& operator=(Formulas&&) noexcept = default;
    ~Formulas() = default;

    void associate(std::shared_ptr<ThermalNetwork> network,
                   std::shared_ptr<Parameters> parameters);

    [[nodiscard]] std::shared_ptr<ThermalNetwork> network() const noexcept {
        return _network;
    }
    [[nodiscard]] std::shared_ptr<Parameters> parameters() const noexcept {
        return _parameters;
    }

    ParameterFormula create_parameter_formula(Entity entity,
                                              const std::string& parameter);

    void add_formula(const Formula& formula);
    void add_formula(const std::shared_ptr<Formula>& formula);

    void validate_for_execution();
    void compile_formulas();
    void apply_formulas();
    void apply_compiled_formulas();
    void calculate_derivatives();
    void lock_parameters_for_execution();
    void unlock_parameters() noexcept;
    [[nodiscard]] bool is_validation_current() const noexcept;
    [[nodiscard]] bool is_compiled_current() const noexcept;

    [[nodiscard]] const std::vector<std::shared_ptr<Formula>>& formulas()
        const noexcept {
        return _formulas;
    }

    [[nodiscard]] const std::unordered_map<
        std::string, std::vector<std::shared_ptr<Formula>>>&
    parameter_dependencies() const noexcept {
        return _parameter_dependencies;
    }

  private:
    std::shared_ptr<ThermalNetwork> _network;
    std::shared_ptr<Parameters> _parameters;
    std::vector<std::shared_ptr<Formula>> _formulas;
    std::unordered_map<std::string, std::vector<std::shared_ptr<Formula>>>
        _parameter_dependencies;
    std::optional<std::uint64_t> _validated_structure_version;
    std::optional<std::uint64_t> _compiled_structure_version;
};

}  // namespace pycanha
