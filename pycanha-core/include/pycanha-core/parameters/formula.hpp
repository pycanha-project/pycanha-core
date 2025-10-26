#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/parameters.hpp"

namespace pycanha {

class Formula {
  public:
    using DependencyList = std::vector<std::string>;

    Formula(const Formula&) = default;
    Formula& operator=(const Formula&) = default;
    Formula(Formula&&) noexcept = default;
    Formula& operator=(Formula&&) noexcept = default;
    virtual ~Formula() = default;

    explicit Formula(ThermalEntity& entity) : _entity(entity.clone()) {
        if (_entity == nullptr) {
            throw std::invalid_argument("Formula requires a valid entity");
        }
    }

    explicit Formula(std::shared_ptr<ThermalEntity> entity)
        : _entity(std::move(entity)) {
        if (_entity == nullptr) {
            throw std::invalid_argument("Formula requires a valid entity");
        }
    }

    bool operator==(const Formula& other) const {
        return entity().is_same_as(other.entity());
    }

    struct Hash {
        [[nodiscard]] std::size_t operator()(const Formula& formula) const {
            return std::hash<std::string>()(
                formula.entity().string_representation());
        }
    };

    [[nodiscard]] ThermalEntity& entity() noexcept { return *_entity; }
    [[nodiscard]] const ThermalEntity& entity() const noexcept {
        return *_entity;
    }

    [[nodiscard]] const DependencyList& parameter_dependencies()
        const noexcept {
        return _dependencies;
    }

    virtual void compile_formula() = 0;
    virtual void apply_formula() = 0;
    virtual void apply_compiled_formula() = 0;

    [[nodiscard]] virtual std::unique_ptr<Formula> clone() const = 0;

    [[nodiscard]] virtual double get_value() const = 0;
    [[nodiscard]] virtual std::vector<double>* get_derivative_values() = 0;

  protected:
    DependencyList& mutable_dependencies() noexcept { return _dependencies; }

    [[nodiscard]] double* entity_data_ptr() noexcept { return _entity_data; }
    void set_entity_data_ptr(double* ptr) noexcept { _entity_data = ptr; }

  private:
    std::shared_ptr<ThermalEntity> _entity;
    DependencyList _dependencies;
    double* _entity_data{nullptr};
};

class ParameterFormula final : public Formula {
  public:
    ParameterFormula(ThermalEntity& entity, Parameters& parameters,
                     std::string parameter_name)
        : Formula(entity),
          _parameters(&parameters),
          _parameter_name(std::move(parameter_name)) {
        if (_parameters == nullptr) {
            throw std::invalid_argument(
                "ParameterFormula requires parameter storage");
        }
        mutable_dependencies().push_back(_parameter_name);
    }

    ParameterFormula(std::shared_ptr<ThermalEntity> entity,
                     Parameters& parameters, std::string parameter_name)
        : Formula(std::move(entity)),
          _parameters(&parameters),
          _parameter_name(std::move(parameter_name)) {
        if (_parameters == nullptr) {
            throw std::invalid_argument(
                "ParameterFormula requires parameter storage");
        }
        mutable_dependencies().push_back(_parameter_name);
    }

    void compile_formula() override {
        auto* entity_ptr = entity().get_value_ref();
        if (entity_ptr == nullptr) {
            throw std::runtime_error(
                "ParameterFormula could not obtain entity pointer");
        }
        set_entity_data_ptr(entity_ptr);

        const auto parameter = _parameters->get_parameter(_parameter_name);
        if (std::get_if<double>(&parameter) == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }

        auto* parameter_ptr =
            static_cast<double*>(_parameters->get_value_ptr(_parameter_name));
        if (parameter_ptr == nullptr) {
            throw std::runtime_error(
                "ParameterFormula could not obtain parameter pointer");
        }
        _parameter_data = parameter_ptr;
    }

    void apply_formula() override {
        const auto parameter = _parameters->get_parameter(_parameter_name);
        const auto* parameter_value = std::get_if<double>(&parameter);
        if (parameter_value == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }
        entity().set_value(*parameter_value);
    }

    void apply_compiled_formula() override {
        if (entity_data_ptr() == nullptr || _parameter_data == nullptr) {
            throw std::runtime_error(
                "ParameterFormula needs to be compiled before applying");
        }
        *entity_data_ptr() = *_parameter_data;
    }

    [[nodiscard]] double get_value() const override {
        const auto parameter = _parameters->get_parameter(_parameter_name);
        const auto* parameter_value = std::get_if<double>(&parameter);
        if (parameter_value == nullptr) {
            throw std::runtime_error(
                "ParameterFormula expects parameter to hold a double value");
        }
        return *parameter_value;
    }

    [[nodiscard]] std::vector<double>* get_derivative_values() override {
        return nullptr;
    }

    [[nodiscard]] std::unique_ptr<Formula> clone() const override {
        return std::make_unique<ParameterFormula>(*this);
    }

  private:
    Parameters* _parameters;
    std::string _parameter_name;
    double* _parameter_data{nullptr};
};

class ValueFormula final : public Formula {
  public:
    explicit ValueFormula(ThermalEntity& entity)
        : Formula(entity), _value(entity.get_value()) {}

    explicit ValueFormula(std::shared_ptr<ThermalEntity> entity)
        : Formula(std::move(entity)), _value(this->entity().get_value()) {}

    void compile_formula() override {
        auto* entity_ptr = entity().get_value_ref();
        if (entity_ptr == nullptr) {
            throw std::runtime_error(
                "ValueFormula could not obtain entity pointer");
        }
        set_entity_data_ptr(entity_ptr);
    }

    void apply_formula() override { entity().set_value(_value); }

    void apply_compiled_formula() override {
        if (entity_data_ptr() == nullptr) {
            throw std::runtime_error(
                "ValueFormula needs to be compiled before applying");
        }
        *entity_data_ptr() = _value;
    }

    [[nodiscard]] double get_value() const override { return _value; }

    [[nodiscard]] std::vector<double>* get_derivative_values() override {
        return &_derivatives;
    }

    [[nodiscard]] std::unique_ptr<Formula> clone() const override {
        return std::make_unique<ValueFormula>(*this);
    }

    void set_value(double value) noexcept { _value = value; }

  private:
    double _value{0.0};
    std::vector<double> _derivatives;
};

}  // namespace pycanha
