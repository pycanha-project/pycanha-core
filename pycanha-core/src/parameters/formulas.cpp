#include "pycanha-core/parameters/formulas.hpp"

#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <symengine/basic.h>
#include <symengine/symbol.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

void DerivativeParameterRegistry::add_parameter(
    const std::string& parameter_name) {
    if (_parameters == nullptr) {
        throw std::runtime_error(
            "Derivative parameter registry requires parameter storage");
    }
    if (!_parameters->contains(parameter_name)) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        throw std::invalid_argument("Unknown parameter '" + parameter_name +
                                    "' cannot be added to the derivative set");
    }

    if (contains(*resolved_name)) {
        return;
    }

    // TODO(formulas): validate that derivative-selected parameters are read
    // only by formulas that read parameters alone.
    _parameter_names.push_back(*resolved_name);
}

bool DerivativeParameterRegistry::remove_parameter(
    const std::string& parameter_name) noexcept {
    if (_parameters == nullptr) {
        return false;
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        return false;
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        return false;
    }

    const auto iterator = std::ranges::find(_parameter_names, *resolved_name);
    if (iterator == _parameter_names.end()) {
        return false;
    }

    _parameter_names.erase(iterator);
    return true;
}

bool DerivativeParameterRegistry::contains(
    const std::string& parameter_name) const noexcept {
    if (_parameters == nullptr) {
        return false;
    }

    const auto parameter_idx = _parameters->get_idx(parameter_name);
    if (!parameter_idx.has_value()) {
        return false;
    }

    const auto resolved_name = _parameters->get_parameter_name(*parameter_idx);
    if (!resolved_name.has_value()) {
        return false;
    }

    return std::ranges::find(_parameter_names, *resolved_name) !=
           _parameter_names.end();
}

namespace {

[[nodiscard]] bool is_load(EntityType type) noexcept {
    return type == EntityType::qs || type == EntityType::qa ||
           type == EntityType::qe || type == EntityType::qi ||
           type == EntityType::qr;
}

[[nodiscard]] std::shared_ptr<ThermalNetwork> ensure_network(
    const std::shared_ptr<ThermalNetwork>& network) {
    if (network == nullptr) {
        throw std::runtime_error("Formulas requires a thermal network");
    }
    return network;
}

[[nodiscard]] std::shared_ptr<Parameters> ensure_parameters(
    const std::shared_ptr<Parameters>& parameters) {
    if (parameters == nullptr) {
        throw std::runtime_error("Formulas requires parameters");
    }
    return parameters;
}

// Why `formula` cannot run, e.g. a node it reads was removed; nothing when it
// can.
[[nodiscard]] std::optional<std::string> formula_problem(
    const Formula& formula) {
    if (!formula.entity().exists()) {
        return "its target does not exist";
    }
    try {
        static_cast<void>(formula.get_value());
    } catch (const std::exception& exception) {
        return exception.what();
    }
    return std::nullopt;
}

// The generated formulas that met an existing formula or were refused: each
// one is logged at DEBUG, and the first few are kept for one summary WARN.
class CollisionLog {
  public:
    void note(const Entity& target, const std::string& what) {
        // get_logger() stays inside the macro: a build with DEBUG compiled
        // away must not leave an unused local behind.
        SPDLOG_LOGGER_DEBUG(get_logger(), "Generated formula for '{}': {}",
                            target.string_representation(), what);
        ++_count;
        if (_count <= listed) {
            _first += _first.empty() ? "" : "; ";
            _first += target.string_representation() + " (" + what + ")";
        }
    }

    [[nodiscard]] bool empty() const noexcept { return _count == 0U; }
    [[nodiscard]] const std::string& first() const noexcept { return _first; }

  private:
    static constexpr std::size_t listed = 5U;
    std::size_t _count{0U};
    std::string _first;
};

// Appends `formula` to `list` unless it is already its last entry: one
// formula may read the same input under two spellings (GL(1,2) and GL(2,1)).
void append_once(std::vector<std::shared_ptr<Formula>>& list,
                 const std::shared_ptr<Formula>& formula) {
    if (list.empty() || list.back() != formula) {
        list.push_back(formula);
    }
}

}  // namespace

Formulas::Formulas() : _network(nullptr), _parameters(nullptr) {}

Formulas::Formulas(std::shared_ptr<ThermalNetwork> network,
                   std::shared_ptr<Parameters> parameters)
    : _network(std::move(network)),
      _parameters(std::move(parameters)),
      _parameters_with_derivatives(_parameters) {}

void Formulas::associate(std::shared_ptr<ThermalNetwork> network,
                         std::shared_ptr<Parameters> parameters) {
    _network = std::move(network);
    _parameters = std::move(parameters);
    _parameters_with_derivatives.associate(_parameters);
    _validated.reset();
    _compiled.reset();
}

Entity Formulas::resolve_entity(std::string_view entity) const {
    const auto associated_network = ensure_network(_network);
    const auto resolved = Entity::from_string(*associated_network, entity);
    if (!resolved.has_value()) {
        throw std::invalid_argument("Unknown formula target '" +
                                    std::string(entity) + "'");
    }

    return *resolved;
}

std::optional<Entity> Formulas::try_resolve_entity(
    std::string_view entity) const {
    if (_network == nullptr) {
        return std::nullopt;
    }
    return Entity::from_string(*_network, entity);
}

std::shared_ptr<Formula> Formulas::create_formula(
    Entity entity, const std::string& formula_string) {
    const auto associated_network = ensure_network(_network);
    auto parameter_storage = ensure_parameters(_parameters);

    // Parsed once here; the formula is built from this parse. Symbols that
    // are not parameters are resolved to entities by ExpressionFormula.
    auto parsed = detail::parse_expression(formula_string);
    if (_temperature_variable_names != nullptr) {
        for (const auto& symbol : parsed.symbols) {
            const auto& symbol_name = symbol->get_name();
            if (!parameter_storage->contains(symbol_name) &&
                _temperature_variable_names->contains(symbol_name)) {
                throw std::invalid_argument(
                    "TemperatureVariable '" + symbol_name +
                    "' in formulas is not implemented yet");
            }
        }
    }

    return std::make_shared<ExpressionFormula>(
        entity, *parameter_storage, formula_string, std::move(parsed),
        associated_network.get());
}

void Formulas::set_temperature_variable_names(
    const std::unordered_set<std::string>* names) noexcept {
    _temperature_variable_names = names;
}

Formula& Formulas::add_formula(const Entity& entity, double value) {
    auto formula = std::make_shared<ExpressionFormula>(entity, value);
    add_formula(formula);
    return *formula;
}

Formula& Formulas::add_formula(std::string_view entity, double value) {
    return add_formula(resolve_entity(entity), value);
}

Formula& Formulas::add_formula(const Entity& entity,
                               const std::string& formula_string) {
    auto formula = create_formula(entity, formula_string);
    add_formula(formula);
    return *formula;
}

Formula& Formulas::add_formula(std::string_view entity,
                               const std::string& formula_string) {
    return add_formula(resolve_entity(entity), formula_string);
}

CallableFormula& Formulas::add_callable_formula(
    const Entity& entity, CallableFormula::Function function) {
    auto formula =
        std::make_shared<CallableFormula>(entity, std::move(function));
    add_formula(formula);
    return *formula;
}

CallableFormula& Formulas::add_callable_formula(
    std::string_view entity, CallableFormula::Function function) {
    return add_callable_formula(resolve_entity(entity), std::move(function));
}

void Formulas::add_formula(const Formula& formula) {
    auto clone = formula.clone();
    auto shared = std::shared_ptr<Formula>(std::move(clone));
    add_formula(shared);
}

void Formulas::add_formula(const std::shared_ptr<Formula>& formula) {
    insert(formula, FormulaOrigin::user);
}

void Formulas::insert(const std::shared_ptr<Formula>& formula,
                      FormulaOrigin origin) {
    if (formula == nullptr) {
        throw std::invalid_argument("Cannot add a null formula");
    }

    const auto& target = formula->entity();
    if (_positions.contains(target)) {
        throw std::invalid_argument("A formula for entity '" +
                                    target.string_representation() +
                                    "' already exists");
    }

    // Allocate the storage of the target, and of every load the formula
    // reads, now: a load is empty until a value pointer into it is taken, so
    // allocating here keeps that allocation out of a later solve.
    if (target.exists() && target.writable()) {
        static_cast<void>(target.get_value_ref());
    }
    for (const auto& input : formula->entity_dependencies()) {
        if (is_load(input.type()) && input.exists()) {
            static_cast<void>(input.get_value_ref());
        }
    }

    const auto position = _positions.emplace(target, _formulas.size()).first;
    try {
        _formulas.push_back(formula);
    } catch (...) {
        _positions.erase(position);
        throw;
    }
    formula->set_origin(origin);
    ++_formulas_version;
}

GenerationReport Formulas::add_generated_formulas(
    FormulaOrigin origin, std::span<const GeneratedFormula> formulas) {
    if (origin == FormulaOrigin::user) {
        throw std::invalid_argument(
            "Generated formulas need a generator origin; add user formulas "
            "with add_formula");
    }

    GenerationReport report;
    CollisionLog collisions;
    for (const auto& generated : formulas) {
        const auto& target = generated.target;
        const auto existing = find(target);
        if (existing != nullptr && existing->origin() == FormulaOrigin::user) {
            ++report.kept_user;
            collisions.note(target, "the user formula stays");
            continue;
        }

        std::shared_ptr<Formula> formula;
        try {
            formula = create_formula(target, generated.expression);
        } catch (const std::exception& exception) {
            report.failed.push_back(
                FailedFormula{.target = target, .message = exception.what()});
            collisions.note(target, exception.what());
            continue;
        }

        if (existing == nullptr) {
            ++report.added;
        } else {
            ++report.replaced;
            collisions.note(target,
                            "replaces a " +
                                std::string(to_string(existing->origin())) +
                                " formula");
            static_cast<void>(remove_formula(target));
        }
        insert(formula, origin);
    }

    if (collisions.empty()) {
        SPDLOG_LOGGER_INFO(get_logger(), "Added {} generated formulas ({})",
                           report.added, to_string(origin));
        return report;
    }
    SPDLOG_LOGGER_WARN(get_logger(),
                       "Generated formulas ({}): {} added, {} replaced a "
                       "generated formula, {} dropped for a user formula, {} "
                       "refused. First: {}",
                       to_string(origin), report.added, report.replaced,
                       report.kept_user, report.failed.size(),
                       collisions.first());
    return report;
}

std::size_t Formulas::remove_formulas(FormulaOrigin origin) {
    std::size_t removed = 0U;
    for (auto& formula : _formulas) {
        if (formula == nullptr || formula->origin() != origin) {
            continue;
        }
        _positions.erase(formula->entity());
        formula.reset();
        ++removed;
    }

    if (removed > 0U) {
        _holes += removed;
        ++_formulas_version;
        compact();
    }
    SPDLOG_LOGGER_DEBUG(get_logger(), "Removed {} formulas of origin {}",
                        removed, to_string(origin));
    return removed;
}

bool Formulas::contains(const Entity& entity) const noexcept {
    return _positions.contains(entity);
}

bool Formulas::contains(std::string_view entity) const {
    const auto resolved = try_resolve_entity(entity);
    return resolved.has_value() && contains(*resolved);
}

std::shared_ptr<Formula> Formulas::find(const Entity& entity) const {
    const auto position = _positions.find(entity);
    if (position == _positions.end()) {
        return nullptr;
    }
    return _formulas[position->second];
}

std::shared_ptr<Formula> Formulas::find(std::string_view entity) const {
    const auto resolved = try_resolve_entity(entity);
    return resolved.has_value() ? find(*resolved) : nullptr;
}

bool Formulas::remove_formula(const Entity& entity) noexcept {
    const auto position = _positions.find(entity);
    if (position == _positions.end()) {
        // noexcept function: compose without std::format and use the
        // non-formatting log path (see pycanha::log_noexcept).
        log_noexcept(spdlog::level::info, "Formula '" +
                                              entity.string_representation() +
                                              "' was not present for removal");
        return false;
    }

    _formulas[position->second].reset();
    _positions.erase(position);
    ++_holes;
    ++_formulas_version;
    return true;
}

void Formulas::compact() const {
    if (_holes == 0U) {
        return;
    }

    std::size_t kept = 0U;
    for (std::size_t position = 0U; position < _formulas.size(); ++position) {
        if (_formulas[position] == nullptr) {
            continue;
        }
        if (position != kept) {
            _formulas[kept] = std::move(_formulas[position]);
            _positions.at(_formulas[kept]->entity()) = kept;
        }
        ++kept;
    }
    _formulas.resize(kept);
    _holes = 0U;
}

FormulaDependencies Formulas::dependencies() const {
    compact();

    FormulaDependencies dependencies;
    for (const auto& formula : _formulas) {
        for (const auto& name : formula->parameter_dependencies()) {
            append_once(dependencies.parameters[name], formula);
        }
        for (const auto& input : formula->entity_dependencies()) {
            append_once(dependencies.entities[input], formula);
        }
    }
    return dependencies;
}

bool Formulas::remove_formula(std::string_view entity) noexcept {
    // Resolve without throwing (resolve_entity throws on unknown targets):
    // an unresolvable target simply means there is nothing to remove.
    if (_network == nullptr) {
        return false;
    }
    const auto resolved = Entity::from_string(*_network, entity);
    if (!resolved.has_value()) {
        // noexcept function: compose without std::format and use the
        // non-formatting log path (see pycanha::log_noexcept).
        log_noexcept(spdlog::level::info, "Formula target '" +
                                              std::string(entity) +
                                              "' was not present for removal");
        return false;
    }
    return remove_formula(*resolved);
}

void Formulas::settle_network_storage() {
    if (_network == nullptr) {
        return;
    }
    // Size the coupling matrices to the nodes and compress them before any
    // value pointer is taken: after that, their values only move when a
    // structure version changes, which makes the compiled state stale.
    _network->synchronize_structure();
    _network->conductive_matrices().compress();
    _network->radiative_matrices().compress();
}

Formulas::Stamp Formulas::current_stamp() const noexcept {
    Stamp stamp;
    stamp.formulas = _formulas_version;
    if (_parameters != nullptr) {
        stamp.parameters = _parameters->get_structure_version();
    }
    if (_network != nullptr) {
        stamp.network = _network->structure_versions();
    }
    return stamp;
}

void Formulas::validate_for_execution() {
    static_cast<void>(ensure_parameters(_parameters));
    settle_network_storage();
    if (is_validation_current()) {
        return;
    }
    compact();

    // Every broken formula is counted, so one error covers them all.
    constexpr std::size_t listed_formulas = 5U;
    std::size_t broken = 0U;
    std::string listing;
    for (const auto& formula : _formulas) {
        const auto problem = formula_problem(*formula);
        if (!problem.has_value()) {
            continue;
        }
        ++broken;
        if (broken <= listed_formulas) {
            listing += "\n  " + formula->entity().string_representation() +
                       ": " + *problem;
        }
    }
    if (broken > 0U) {
        throw std::runtime_error(std::to_string(broken) + " of " +
                                 std::to_string(_formulas.size()) +
                                 " formulas cannot run; remove or fix them. "
                                 "First:" +
                                 listing);
    }

    _validated = current_stamp();
    _compiled.reset();
}

void Formulas::compile_formulas() {
    validate_for_execution();
    if (is_compiled_current()) {
        return;
    }

    for (const auto& formula : _formulas) {
        formula->compile_formula();
    }

    _compiled = current_stamp();
}

void Formulas::apply_formulas() {
    compact();
    if (!debug_formulas) {
        for (const auto& formula : _formulas) {
            formula->apply_formula();
        }
        return;
    }

    // get_logger() is called inside the macro rather than hoisted into a local
    // so that a build with debug records compiled away leaves no unused local
    // behind.
    for (const auto& formula : _formulas) {
        formula->apply_formula();
        SPDLOG_LOGGER_DEBUG(get_logger(), "{} = {}",
                            formula->entity().string_representation(),
                            formula->get_value());
    }
}

void Formulas::apply_compiled_formulas() {
    if (!is_compiled_current()) {
        throw std::runtime_error(
            "Compiled formulas are stale or unavailable for execution");
    }

    if (!debug_formulas) {
        for (const auto& formula : _formulas) {
            formula->apply_compiled_formula();
        }
        return;
    }

    // As in apply_formulas(): keep the logger lookup inside the macro so a
    // build with debug records compiled away leaves no unused local behind.
    for (const auto& formula : _formulas) {
        formula->apply_compiled_formula();
        SPDLOG_LOGGER_DEBUG(get_logger(), "{} = {}",
                            formula->entity().string_representation(),
                            formula->get_value());
    }
}

void Formulas::calculate_derivatives() {
    compact();
    for (const auto& formula : _formulas) {
        if (formula->kind() == FormulaKind::Parameters) {
            formula->calculate_derivatives();
        }
    }
}

void Formulas::lock_parameters_for_execution() {
    auto parameter_storage = ensure_parameters(_parameters);
    if (!is_validation_current()) {
        throw std::runtime_error(
            "Formulas validation is stale; validate before locking "
            "parameters for execution");
    }

    parameter_storage->lock_structure();
}

void Formulas::unlock_parameters() noexcept {
    if (_parameters == nullptr) {
        return;
    }

    _parameters->unlock_structure();
}

bool Formulas::is_validation_current() const noexcept {
    return (_parameters != nullptr) && _validated.has_value() &&
           (*_validated == current_stamp());
}

bool Formulas::is_compiled_current() const noexcept {
    return (_parameters != nullptr) && _compiled.has_value() &&
           (*_compiled == current_stamp());
}

}  // namespace pycanha
