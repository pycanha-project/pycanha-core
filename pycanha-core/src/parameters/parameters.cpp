#include "pycanha-core/parameters/parameters.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

template <typename T>
struct IsMatrixType : std::false_type {};

template <typename Scalar>
struct IsMatrixType<
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    : std::true_type {};

template <typename T>
constexpr bool is_matrix_type_v = IsMatrixType<T>::value;

[[nodiscard]] std::string canonicalize_parameter_key(std::string_view name) {
    std::string canonical;
    canonical.resize(name.size());

    std::ranges::transform(name, canonical.begin(), [](const char ch) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    });

    return canonical;
}

[[nodiscard]] bool is_reserved_user_parameter_name(
    std::string_view name) noexcept {
    return canonicalize_parameter_key(name) == "time";
}

[[nodiscard]] bool is_entity_like_parameter_name(std::string_view name) {
    ThermalNetwork network;
    return Entity::from_string(network, name).has_value();
}

[[nodiscard]] bool is_valid_user_parameter_name(std::string_view name) {
    const auto canonical_name = canonicalize_parameter_key(name);
    if (canonical_name.empty()) {
        return false;
    }

    if (is_reserved_user_parameter_name(name)) {
        return false;
    }

    return !is_entity_like_parameter_name(name);
}

// Lower-case names a table cannot take: the entity tokens, and every
// function and constant the expression parser knows, since `name(x)` would
// call the built-in rather than the table.
[[nodiscard]] bool is_reserved_table_name(const std::string& canonical_name) {
    static const std::unordered_set<std::string> reserved = {
        "t", "c", "qs", "qa", "qe", "qi", "qr", "gl", "gr", "time",
        // Functions of one argument.
        "sin", "cos", "tan", "cot", "csc", "sec", "asin", "arcsin", "acos",
        "arccos", "atan", "arctan", "asec", "arcsec", "acsc", "arccsc", "acot",
        "arccot", "sinh", "cosh", "tanh", "coth", "sech", "csch", "asinh",
        "arcsinh", "acosh", "arccosh", "atanh", "arctanh", "asech", "arcsech",
        "acoth", "arccoth", "acsch", "arccsch", "gamma", "sqrt", "abs", "sign",
        "exp", "erf", "erfc", "loggamma", "lambertw", "dirichlet_eta", "floor",
        "ceiling", "truncate", "ln", "log", "zeta", "primepi", "primorial",
        // Functions of two or more arguments.
        "pow", "beta", "lowergamma", "uppergamma", "polygamma",
        "kronecker_delta", "atan2", "max", "min", "levi_civita",
        // Relations and logic.
        "eq", "equality", "ne", "unequality", "ge", "greaterthan", "gt",
        "strictgreaterthan", "le", "lessthan", "lt", "strictlessthan", "not",
        "xor", "xnor", "and", "or", "nand", "nor",
        // Constants.
        "e", "eulergamma", "catalan", "goldenratio", "pi", "i", "oo", "inf",
        "zoo", "nan", "true", "false"};
    return reserved.contains(canonical_name);
}

// A name the expression parser reads as one identifier.
[[nodiscard]] bool is_identifier(std::string_view name) noexcept {
    if (name.empty() ||
        (std::isdigit(static_cast<unsigned char>(name.front())) != 0)) {
        return false;
    }
    return std::ranges::all_of(name, [](const char ch) {
        return (std::isalnum(static_cast<unsigned char>(ch)) != 0) ||
               (ch == '_');
    });
}

}  // namespace

struct Parameters::ParameterSize {
    template <typename T>
    std::size_t operator()(const T& data) const noexcept {
        if constexpr (std::is_same_v<T, std::string>) {
            return data.size() + 1U;
        } else if constexpr (is_matrix_type_v<T>) {
            return static_cast<std::size_t>(data.size()) *
                   sizeof(typename T::Scalar);
        } else {
            return sizeof(T);
        }
    }
};

Parameters::ThermalValue Parameters::missing_parameter_value() {
    return ThermalValue{std::numeric_limits<double>::quiet_NaN()};
}

Parameters::ParameterSlot* Parameters::find_slot(
    const std::string& name) noexcept {
    const auto iterator = _name_to_slot.find(canonicalize_parameter_key(name));
    if (iterator == _name_to_slot.end()) {
        return nullptr;
    }

    return find_slot(to_idx(iterator->second));
}

const Parameters::ParameterSlot* Parameters::find_slot(
    const std::string& name) const noexcept {
    const auto iterator = _name_to_slot.find(canonicalize_parameter_key(name));
    if (iterator == _name_to_slot.end()) {
        return nullptr;
    }

    return find_slot(to_idx(iterator->second));
}

Parameters::ParameterSlot* Parameters::find_slot(Index idx) noexcept {
    if (idx < 0) {
        return nullptr;
    }

    const auto position = to_sizet(idx);
    if (position >= _slots.size()) {
        return nullptr;
    }

    auto* slot = std::addressof(_slots[position]);
    if (!slot->active) {
        return nullptr;
    }

    return slot;
}

const Parameters::ParameterSlot* Parameters::find_slot(
    Index idx) const noexcept {
    if (idx < 0) {
        return nullptr;
    }

    const auto position = to_sizet(idx);
    if (position >= _slots.size()) {
        return nullptr;
    }

    const auto* slot = std::addressof(_slots[position]);
    if (!slot->active) {
        return nullptr;
    }

    return slot;
}

void Parameters::invalidate_data_cache() const noexcept {
    _data_cache_dirty = true;
}

void Parameters::note_change(ParameterSlot& slot) noexcept {
    slot.changed = ++_changes;
}

void Parameters::mark_structural_change() noexcept {
    ++_structure_version;
    invalidate_data_cache();
}

bool Parameters::Parameter::is_valid() const noexcept {
    return (_parameters != nullptr) && _parameters->is_parameter_valid(_idx);
}

std::optional<Index> Parameters::Parameter::get_idx() const noexcept {
    if (!is_valid()) {
        return std::nullopt;
    }

    return _idx;
}

std::optional<std::string> Parameters::Parameter::get_name() const {
    if (_parameters == nullptr) {
        return std::nullopt;
    }

    return _parameters->get_parameter_name(_idx);
}

std::optional<Parameters::ThermalValue> Parameters::Parameter::get_value()
    const {
    if (_parameters == nullptr) {
        return std::nullopt;
    }

    return _parameters->get_parameter_optional(_idx);
}

void Parameters::Parameter::set_value(ThermalValue value) {
    if (_parameters == nullptr) {
        return;
    }

    const auto name = get_name();
    if (!name.has_value()) {
        return;
    }

    _parameters->set_parameter(*name, std::move(value));
}

void Parameters::Parameter::rename(std::string new_name) {
    if (_parameters == nullptr) {
        return;
    }

    const auto current_name = get_name();
    if (!current_name.has_value()) {
        return;
    }

    _parameters->rename_parameter(*current_name, std::move(new_name));
}

void Parameters::Parameter::remove() {
    if (_parameters == nullptr) {
        return;
    }

    const auto name = get_name();
    if (!name.has_value()) {
        return;
    }

    _parameters->remove_parameter(*name);
}

void Parameters::add_parameter(std::string name, ThermalValue value) {
    if (_structure_locked) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' cannot be added while parameters "
                           "are structurally locked",
                           name);
        return;
    }

    if (!is_valid_user_parameter_name(name)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' uses a reserved or entity-like "
                           "name",
                           name);
        return;
    }

    if (_name_to_slot.contains(canonicalize_parameter_key(name)) ||
        is_table_name(name)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' already exists", name);
        return;
    }

    insert_slot(std::move(name), std::move(value), /*is_internal=*/false);
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "Parameter '{}' added",
                        _slots.back().name);
}

void Parameters::add_internal_parameter(std::string name, ThermalValue value) {
    if (_name_to_slot.contains(canonicalize_parameter_key(name)) ||
        is_table_name(name)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' already exists", name);
        return;
    }

    insert_slot(std::move(name), std::move(value), /*is_internal=*/true);
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "Internal parameter '{}' added",
                        _slots.back().name);
}

void Parameters::insert_slot(std::string name, ThermalValue value,
                             bool is_internal) {
    _name_to_slot.emplace(canonicalize_parameter_key(name), _slots.size());
    _slots.push_back(ParameterSlot{.name = std::move(name),
                                   .value = std::move(value),
                                   .active = true,
                                   .is_internal = is_internal});
    ++_active_size;
    mark_structural_change();
}

void Parameters::deactivate_slot(ParameterSlot& slot) {
    _name_to_slot.erase(canonicalize_parameter_key(slot.name));
    slot.active = false;
    slot.value = missing_parameter_value();
    slot.is_internal = false;
    --_active_size;
    mark_structural_change();
}

void Parameters::remove_parameter(const std::string& name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' cannot be removed while parameters "
                           "are structurally locked",
                           name);
        return;
    }

    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be removed with "
                           "remove_parameter",
                           name);
        return;
    }

    deactivate_slot(*slot);
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "Parameter '{}' removed", name);
}

void Parameters::remove_internal_parameter(const std::string& name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be removed while "
                           "parameters are structurally locked",
                           name);
        return;
    }

    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' doesn't exist", name);
        return;
    }

    if (!slot->is_internal) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' is not internal", name);
        return;
    }

    if (is_table_name(name)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' is the value of a table; remove "
                           "the table instead",
                           name);
        return;
    }

    deactivate_slot(*slot);
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(),
                        "Internal parameter '{}' removed", name);
}

void Parameters::rename_parameter(const std::string& current_name,
                                  std::string new_name) {
    if (_structure_locked) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' cannot be renamed while parameters "
                           "are structurally locked",
                           current_name);
        return;
    }

    auto* slot = find_slot(current_name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", current_name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be renamed",
                           current_name);
        return;
    }

    if (!is_valid_user_parameter_name(new_name)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' uses a reserved or entity-like "
                           "name",
                           new_name);
        return;
    }

    const auto current_canonical_name = canonicalize_parameter_key(slot->name);
    const auto new_canonical_name = canonicalize_parameter_key(new_name);

    if ((new_canonical_name != current_canonical_name) &&
        (_name_to_slot.contains(new_canonical_name) ||
         is_table_name(new_name))) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' already exists", new_name);
        return;
    }

    const auto slot_index = _name_to_slot.at(current_canonical_name);
    _name_to_slot.erase(current_canonical_name);
    _name_to_slot.emplace(new_canonical_name, slot_index);
    slot->name = std::move(new_name);
    mark_structural_change();

    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "Parameter '{}' renamed",
                        slot->name);
}

Parameters::Parameter Parameters::get_parameter_handle(
    const std::string& name) noexcept {
    const auto index = find_idx(name);
    if (!index.has_value()) {
        return {};
    }

    return {this, *index};
}

Parameters::ThermalValue Parameters::get_parameter(
    const std::string& name) const {
    const auto parameter = get_parameter_optional(name);
    if (parameter.has_value()) {
        return *parameter;
    }

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Parameter '{}' doesn't exist",
                       name);

    return missing_parameter_value();
}

std::optional<Parameters::ThermalValue> Parameters::get_parameter_optional(
    const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->value;
}

std::optional<Parameters::ThermalValue> Parameters::get_parameter_optional(
    Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->value;
}

void Parameters::set_parameter(const std::string& name, ThermalValue value) {
    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return;
    }

    if (slot->is_internal) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' cannot be modified with "
                           "set_parameter",
                           name);
        return;
    }

    if (slot->value.index() != value.index()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' type mismatch", name);
        return;
    }

    bool changed = true;
    std::visit(
        [&](auto& existing) {
            using ExistingType = std::decay_t<decltype(existing)>;
            auto* incoming = std::get_if<ExistingType>(&value);
            if (incoming == nullptr) {
                changed = false;
                return;
            }

            if constexpr (is_matrix_type_v<ExistingType>) {
                if ((existing.rows() != incoming->rows()) ||
                    (existing.cols() != incoming->cols())) {
                    SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                                       "Parameter '{}' shape mismatch", name);
                    return;
                }
            }

            // A double that keeps its value is not a change.
            if constexpr (std::is_same_v<ExistingType, double>) {
                changed = existing != *incoming;
            }
            existing = std::move(*incoming);
        },
        slot->value);

    if (changed) {
        note_change(*slot);
    }
    invalidate_data_cache();
}

void Parameters::set_internal_parameter(const std::string& name,
                                        ThermalValue value) {
    auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' doesn't exist", name);
        return;
    }

    if (!slot->is_internal) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Parameter '{}' is not internal", name);
        return;
    }

    if (slot->value.index() != value.index()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Internal parameter '{}' type mismatch", name);
        return;
    }

    bool changed = true;
    std::visit(
        [&](auto& existing) {
            using ExistingType = std::decay_t<decltype(existing)>;
            auto* incoming = std::get_if<ExistingType>(&value);
            if (incoming == nullptr) {
                changed = false;
                return;
            }

            if constexpr (is_matrix_type_v<ExistingType>) {
                if ((existing.rows() != incoming->rows()) ||
                    (existing.cols() != incoming->cols())) {
                    SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                                       "Internal parameter '{}' shape "
                                       "mismatch",
                                       name);
                    return;
                }
            }

            // A double that keeps its value is not a change.
            if constexpr (std::is_same_v<ExistingType, double>) {
                changed = existing != *incoming;
            }
            existing = std::move(*incoming);
        },
        slot->value);

    if (changed) {
        note_change(*slot);
    }
    invalidate_data_cache();
}

// The print_* helpers below write to stdout rather than through the logger.
// The caller asked for output, so it must appear whatever the log thresholds
// are set to, and a dump of a value is not a record of anything happening.

void Parameters::print_parameter(const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        std::cout << "Parameter '" << name << "' doesn't exist\n";
        return;
    }

    std::cout << name << " = ";
    std::visit(
        [](const auto& value) {
            if constexpr (is_matrix_type_v<std::decay_t<decltype(value)>>) {
                std::cout << '\n' << value;
            } else {
                std::cout << value;
            }
        },
        slot->value);
    std::cout << '\n';
}

double* Parameters::get_double_ptr(Index idx) {
    auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::get_if<double>(&slot->value);
}

const double* Parameters::get_double_ptr(Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return nullptr;
    }

    return std::get_if<double>(&slot->value);
}

const double* detail::ParameterAccess::value(const Parameters& parameters,
                                             Index idx) {
    return parameters.get_double_ptr(idx);
}

const std::uint64_t* detail::ParameterAccess::changes(
    const Parameters& parameters) noexcept {
    return &parameters._changes;
}

const std::uint64_t* detail::ParameterAccess::changed(
    const Parameters& parameters, Index idx) {
    const auto* slot = parameters.find_slot(idx);
    return slot == nullptr ? nullptr : &slot->changed;
}

const std::uint64_t* detail::ParameterAccess::changed(
    const Table& table) noexcept {
    return &table._changed;
}

std::optional<Index> Parameters::find_idx(
    const std::string& name) const noexcept {
    const auto iterator = _name_to_slot.find(canonicalize_parameter_key(name));
    if (iterator == _name_to_slot.end()) {
        return std::nullopt;
    }

    const auto index = to_idx(iterator->second);
    if (find_slot(index) == nullptr) {
        return std::nullopt;
    }

    return index;
}

std::optional<Index> Parameters::get_idx(const std::string& name) const {
    const auto index = find_idx(name);
    if (!index.has_value()) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
    }

    return index;
}

bool Parameters::is_internal_parameter(const std::string& name) const noexcept {
    const auto* slot = find_slot(name);
    return (slot != nullptr) && slot->is_internal;
}

bool Parameters::is_parameter_valid(Index idx) const noexcept {
    return find_slot(idx) != nullptr;
}

std::optional<std::string> Parameters::get_parameter_name(Index idx) const {
    const auto* slot = find_slot(idx);
    if (slot == nullptr) {
        return std::nullopt;
    }

    return slot->name;
}

std::size_t Parameters::get_size_of_parameter(const std::string& name) const {
    const auto* slot = find_slot(name);
    if (slot == nullptr) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                           "Parameter '{}' doesn't exist", name);
        return 0U;
    }

    return std::visit(ParameterSize{}, slot->value);
}

void Parameters::lock_structure() noexcept { _structure_locked = true; }

void Parameters::unlock_structure() noexcept { _structure_locked = false; }

bool Parameters::is_structure_locked() const noexcept {
    return _structure_locked;
}

std::uint64_t Parameters::get_structure_version() const noexcept {
    return _structure_version;
}

bool Parameters::contains(const std::string& name) const noexcept {
    return _name_to_slot.contains(canonicalize_parameter_key(name));
}

std::size_t Parameters::size() const noexcept { return _active_size; }

const Parameters::ParametersDict& Parameters::data() const {
    if (_data_cache_dirty) {
        _data_cache.clear();
        for (const auto& slot : _slots) {
            if (!slot.active) {
                continue;
            }

            _data_cache.emplace(slot.name, slot.value);
        }

        _data_cache_dirty = false;
    }

    return _data_cache;
}

bool Parameters::is_table_name(std::string_view name) const {
    return _tables.contains(canonicalize_parameter_key(name));
}

void Parameters::check_table_change(std::string_view name) const {
    if (_structure_locked) {
        throw std::runtime_error("Table '" + std::string(name) +
                                 "' cannot be added, removed or replaced "
                                 "while parameters are structurally locked");
    }
}

Table& Parameters::existing_table(const std::string& name) {
    const auto iterator = _tables.find(canonicalize_parameter_key(name));
    if (iterator == _tables.end()) {
        throw std::out_of_range("Table '" + name + "' does not exist");
    }
    return iterator->second;
}

void Parameters::publish_table(Table& table) {
    if (table._evaluated_at.empty()) {
        return;
    }
    const auto argument_idx = find_idx(table._evaluated_at);
    if (!argument_idx.has_value() ||
        (get_double_ptr(*argument_idx) == nullptr)) {
        throw std::invalid_argument("Table '" + table._name +
                                    "' is evaluated at '" +
                                    table._evaluated_at +
                                    "', which is not a parameter storing a "
                                    "double");
    }
    table._argument_idx = *argument_idx;
    const double value = table.evaluate(*get_double_ptr(*argument_idx));
    insert_slot(table._name, value, /*is_internal=*/true);
    table._published_idx = to_idx(_slots.size() - 1U);
}

void Parameters::unpublish_table(Table& table) {
    auto* slot = find_slot(table._published_idx);
    if (slot != nullptr) {
        deactivate_slot(*slot);
    }
    table._argument_idx = -1;
    table._published_idx = -1;
}

void Parameters::add_table(std::string name, LookupTable1D lookup_table,
                           std::string evaluated_at) {
    check_table_change(name);
    if (!is_identifier(name)) {
        throw std::invalid_argument("Table name '" + name +
                                    "' is not an identifier");
    }
    const auto canonical_name = canonicalize_parameter_key(name);
    if (is_reserved_table_name(canonical_name) ||
        is_entity_like_parameter_name(name)) {
        throw std::invalid_argument(
            "Table name '" + name +
            "' is reserved: it is an entity token, a built-in function or "
            "a constant");
    }
    if (_tables.contains(canonical_name)) {
        throw std::invalid_argument("Table '" + name + "' already exists");
    }
    if (_name_to_slot.contains(canonical_name)) {
        throw std::invalid_argument("Table name '" + name +
                                    "' is already a parameter");
    }

    Table added(std::move(name), std::move(lookup_table), Origin::user,
                std::move(evaluated_at));
    publish_table(added);
    _tables.emplace(canonical_name, std::move(added));
    mark_structural_change();
}

TableGeneration Parameters::add_generated_table(Origin origin, std::string name,
                                                LookupTable1D lookup_table,
                                                std::string evaluated_at) {
    if (origin == Origin::user) {
        throw std::invalid_argument(
            "Generated tables need a generator origin; add user tables with "
            "add_table");
    }
    const auto canonical_name = canonicalize_parameter_key(name);
    const auto iterator = _tables.find(canonical_name);
    if (iterator == _tables.end()) {
        add_table(std::move(name), std::move(lookup_table),
                  std::move(evaluated_at));
        _tables.at(canonical_name)._origin = origin;
        return TableGeneration::added;
    }

    auto& existing = iterator->second;
    if (existing._origin == Origin::user) {
        SPDLOG_LOGGER_WARN(get_logger(),
                           "Generated table '{}' ({}) not written: a user "
                           "table with that name stays",
                           name, to_string(origin));
        return TableGeneration::kept_user;
    }

    check_table_change(name);
    if (existing._origin != origin) {
        SPDLOG_LOGGER_WARN(
            get_logger(), "Generated table '{}' ({}) replaces a {} table", name,
            to_string(origin), to_string(existing._origin));
    }
    Table replacement(existing._name, std::move(lookup_table), origin,
                      std::move(evaluated_at));
    unpublish_table(existing);
    try {
        publish_table(replacement);
    } catch (...) {
        // A refused `evaluated_at` leaves the existing table as it was.
        publish_table(existing);
        throw;
    }
    existing = std::move(replacement);
    mark_structural_change();
    return TableGeneration::overwritten;
}

bool Parameters::remove_table(const std::string& name) {
    const auto iterator = _tables.find(canonicalize_parameter_key(name));
    if (iterator == _tables.end()) {
        SPDLOG_LOGGER_INFO(get_logger(), "Table '{}' does not exist", name);
        return false;
    }
    check_table_change(name);
    unpublish_table(iterator->second);
    _tables.erase(iterator);
    mark_structural_change();
    return true;
}

bool Parameters::has_table(const std::string& name) const {
    return find_table(name) != nullptr;
}

const Table* Parameters::find_table(const std::string& name) const {
    const auto iterator = _tables.find(canonicalize_parameter_key(name));
    return iterator == _tables.end() ? nullptr : &iterator->second;
}

const Table& Parameters::table(const std::string& name) const {
    const auto* found = find_table(name);
    if (found == nullptr) {
        throw std::out_of_range("Table '" + name + "' does not exist");
    }
    return *found;
}

std::vector<std::string> Parameters::table_names() const {
    std::vector<std::string> names;
    names.reserve(_tables.size());
    std::ranges::transform(
        _tables, std::back_inserter(names),
        [](const auto& entry) { return entry.second.name(); });
    std::ranges::sort(names);
    return names;
}

void Parameters::set_table_values(const std::string& name,
                                  std::span<const double> values) {
    auto& written = existing_table(name);
    written._lookup_table.assign_y(values);
    written.update_slopes();
    written._changed = ++_changes;
}

void Parameters::set_table_x(const std::string& name,
                             std::span<const double> x) {
    auto& written = existing_table(name);
    written._lookup_table.assign_x(x);
    written.update_slopes();
    written._changed = ++_changes;
}

void Parameters::update_evaluated_tables() {
    for (const auto& entry : _tables) {
        const auto& evaluated = entry.second;
        if (evaluated._evaluated_at.empty()) {
            continue;
        }
        const double* argument = get_double_ptr(evaluated._argument_idx);
        auto* slot = find_slot(evaluated._published_idx);
        double* published =
            slot == nullptr ? nullptr : std::get_if<double>(&slot->value);
        if ((argument == nullptr) || (published == nullptr)) {
            throw std::runtime_error(
                "Table '" + evaluated._name + "' is evaluated at '" +
                evaluated._evaluated_at + "', which no longer exists");
        }
        const double value =
            detail::table_value(evaluated.view(time_ptr()), *argument);
        if (value != *published) {
            *published = value;
            note_change(*slot);
        }
    }
}

std::vector<TableRange> Parameters::table_ranges() const {
    std::vector<TableRange> ranges;
    for (const auto& entry : _tables) {
        const auto& queried = entry.second;
        const auto& range = queried._range;
        if (!range.out_of_range) {
            continue;
        }
        const auto& x = queried.lookup_table().x();
        ranges.push_back(TableRange{.table = queried.name(),
                                    .first_time = range.first_time,
                                    .first_argument = range.first_argument,
                                    .min_argument = range.min_argument,
                                    .max_argument = range.max_argument,
                                    .x_min = x(0),
                                    .x_max = x(x.size() - 1)});
    }
    std::ranges::sort(ranges, {}, &TableRange::table);
    return ranges;
}

void Parameters::reset_table_ranges() const noexcept {
    for (const auto& entry : _tables) {
        entry.second._range.reset();
    }
}

const double* Parameters::time_ptr() const {
    const auto index = find_idx("time");
    return index.has_value() ? get_double_ptr(*index) : nullptr;
}

}  // namespace pycanha
