#include "pycanha-core/parameters/formulas.hpp"

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ratio>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/formula_shape.hpp"
#include "pycanha-core/parameters/origin.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/thread_pool.hpp"

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

constexpr std::uint32_t dropped = std::numeric_limits<std::uint32_t>::max();
// Shorter runs are evaluated one formula at a time: a block's loops cost
// more than they save on a few formulas. Measured: blocks lose on runs of
// 2, break even at 3 and win from 4.
constexpr std::uint32_t block_run_minimum = 4U;

[[nodiscard]] std::uint32_t to_u32(std::size_t value) {
    return static_cast<std::uint32_t>(value);
}

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

// Appends `target` to `list` unless it is already its last entry: one
// formula may read the same input under two spellings (GL(1,2) and GL(2,1)).
void append_once(std::vector<Entity>& list, const Entity& target) {
    if (list.empty() || !(list.back() == target)) {
        list.push_back(target);
    }
}

[[nodiscard]] std::string already_exists(const Entity& target) {
    return "A formula for entity '" + target.string_representation() +
           "' already exists";
}

void warn_self_reference(const Entity& target) {
    SPDLOG_LOGGER_WARN(get_logger(),
                       "ExpressionFormula for '{}' references itself on the "
                       "right-hand side",
                       target.string_representation());
}

}  // namespace

// The generated formulas that met an existing formula or were refused: each
// one is logged at DEBUG, and the first few are kept for one summary WARN.
class Formulas::Collisions {
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

    void log(Origin origin, const GenerationReport& report) const {
        if (empty()) {
            SPDLOG_LOGGER_INFO(get_logger(), "Added {} generated formulas ({})",
                               report.added, to_string(origin));
            return;
        }
        SPDLOG_LOGGER_WARN(get_logger(),
                           "Generated formulas ({}): {} added, {} replaced a "
                           "generated formula, {} dropped for a user formula, "
                           "{} refused. First: {}",
                           to_string(origin), report.added, report.replaced,
                           report.kept_user, report.failed.size(), _first);
    }

  private:
    static constexpr std::size_t listed = 5U;
    std::size_t _count{0U};
    std::string _first;
};

// A template call resolved to its shape, with the values of each slot in
// slot order.
struct Formulas::PreparedTemplate {
    std::shared_ptr<const detail::FormulaShape> shape;
    std::vector<std::span<const NodeNum>> nodes;
    std::vector<std::span<const double>> numbers;
    std::size_t count{1U};

    [[nodiscard]] detail::ShapeValues values() const {
        return {.nodes = std::vector<NodeNum>(nodes.size()),
                .numbers = std::vector<double>(numbers.size())};
    }
    // Writes the values of formula `formula` into `values`.
    void gather(std::size_t formula, detail::ShapeValues& values) const {
        std::ranges::transform(
            nodes, values.nodes.begin(),
            [formula](std::span<const NodeNum> slot) { return slot[formula]; });
        std::ranges::transform(
            numbers, values.numbers.begin(),
            [formula](std::span<const double> slot) { return slot[formula]; });
    }
};

std::size_t Formulas::TargetKey::Hash::operator()(
    const TargetKey& key) const noexcept {
    std::uint64_t value =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.node_1))
         << 32U) |
        static_cast<std::uint32_t>(key.node_2);
    value ^= (static_cast<std::uint64_t>(std::to_underlying(key.type)) + 1U) *
             0x9E3779B97F4A7C15ULL;
    // A 64-bit finalizer, so nearby node numbers spread over the buckets.
    value ^= value >> 33U;
    value *= 0xFF51AFD7ED558CCDULL;
    value ^= value >> 33U;
    return static_cast<std::size_t>(value);
}

std::size_t Formulas::BatchKey::Hash::operator()(
    const BatchKey& key) const noexcept {
    return std::hash<const void*>{}(key.shape) ^
           (static_cast<std::size_t>(std::to_underlying(key.origin)) << 1U);
}

std::size_t Formulas::Batch::node_count() const noexcept {
    return shape == nullptr ? 0U : shape->text.node_slots;
}

std::size_t Formulas::Batch::number_count() const noexcept {
    return shape == nullptr ? 0U : shape->text.numbers.size();
}

std::span<const NodeNum> Formulas::Batch::nodes_of(std::size_t instance) const {
    const auto count = node_count();
    return std::span(nodes).subspan(instance * count, count);
}

std::span<const double> Formulas::Batch::numbers_of(
    std::size_t instance) const {
    const auto count = number_count();
    return std::span(numbers).subspan(instance * count, count);
}

std::shared_ptr<Formula> Formulas::Range::Iterator::operator*() const {
    return _formulas->make_formula(_formulas->_order[_position]);
}

void Formulas::Range::Iterator::skip_removed() {
    while ((_position < _formulas->_order.size()) &&
           !_formulas->is_alive(_formulas->_order[_position])) {
        ++_position;
    }
}

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
    _shapes.clear();
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

Formulas::TargetKey Formulas::key_of(const Entity& entity) noexcept {
    return {.node_1 = entity.node_1(),
            .node_2 = entity.node_2(),
            .type = entity.type()};
}

Entity Formulas::target_of(const Batch& batch, std::size_t instance) const {
    if (batch.shape != nullptr) {
        return batch.shape->text.target.resolve(*_network,
                                                batch.nodes_of(instance));
    }
    return batch.callable->entity();
}

Entity Formulas::target_of(detail::FormulaRef formula) const {
    return target_of(_batches[formula.batch], formula.instance);
}

bool Formulas::is_alive(detail::FormulaRef formula) const noexcept {
    return _batches[formula.batch].alive[formula.instance] != 0U;
}

std::shared_ptr<Formula> Formulas::make_formula(
    detail::FormulaRef formula) const {
    const auto& batch = _batches[formula.batch];
    if (batch.callable != nullptr) {
        return batch.callable;
    }
    const auto nodes = batch.nodes_of(formula.instance);
    const auto numbers = batch.numbers_of(formula.instance);
    return std::make_shared<ExpressionFormula>(
        ExpressionFormula::Key{}, target_of(batch, formula.instance),
        batch.shape,
        detail::ShapeValues{.nodes = {nodes.begin(), nodes.end()},
                            .numbers = {numbers.begin(), numbers.end()}},
        _parameters.get(), _network.get(), batch.origin);
}

std::shared_ptr<const detail::FormulaShape> Formulas::shape_for(
    detail::ShapeText text, std::string_view display) {
    auto key = text.key();
    const auto version =
        _parameters == nullptr ? 0U : _parameters->get_structure_version();
    if (const auto found = _shapes.find(key); found != _shapes.end()) {
        auto shape = found->second.lock();
        if ((shape != nullptr) && (shape->parameters == _parameters.get()) &&
            (shape->parameters_version == version)) {
            return shape;
        }
    }
    auto shape = detail::build_shape(std::move(text), _parameters.get(),
                                     _network.get(), display);
    _shapes.insert_or_assign(std::move(key), shape);
    return shape;
}

std::shared_ptr<const detail::FormulaShape> Formulas::adopt_shape(
    const std::shared_ptr<const detail::FormulaShape>& shape) {
    const auto version =
        _parameters == nullptr ? 0U : _parameters->get_structure_version();
    if ((shape->parameters != _parameters.get()) ||
        (shape->parameters_version != version)) {
        return shape_for(shape->text, shape->key);
    }
    auto& cached = _shapes[shape->key];
    if (auto existing = cached.lock();
        existing != nullptr && existing->parameters_version == version) {
        return existing;
    }
    cached = shape;
    return shape;
}

std::optional<std::string> Formulas::missing_input(
    const detail::FormulaShape& shape, std::span<const NodeNum> nodes) const {
    for (const auto& pattern : shape.entities) {
        const auto input = pattern.resolve(*_network, nodes);
        if (!input.exists()) {
            return "ExpressionFormula references unknown symbol '" +
                   input.string_representation() + "'";
        }
    }
    return std::nullopt;
}

void Formulas::settle_load_storage(const detail::FormulaShape& shape,
                                   std::span<const NodeNum> nodes) {
    // A load is empty until a value pointer into it is taken; taking one
    // here keeps that allocation out of a later solve.
    const auto settle = [this, nodes](const detail::EntityPattern& pattern) {
        if (!is_load(pattern.type)) {
            return;
        }
        const auto entity = pattern.resolve(*_network, nodes);
        if (entity.exists()) {
            static_cast<void>(entity.get_value_ref());
        }
    };
    settle(shape.text.target);
    std::ranges::for_each(shape.entities, settle);
}

std::uint32_t Formulas::batch_for(
    const std::shared_ptr<const detail::FormulaShape>& shape, Origin origin) {
    const BatchKey key{.shape = shape.get(), .origin = origin};
    if (const auto found = _open.find(key); found != _open.end()) {
        return found->second;
    }
    _batches.push_back(Batch{.shape = shape,
                             .callable = nullptr,
                             .nodes = {},
                             .numbers = {},
                             .alive = {},
                             .live = 0U,
                             .origin = origin});
    const auto batch = to_u32(_batches.size() - 1U);
    _open.emplace(key, batch);
    return batch;
}

void Formulas::append(std::uint32_t batch, const Entity& target,
                      std::span<const NodeNum> nodes,
                      std::span<const double> numbers) {
    auto& stored = _batches[batch];
    const detail::FormulaRef formula{.batch = batch,
                                     .instance = to_u32(stored.alive.size())};
    _positions.emplace(key_of(target), formula);
    stored.nodes.insert(stored.nodes.end(), nodes.begin(), nodes.end());
    stored.numbers.insert(stored.numbers.end(), numbers.begin(), numbers.end());
    stored.alive.push_back(1U);
    ++stored.live;
    _order.push_back(formula);
    ++_formulas_version;
}

void Formulas::append_callable(const std::shared_ptr<CallableFormula>& callable,
                               Origin origin) {
    const auto& target = callable->entity();
    if (_positions.contains(key_of(target))) {
        throw std::invalid_argument(already_exists(target));
    }
    if (is_load(target.type()) && target.exists()) {
        static_cast<void>(target.get_value_ref());
    }
    callable->set_origin(origin);
    _batches.push_back(Batch{.shape = nullptr,
                             .callable = callable,
                             .nodes = {},
                             .numbers = {},
                             .alive = {1U},
                             .live = 1U,
                             .origin = origin});
    const detail::FormulaRef formula{.batch = to_u32(_batches.size() - 1U),
                                     .instance = 0U};
    _positions.emplace(key_of(target), formula);
    _order.push_back(formula);
    ++_formulas_version;
}

void Formulas::erase(detail::FormulaRef formula) noexcept {
    auto& batch = _batches[formula.batch];
    batch.alive[formula.instance] = 0U;
    --batch.live;
    ++_removed;
    ++_formulas_version;
}

std::pair<std::shared_ptr<const detail::FormulaShape>, detail::ShapeValues>
Formulas::read_formula(const Entity& entity, const std::string& expression) {
    const auto associated_network = ensure_network(_network);
    const auto parameter_storage = ensure_parameters(_parameters);
    auto formula = detail::templatize(
        entity, expression, parameter_storage.get(), associated_network.get());
    auto shape = shape_for(std::move(formula.text), expression);
    if (auto missing = missing_input(*shape, formula.values.nodes);
        missing.has_value()) {
        throw std::invalid_argument(*missing);
    }
    return {std::move(shape), std::move(formula.values)};
}

void Formulas::insert_user(
    const std::shared_ptr<const detail::FormulaShape>& shape,
    const Entity& target, const detail::ShapeValues& values) {
    if (_positions.contains(key_of(target))) {
        throw std::invalid_argument(already_exists(target));
    }
    if (shape->reads_target) {
        warn_self_reference(target);
    }
    settle_load_storage(*shape, values.nodes);
    append(batch_for(shape, Origin::user), target, values.nodes,
           values.numbers);
}

std::shared_ptr<Formula> Formulas::create_formula(
    const Entity& entity, const std::string& formula_string) {
    auto [shape, values] = read_formula(entity, formula_string);
    return std::make_shared<ExpressionFormula>(
        ExpressionFormula::Key{}, entity, std::move(shape), std::move(values),
        _parameters.get(), _network.get(), Origin::user);
}

void Formulas::add_formula(const Entity& entity, double value) {
    static_cast<void>(ensure_network(_network));
    auto formula = detail::constant_formula(entity, value);
    const auto shape = shape_for(std::move(formula.text), "constant");
    insert_user(shape, entity, formula.values);
}

void Formulas::add_formula(std::string_view entity, double value) {
    add_formula(resolve_entity(entity), value);
}

void Formulas::add_formula(const Entity& entity,
                           const std::string& formula_string) {
    if (_positions.contains(key_of(entity))) {
        throw std::invalid_argument(already_exists(entity));
    }
    const auto [shape, values] = read_formula(entity, formula_string);
    insert_user(shape, entity, values);
}

void Formulas::add_formula(std::string_view entity,
                           const std::string& formula_string) {
    add_formula(resolve_entity(entity), formula_string);
}

void Formulas::add_callable_formula(const Entity& entity,
                                    CallableFormula::Function function) {
    append_callable(
        std::make_shared<CallableFormula>(entity, std::move(function)),
        Origin::user);
}

void Formulas::add_callable_formula(std::string_view entity,
                                    CallableFormula::Function function) {
    add_callable_formula(resolve_entity(entity), std::move(function));
}

void Formulas::add_formula(const Formula& formula) {
    if (const auto* callable = dynamic_cast<const CallableFormula*>(&formula);
        callable != nullptr) {
        append_callable(std::make_shared<CallableFormula>(callable->entity(),
                                                          callable->function()),
                        Origin::user);
        return;
    }
    const auto& expression = dynamic_cast<const ExpressionFormula&>(formula);
    static_cast<void>(ensure_network(_network));
    const auto shape = adopt_shape(expression._shape);
    if (auto missing = missing_input(*shape, expression._values.nodes);
        missing.has_value()) {
        throw std::invalid_argument(*missing);
    }
    insert_user(shape, expression.entity(), expression._values);
}

void Formulas::add_formula(const std::shared_ptr<Formula>& formula) {
    if (formula == nullptr) {
        throw std::invalid_argument("Cannot add a null formula");
    }
    // A callable added by pointer is stored as it is, so the caller keeps
    // the same object.
    if (auto callable = std::dynamic_pointer_cast<CallableFormula>(formula);
        callable != nullptr) {
        append_callable(callable, Origin::user);
        return;
    }
    add_formula(*formula);
}

Formulas::PreparedTemplate Formulas::prepare_template(
    const FormulaTemplate& shape, std::span<const NodeSlot> nodes,
    std::span<const NumberSlot> numbers) {
    static_cast<void>(ensure_network(_network));
    static_cast<void>(ensure_parameters(_parameters));
    auto named = detail::parse_template(shape.target, shape.expression);

    PreparedTemplate prepared;
    std::optional<std::size_t> count;
    const auto take = [&count](const std::string& name, auto values) {
        if (count.has_value() && (*count != values.size())) {
            throw std::invalid_argument(
                "Slot '" + name + "' has " + std::to_string(values.size()) +
                " values, the other slots " + std::to_string(*count));
        }
        count = values.size();
        return values;
    };
    const auto values_of =
        [](const auto& slots,
           const std::string& name) -> decltype(slots.front().values) {
        const auto found = std::ranges::find(
            slots, name, [](const auto& slot) { return slot.name; });
        if (found == slots.end()) {
            throw std::invalid_argument("Template slot '" + name +
                                        "' has no values");
        }
        return found->values;
    };
    for (const auto& name : named.node_slots) {
        prepared.nodes.push_back(take(name, values_of(nodes, name)));
    }
    for (const auto& name : named.number_slots) {
        prepared.numbers.push_back(take(name, values_of(numbers, name)));
    }
    const auto unused = [](const auto& slots, const auto& names) {
        const auto extra =
            std::ranges::find_if(slots, [&names](const auto& slot) {
                return std::ranges::find(names, slot.name) == names.end();
            });
        if (extra != slots.end()) {
            throw std::invalid_argument("Slot '" + extra->name +
                                        "' is not a slot of that kind in the "
                                        "template");
        }
    };
    unused(nodes, named.node_slots);
    unused(numbers, named.number_slots);
    prepared.count = count.value_or(1U);
    prepared.shape = shape_for(std::move(named.text), shape.expression);
    return prepared;
}

void Formulas::add_formulas(const FormulaTemplate& shape,
                            std::span<const NodeSlot> nodes,
                            std::span<const NumberSlot> numbers) {
    const auto prepared = prepare_template(shape, nodes, numbers);
    const auto& pattern = prepared.shape->text.target;
    auto values = prepared.values();

    // Everything is checked before anything is added.
    std::vector<TargetKey> keys;
    keys.reserve(prepared.count);
    for (std::size_t formula = 0U; formula < prepared.count; ++formula) {
        prepared.gather(formula, values);
        const auto target = pattern.resolve(*_network, values.nodes);
        if (_positions.contains(key_of(target))) {
            throw std::invalid_argument(already_exists(target));
        }
        if (auto missing = missing_input(*prepared.shape, values.nodes);
            missing.has_value()) {
            throw std::invalid_argument(*missing);
        }
        keys.push_back(key_of(target));
    }
    std::ranges::sort(keys, {}, [](const TargetKey& key) {
        return std::tuple(key.type, key.node_1, key.node_2);
    });
    if (const auto twice = std::ranges::adjacent_find(keys);
        twice != keys.end()) {
        throw std::invalid_argument(
            "The template gives two formulas for '" +
            Entity::make(*_network, twice->type, twice->node_1, twice->node_2)
                .string_representation() +
            "'");
    }

    if (prepared.shape->reads_target) {
        warn_self_reference(pattern.resolve(*_network, values.nodes));
    }
    const auto batch = batch_for(prepared.shape, Origin::user);
    for (std::size_t formula = 0U; formula < prepared.count; ++formula) {
        prepared.gather(formula, values);
        settle_load_storage(*prepared.shape, values.nodes);
        append(batch, pattern.resolve(*_network, values.nodes), values.nodes,
               values.numbers);
    }
}

void Formulas::insert_generated(std::uint32_t batch, const Entity& target,
                                std::span<const NodeNum> nodes,
                                std::span<const double> numbers,
                                GenerationReport& report,
                                Collisions& collisions) {
    const auto existing = _positions.find(key_of(target));
    if ((existing != _positions.end()) &&
        (_batches[existing->second.batch].origin == Origin::user)) {
        ++report.kept_user;
        collisions.note(target, "the user formula stays");
        return;
    }
    const auto& shape = *_batches[batch].shape;
    if (auto missing = missing_input(shape, nodes); missing.has_value()) {
        collisions.note(target, *missing);
        report.failed.push_back(
            FailedFormula{.target = target, .message = std::move(*missing)});
        return;
    }
    if (existing == _positions.end()) {
        ++report.added;
    } else {
        ++report.replaced;
        collisions.note(target,
                        "replaces a " +
                            std::string(to_string(
                                _batches[existing->second.batch].origin)) +
                            " formula");
        erase(existing->second);
        _positions.erase(existing);
    }
    settle_load_storage(shape, nodes);
    append(batch, target, nodes, numbers);
}

GenerationReport Formulas::add_generated_formulas(
    Origin origin, std::span<const GeneratedFormula> formulas) {
    if (origin == Origin::user) {
        throw std::invalid_argument(
            "Generated formulas need a generator origin; add user formulas "
            "with add_formula");
    }

    GenerationReport report;
    Collisions collisions;
    for (const auto& generated : formulas) {
        const auto& target = generated.target;
        if (const auto existing = _positions.find(key_of(target));
            (existing != _positions.end()) &&
            (_batches[existing->second.batch].origin == Origin::user)) {
            ++report.kept_user;
            collisions.note(target, "the user formula stays");
            continue;
        }

        std::shared_ptr<const detail::FormulaShape> shape;
        detail::ShapeValues values;
        try {
            std::tie(shape, values) =
                read_formula(target, generated.expression);
        } catch (const std::exception& exception) {
            report.failed.push_back(
                FailedFormula{.target = target, .message = exception.what()});
            collisions.note(target, exception.what());
            continue;
        }
        if (shape->reads_target) {
            warn_self_reference(target);
        }
        insert_generated(batch_for(shape, origin), target, values.nodes,
                         values.numbers, report, collisions);
    }
    collisions.log(origin, report);
    return report;
}

GenerationReport Formulas::add_generated_formulas(
    Origin origin, const FormulaTemplate& shape,
    std::span<const NodeSlot> nodes, std::span<const NumberSlot> numbers) {
    if (origin == Origin::user) {
        throw std::invalid_argument(
            "Generated formulas need a generator origin; add user formulas "
            "with add_formulas");
    }
    const auto prepared = prepare_template(shape, nodes, numbers);
    const auto& pattern = prepared.shape->text.target;
    auto values = prepared.values();
    const auto batch = batch_for(prepared.shape, origin);
    _batches[batch].nodes.reserve(_batches[batch].nodes.size() +
                                  (prepared.count * prepared.nodes.size()));
    _batches[batch].numbers.reserve(_batches[batch].numbers.size() +
                                    (prepared.count * prepared.numbers.size()));

    GenerationReport report;
    Collisions collisions;
    for (std::size_t formula = 0U; formula < prepared.count; ++formula) {
        prepared.gather(formula, values);
        insert_generated(batch, pattern.resolve(*_network, values.nodes),
                         values.nodes, values.numbers, report, collisions);
    }
    if (prepared.shape->reads_target && (prepared.count > 0U)) {
        warn_self_reference(pattern.resolve(*_network, values.nodes));
    }
    collisions.log(origin, report);
    return report;
}

std::size_t Formulas::remove_formulas(Origin origin) {
    std::size_t removed = 0U;
    for (auto& batch : _batches) {
        if ((batch.origin != origin) || (batch.live == 0U)) {
            continue;
        }
        for (std::size_t instance = 0U; instance < batch.alive.size();
             ++instance) {
            if (batch.alive[instance] == 0U) {
                continue;
            }
            _positions.erase(key_of(target_of(batch, instance)));
            batch.alive[instance] = 0U;
            ++removed;
        }
        batch.live = 0U;
    }

    if (removed > 0U) {
        _removed += removed;
        ++_formulas_version;
        compact();
    }
    SPDLOG_LOGGER_DEBUG(get_logger(), "Removed {} formulas of origin {}",
                        removed, to_string(origin));
    return removed;
}

std::optional<std::size_t> Formulas::order_position(
    detail::FormulaRef formula) const noexcept {
    const auto found = std::ranges::find(_order, formula);
    if (found == _order.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(
        std::ranges::distance(_order.begin(), found));
}

bool Formulas::move_formula(const Entity& target, const Entity& before) {
    const auto moved = _positions.find(key_of(target));
    const auto anchor = _positions.find(key_of(before));
    if ((moved == _positions.end()) || (anchor == _positions.end())) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula '{}' not moved before '{}': both need a "
                           "formula",
                           target.string_representation(),
                           before.string_representation());
        return false;
    }
    if (moved->second == anchor->second) {
        return true;
    }
    const auto from = order_position(moved->second);
    const auto to = order_position(anchor->second);
    if (!from.has_value() || !to.has_value()) {
        return false;
    }
    const auto formula = moved->second;
    _order.erase(_order.begin() + static_cast<std::ptrdiff_t>(*from));
    // The anchor moved down one place if the formula was before it.
    const auto at = *to > *from ? *to - 1U : *to;
    _order.insert(_order.begin() + static_cast<std::ptrdiff_t>(at), formula);
    ++_formulas_version;
    return true;
}

bool Formulas::move_formula(std::string_view target, std::string_view before) {
    const auto resolved_target = try_resolve_entity(target);
    const auto resolved_before = try_resolve_entity(before);
    if (!resolved_target.has_value() || !resolved_before.has_value()) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula '{}' not moved before '{}': unknown "
                           "target",
                           target, before);
        return false;
    }
    return move_formula(*resolved_target, *resolved_before);
}

bool Formulas::move_formula_to_end(const Entity& target) {
    const auto moved = _positions.find(key_of(target));
    if (moved == _positions.end()) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula '{}' not moved: it has no formula",
                           target.string_representation());
        return false;
    }
    const auto from = order_position(moved->second);
    if (!from.has_value()) {
        return false;
    }
    const auto formula = moved->second;
    _order.erase(_order.begin() + static_cast<std::ptrdiff_t>(*from));
    _order.push_back(formula);
    ++_formulas_version;
    return true;
}

bool Formulas::move_formula_to_end(std::string_view target) {
    const auto resolved = try_resolve_entity(target);
    if (!resolved.has_value()) {
        SPDLOG_LOGGER_INFO(get_logger(),
                           "Formula '{}' not moved: unknown target", target);
        return false;
    }
    return move_formula_to_end(*resolved);
}

bool Formulas::contains(const Entity& entity) const noexcept {
    return _positions.contains(key_of(entity));
}

bool Formulas::contains(std::string_view entity) const {
    const auto resolved = try_resolve_entity(entity);
    return resolved.has_value() && contains(*resolved);
}

std::shared_ptr<Formula> Formulas::find(const Entity& entity) const {
    const auto position = _positions.find(key_of(entity));
    if (position == _positions.end()) {
        return nullptr;
    }
    return make_formula(position->second);
}

std::shared_ptr<Formula> Formulas::find(std::string_view entity) const {
    const auto resolved = try_resolve_entity(entity);
    return resolved.has_value() ? find(*resolved) : nullptr;
}

bool Formulas::remove_formula(const Entity& entity) noexcept {
    const auto position = _positions.find(key_of(entity));
    if (position == _positions.end()) {
        // noexcept function: compose without std::format and use the
        // non-formatting log path (see pycanha::log_noexcept).
        log_noexcept(spdlog::level::info, "Formula '" +
                                              entity.string_representation() +
                                              "' was not present for removal");
        return false;
    }

    erase(position->second);
    _positions.erase(position);
    return true;
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

void Formulas::pack(std::uint32_t batch, std::vector<std::uint32_t>& moved) {
    auto& stored = _batches[batch];
    const auto node_count = stored.node_count();
    const auto number_count = stored.number_count();
    moved.assign(stored.alive.size(), dropped);
    std::size_t kept = 0U;
    for (std::size_t instance = 0U; instance < stored.alive.size();
         ++instance) {
        if (stored.alive[instance] == 0U) {
            continue;
        }
        moved[instance] = to_u32(kept);
        if (kept != instance) {
            const auto nodes = stored.nodes_of(instance);
            const auto numbers = stored.numbers_of(instance);
            std::ranges::copy(
                nodes, stored.nodes.begin() +
                           static_cast<std::ptrdiff_t>(kept * node_count));
            std::ranges::copy(
                numbers, stored.numbers.begin() +
                             static_cast<std::ptrdiff_t>(kept * number_count));
        }
        ++kept;
    }
    stored.nodes.resize(kept * node_count);
    stored.numbers.resize(kept * number_count);
    stored.alive.assign(kept, 1U);
    stored.live = kept;
    if (kept > 0U) {
        return;
    }
    // Empty: the batch takes no more formulas and frees its storage.
    if (const auto open = _open.find(
            BatchKey{.shape = stored.shape.get(), .origin = stored.origin});
        (open != _open.end()) && (open->second == batch)) {
        _open.erase(open);
    }
    stored = Batch{};
}

void Formulas::compact() {
    if (_removed == 0U) {
        return;
    }

    // A batch more than half removed is packed; the others keep their
    // holes, which cost a byte each, until they are.
    std::vector<std::vector<std::uint32_t>> moved(_batches.size());
    for (std::size_t batch = 0U; batch < _batches.size(); ++batch) {
        const auto& stored = _batches[batch];
        const auto removed = stored.alive.size() - stored.live;
        if (removed > stored.live) {
            pack(to_u32(batch), moved[batch]);
        }
    }

    std::erase_if(_order, [this, &moved](detail::FormulaRef& formula) {
        const auto& remap = moved[formula.batch];
        if (remap.empty()) {
            return !is_alive(formula);
        }
        formula.instance = remap[formula.instance];
        return formula.instance == dropped;
    });
    for (std::size_t batch = 0U; batch < _batches.size(); ++batch) {
        if (moved[batch].empty()) {
            continue;
        }
        for (std::size_t instance = 0U; instance < _batches[batch].live;
             ++instance) {
            _positions.at(key_of(target_of(_batches[batch], instance))) = {
                .batch = to_u32(batch), .instance = to_u32(instance)};
        }
    }
    std::erase_if(_shapes,
                  [](const auto& entry) { return entry.second.expired(); });
    _removed = 0U;
}

FormulaDependencies Formulas::dependencies() const {
    FormulaDependencies dependencies;
    for (const auto formula : _order) {
        const auto& batch = _batches[formula.batch];
        if (!is_alive(formula) || (batch.shape == nullptr)) {
            continue;
        }
        const auto& shape = *batch.shape;
        const auto nodes = batch.nodes_of(formula.instance);
        const auto target = shape.text.target.resolve(*_network, nodes);
        for (const auto& name : shape.parameter_names) {
            append_once(dependencies.parameters[name], target);
        }
        for (const auto& pattern : shape.entities) {
            append_once(
                dependencies.entities[pattern.resolve(*_network, nodes)],
                target);
        }
        for (const auto& name : shape.lowered.tables) {
            append_once(dependencies.tables[name], target);
        }
    }
    return dependencies;
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

std::optional<std::string> Formulas::problem(detail::FormulaRef formula) const {
    const auto& batch = _batches[formula.batch];
    const auto target = target_of(batch, formula.instance);
    if (!target.exists()) {
        return "its target does not exist";
    }
    if (!target.writable()) {
        return "its target cannot be written";
    }
    if (batch.shape == nullptr) {
        return std::nullopt;
    }
    const auto& shape = *batch.shape;
    for (std::size_t input = 0U; input < shape.parameter_indices.size();
         ++input) {
        if (detail::ParameterAccess::value(
                *_parameters, shape.parameter_indices[input]) == nullptr) {
            return "it reads '" + shape.parameter_names[input] +
                   "', which does not exist";
        }
    }
    const auto nodes = batch.nodes_of(formula.instance);
    for (const auto& pattern : shape.entities) {
        const auto input = pattern.resolve(*_network, nodes);
        if (!input.exists()) {
            return "it reads '" + input.string_representation() +
                   "', which does not exist";
        }
    }
    const auto absent = std::ranges::find_if(
        shape.lowered.tables,
        [this](const auto& name) { return !_parameters->has_table(name); });
    if (absent != shape.lowered.tables.end()) {
        return "it calls table '" + *absent + "', which does not exist";
    }
    return std::nullopt;
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
    for (const auto formula : _order) {
        const auto reason = problem(formula);
        if (!reason.has_value()) {
            continue;
        }
        ++broken;
        if (broken <= listed_formulas) {
            listing += "\n  " + target_of(formula).string_representation() +
                       ": " + *reason;
        }
    }
    if (broken > 0U) {
        throw std::runtime_error(std::to_string(broken) + " of " +
                                 std::to_string(_order.size()) +
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

    build_program();
    _compiled = current_stamp();
}

detail::FormulaPointers Formulas::collect_pointers(
    std::span<const detail::FormulaRef> formulas) const {
    const std::size_t count = formulas.size();
    detail::FormulaPointers pointers;
    pointers.targets.resize(count, nullptr);
    pointers.first_input.resize(count + 1U, 0U);
    pointers.first_callable = count;
    for (std::size_t index = 0U; index < count; ++index) {
        const auto formula = formulas[index];
        const auto& batch = _batches[formula.batch];
        pointers.targets[index] =
            target_of(batch, formula.instance).get_value_ref();
        pointers.first_input[index] = pointers.inputs.size();
        if (batch.shape == nullptr) {
            pointers.first_callable = std::min(pointers.first_callable, index);
            continue;
        }
        const auto& shape = *batch.shape;
        const auto nodes = batch.nodes_of(formula.instance);
        for (const auto& input : shape.inputs) {
            if (input.kind == detail::ShapeInput::Kind::Parameter) {
                pointers.inputs.push_back(detail::ParameterAccess::value(
                    *_parameters, shape.parameter_indices[input.index]));
                continue;
            }
            auto* value = shape.entities[input.index]
                              .resolve(*_network, nodes)
                              .get_value_ref();
            pointers.inputs.push_back(value);
            pointers.entity_reads.emplace_back(value, index);
        }
    }
    pointers.first_input[count] = pointers.inputs.size();
    return pointers;
}

void Formulas::build_program() {
    const auto pointers = collect_pointers(_order);
    auto execution = detail::execution_order(pointers);
    auto& order = execution.order;

    // One id per shape, so the independent formulas can be grouped by it.
    std::unordered_map<const detail::FormulaShape*, std::uint32_t> ids;
    std::vector<std::uint32_t> shape_ids(_batches.size(), 0U);
    std::size_t total_numbers = 0U;
    for (std::size_t batch = 0U; batch < _batches.size(); ++batch) {
        const auto& stored = _batches[batch];
        shape_ids[batch] =
            ids.try_emplace(stored.shape.get(), to_u32(ids.size()))
                .first->second;
        total_numbers += stored.numbers.size();
    }
    const auto shape_of = [&](std::size_t index) {
        return shape_ids[_order[index].batch];
    };

    // Independent formulas may run in any order: grouping them by shape
    // (a stable counting sort) gives one run per shape.
    std::vector<std::size_t> first(ids.size() + 1U, 0U);
    for (std::size_t position = 0U; position < execution.independent;
         ++position) {
        ++first[shape_of(order[position]) + 1U];
    }
    std::partial_sum(first.begin(), first.end(), first.begin());
    std::vector<std::size_t> grouped(execution.independent);
    for (std::size_t position = 0U; position < execution.independent;
         ++position) {
        grouped[first[shape_of(order[position])]++] = order[position];
    }
    std::ranges::copy(grouped, order.begin());

    detail::FormulaProgram program;
    program.reserve(_order.size(), pointers.inputs.size(), total_numbers);
    const double* time = _parameters->time_ptr();
    std::vector<detail::TableView> tables;
    std::size_t begin = 0U;
    while (begin < order.size()) {
        if (begin == execution.independent) {
            program.end_independent();
        }
        const auto formula = _order[order[begin]];
        const auto& batch = _batches[formula.batch];
        if (batch.shape == nullptr) {
            program.add_callable(*batch.callable, formula,
                                 pointers.targets[order[begin]]);
            ++begin;
            continue;
        }
        // The formulas of one shape that run one after another, without
        // mixing independent formulas with the others.
        const std::size_t limit = begin < execution.independent
                                      ? execution.independent
                                      : order.size();
        std::size_t end = begin + 1U;
        while ((end < limit) && (shape_ids[_order[order[end]].batch] ==
                                 shape_ids[formula.batch])) {
            ++end;
        }
        const auto run = std::span(order).subspan(begin, end - begin);
        const auto& shape = *batch.shape;
        tables.clear();
        std::ranges::transform(shape.lowered.tables, std::back_inserter(tables),
                               [this, time](const std::string& name) {
                                   return _parameters->table(name).view(time);
                               });
        program.open_run(shape.lowered, to_u32(shape.inputs.size()),
                         shared_numbers(run), tables);
        for (const std::size_t index : run) {
            const auto member = _order[index];
            program.add_formula(
                member, pointers.targets[index],
                std::span(pointers.inputs)
                    .subspan(pointers.first_input[index],
                             pointers.first_input[index + 1U] -
                                 pointers.first_input[index]),
                _batches[member.batch].numbers_of(member.instance));
        }
        if (shape.kind == FormulaKind::Parameters) {
            // Evaluated again only when what it reads changed.
            std::vector<const std::uint64_t*> changed;
            changed.reserve(shape.parameter_indices.size() +
                            shape.lowered.tables.size());
            std::ranges::transform(shape.parameter_indices,
                                   std::back_inserter(changed),
                                   [this](Index index) {
                                       return detail::ParameterAccess::changed(
                                           *_parameters, index);
                                   });
            std::ranges::transform(shape.lowered.tables,
                                   std::back_inserter(changed),
                                   [this](const std::string& name) {
                                       return detail::ParameterAccess::changed(
                                           _parameters->table(name));
                                   });
            program.watch_run(changed);
        }
        begin = end;
    }
    program.finish();
    _program = std::move(program);
}

std::vector<std::optional<double>> Formulas::shared_numbers(
    std::span<const std::size_t> run) const {
    const auto first = _order[run.front()];
    const auto numbers = _batches[first.batch].numbers_of(first.instance);
    std::vector<std::optional<double>> shared(numbers.begin(), numbers.end());
    for (const std::size_t index : run.subspan(1U)) {
        const auto formula = _order[index];
        const auto own = _batches[formula.batch].numbers_of(formula.instance);
        for (std::size_t slot = 0U; slot < shared.size(); ++slot) {
            auto& value = shared[slot];
            if (value.has_value() && (*value != own[slot])) {
                value.reset();
            }
        }
    }
    return shared;
}

void Formulas::apply_formulas() {
    if (_parameters == nullptr) {
        if (size() > 0U) {
            throw std::runtime_error("Formulas requires parameters");
        }
        return;
    }

    _parameters->update_evaluated_tables();
    if (!active) {
        return;
    }
    if (!is_compiled_current()) {
        compile_formulas();
    }
    if (debug_formulas) {
        run_program_logged();
        return;
    }
    run_program();
}

void Formulas::run_program() {
    const int threads = detail::active_threads();
    _program.reserve_workers(static_cast<std::size_t>(threads));
    const auto program = _program.view();
    // Read at each use: a callable formula may set a parameter.
    const std::uint64_t& changes =
        *detail::ParameterAccess::changes(*_parameters);
    const auto runs = _program.runs();
    const std::size_t independent = _program.independent_runs();
    if (independent > 0U) {
        run_group(program, runs.first(independent), _program.plan(0U), changes,
                  threads);
    }
    for (std::size_t index = independent; index < runs.size(); ++index) {
        const auto& run = runs[index];
        // Often a formula or two: straight to the evaluator.
        if (!run.callables && (run.watch == detail::no_watch) &&
            (!run.blocks || (run.count < block_run_minimum))) {
            static_cast<void>(detail::run_formulas(program, run));
            continue;
        }
        // Only a run of several blocks can be shared among threads.
        if (run.callables || !run.blocks ||
            (run.count <= detail::block_formulas)) {
            run_alone(program, run, changes);
        } else {
            run_group(program, runs.subspan(index, 1U), _program.plan(index),
                      changes, threads);
        }
    }
}

void Formulas::call_run(const detail::ProgramView& program,
                        const detail::ProgramRun& run) {
    // Local views: read through `program`, they would be read again after
    // every call, which the compiler cannot see into.
    const auto targets = program.targets.subspan(run.first, run.count);
    const auto callables =
        program.callables.subspan(run.callable_begin, run.count);
    for (std::size_t formula = 0U; formula < targets.size(); ++formula) {
        *targets[formula] = callables[formula]->call();
    }
}

void Formulas::run_alone(const detail::ProgramView& program,
                         const detail::ProgramRun& run,
                         const std::uint64_t& changes) {
    if (run.callables) {
        call_run(program, run);
        return;
    }
    const bool watched = run.watch != detail::no_watch;
    if (watched && _program.replay(run, changes)) {
        return;
    }
    if (run.blocks && (run.count >= block_run_minimum)) {
        for (std::size_t begin = 0U; begin < run.count;
             begin += detail::block_formulas) {
            detail::run_block(program, run, begin,
                              std::min<std::size_t>(detail::block_formulas,
                                                    run.count - begin),
                              program.columns.first(program.column_count),
                              program.scratch.first(program.scratch_count));
        }
    } else {
        static_cast<void>(detail::run_formulas(program, run));
    }
    if (watched) {
        _program.record(run, changes);
    }
}

void Formulas::run_group(const detail::ProgramView& program,
                         std::span<const detail::ProgramRun> runs,
                         detail::GroupPlan& plan, const std::uint64_t& changes,
                         int threads) {
    _block_tasks.clear();
    _evaluated_watches.clear();
    for (const auto& run : runs) {
        if (run.callables) {
            call_run(program, run);
            continue;
        }
        if (run.watch != detail::no_watch) {
            if (_program.replay(run, changes)) {
                continue;
            }
            _evaluated_watches.push_back(&run);
        }
        if (run.blocks && (run.count >= block_run_minimum)) {
            for (std::size_t begin = 0U; begin < run.count;
                 begin += detail::block_formulas) {
                _block_tasks.push_back(
                    {.run = &run,
                     .begin = begin,
                     .count = std::min<std::size_t>(detail::block_formulas,
                                                    run.count - begin)});
            }
        } else {
            static_cast<void>(detail::run_formulas(program, run));
        }
    }
    run_block_tasks(program, plan, threads);
    for (const auto* run : _evaluated_watches) {
        _program.record(*run, changes);
    }
}

void Formulas::run_block_tasks(const detail::ProgramView& program,
                               detail::GroupPlan& plan, int threads) {
    const auto run_task = [&program, this](std::size_t index,
                                           std::size_t worker) {
        const auto& task = _block_tasks[index];
        detail::run_block(
            program, *task.run, task.begin, task.count,
            program.columns.subspan(worker * program.column_count,
                                    program.column_count),
            program.scratch.subspan(worker * program.scratch_count,
                                    program.scratch_count));
    };
    const auto serially = [&run_task, this] {
        for (std::size_t index = 0U; index < _block_tasks.size(); ++index) {
            run_task(index, 0U);
        }
    };
    const std::size_t tasks = _block_tasks.size();
    if ((threads < 2) || (tasks < 2U)) {
        serially();
        return;
    }
    if (plan.applies == 0U) {
        // Warms the caches for the timed apply.
        serially();
        plan.applies = 1U;
        return;
    }
    if (plan.applies == 1U) {
        const auto start = std::chrono::steady_clock::now();
        serially();
        const std::chrono::duration<double, std::nano> serial =
            std::chrono::steady_clock::now() - start;
        plan.block_ns = serial.count() / static_cast<double>(tasks);
        plan.applies = 2U;
        return;
    }
    const int count =
        fastest_threads(plan.block_ns * static_cast<double>(tasks), threads);
    if (count > 1) {
        detail::ThreadPool::shared().run(tasks, count, run_task);
    } else {
        serially();
    }
}

int Formulas::fastest_threads(double serial_ns, int threads) {
    // The counts weighed are the powers of two up to `threads` and `threads`
    // itself, each costing the pool's overhead plus its share of the work.
    if (_thread_costs_for != threads) {
        _thread_costs.clear();
        auto& pool = detail::ThreadPool::shared();
        for (int count = 2;; count = std::min(2 * count, threads)) {
            const auto overhead = pool.overhead_ns(count);
            if (!overhead.has_value()) {
                // The pool runs another job: one thread now, and the
                // overheads are fetched again at the next apply.
                _thread_costs_for = 0;
                return 1;
            }
            _thread_costs.push_back(
                {.threads = count, .overhead_ns = *overhead});
            if (count >= threads) {
                break;
            }
        }
        _thread_costs_for = threads;
    }
    // More threads must save at least a fifth of the time.
    int best = 1;
    double best_ns = 0.8 * serial_ns;
    for (const auto& cost : _thread_costs) {
        const double time_ns =
            cost.overhead_ns + (serial_ns / static_cast<double>(cost.threads));
        if (time_ns < best_ns) {
            best = cost.threads;
            best_ns = time_ns;
        }
    }
    return best;
}

void Formulas::run_program_logged() {
    const auto program = _program.view();
    const auto log = [this](std::size_t position) {
        SPDLOG_LOGGER_INFO(
            get_logger(), "{} = {}",
            target_of(_program.formula_at(position)).string_representation(),
            _program.value_at(position));
    };
    for (const auto& run : _program.runs()) {
        if (run.callables) {
            for (std::uint32_t formula = 0U; formula < run.count; ++formula) {
                *program.targets[run.first + formula] =
                    program.callables[run.callable_begin + formula]->call();
                log(run.first + formula);
            }
            continue;
        }
        // One formula at a time.
        auto single = run;
        single.count = 1U;
        for (std::uint32_t formula = 0U; formula < run.count; ++formula) {
            single.first = run.first + formula;
            single.input_begin = run.input_begin + (formula * run.input_stride);
            single.slot_begin = run.slot_begin + (formula * run.slot_stride);
            static_cast<void>(detail::run_formulas(program, single));
            log(single.first);
        }
    }
}

void Formulas::calculate_derivatives() {
    for (const auto& batch : _batches) {
        if ((batch.shape != nullptr) && (batch.live > 0U) &&
            (batch.shape->kind == FormulaKind::Parameters)) {
            static_cast<void>(detail::shape_derivatives(*batch.shape));
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
