#include "pycanha-core/parameters/formula_shape.hpp"

#include <symengine/basic.h>
#include <symengine/derivative.h>
#include <symengine/parser.h>
#include <symengine/symbol.h>
#include <symengine/symengine_rcp.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_kind.hpp"
#include "pycanha-core/parameters/formula_program.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha::detail {

namespace {

using ExpressionNode = SymEngine::RCP<const SymEngine::Basic>;
using SymbolNode = SymEngine::RCP<const SymEngine::Symbol>;

// Placeholder symbols start with this, or with more leading underscores
// when a name of the formula already does.
constexpr std::string_view default_prefix = "_f";

[[nodiscard]] std::uint32_t to_u32(std::size_t value) {
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::string replace_python_power(std::string expression) {
    std::size_t position = 0U;
    while ((position = expression.find("**", position)) != std::string::npos) {
        expression.replace(position, 2U, "^");
        position += 1U;
    }
    return expression;
}

[[nodiscard]] std::string preprocess_expression(const std::string& expression) {
    if ((expression.contains('[')) || (expression.contains(']'))) {
        throw std::invalid_argument(
            "Formulas do not support matrix or array access yet");
    }

    return replace_python_power(expression);
}

// Free symbols of `expr`, unique and sorted by name. The names of called
// functions are not symbols.
[[nodiscard]] std::vector<SymbolNode> collect_symbols(
    const ExpressionNode& expr) {
    std::map<std::string, SymbolNode> symbols;
    std::vector<ExpressionNode> pending{expr};

    while (!pending.empty()) {
        const ExpressionNode current = pending.back();
        pending.pop_back();

        if (const auto* symbol =
                dynamic_cast<const SymEngine::Symbol*>(current.get());
            symbol != nullptr) {
            symbols.emplace(
                symbol->get_name(),
                SymEngine::rcp_static_cast<const SymEngine::Symbol>(current));
            continue;
        }

        const auto arguments = current->get_args();
        std::ranges::copy(std::views::reverse(arguments),
                          std::back_inserter(pending));
    }

    std::vector<SymbolNode> sorted;
    sorted.reserve(symbols.size());
    std::ranges::copy(std::views::values(symbols), std::back_inserter(sorted));
    return sorted;
}

[[nodiscard]] bool is_blank(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] bool is_digit(char ch) {
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] bool starts_identifier(char ch) {
    return (std::isalpha(static_cast<unsigned char>(ch)) != 0) || (ch == '_');
}

[[nodiscard]] std::size_t skip_blanks(std::string_view text,
                                      std::size_t position) {
    while ((position < text.size()) && is_blank(text[position])) {
        ++position;
    }
    return position;
}

[[nodiscard]] std::size_t scan_identifier(std::string_view text,
                                          std::size_t position) {
    while ((position < text.size()) && is_symbol_char(text[position])) {
        ++position;
    }
    return position;
}

[[nodiscard]] std::size_t scan_digits(std::string_view text,
                                      std::size_t position) {
    while ((position < text.size()) && is_digit(text[position])) {
        ++position;
    }
    return position;
}

// A number: digits, an optional fraction and an optional exponent.
[[nodiscard]] std::size_t scan_number(std::string_view text,
                                      std::size_t position) {
    position = scan_digits(text, position);
    if ((position < text.size()) && (text[position] == '.')) {
        position = scan_digits(text, position + 1U);
    }
    if ((position < text.size()) &&
        ((text[position] == 'e') || (text[position] == 'E'))) {
        std::size_t exponent = position + 1U;
        if ((exponent < text.size()) &&
            ((text[exponent] == '+') || (text[exponent] == '-'))) {
            ++exponent;
        }
        if ((exponent < text.size()) && is_digit(text[exponent])) {
            position = scan_digits(text, exponent);
        }
    }
    return position;
}

// The last character before `position` that is not blank, or '\0'.
[[nodiscard]] char previous_char(std::string_view text, std::size_t& position) {
    while ((position > 0U) && is_blank(text[position - 1U])) {
        --position;
    }
    return position == 0U ? '\0' : text[position - 1U];
}

// Whether the number at `position` is an exponent: after ^ or **, perhaps
// behind a sign or an opening parenthesis.
[[nodiscard]] bool is_exponent(std::string_view text, std::size_t position) {
    char before = previous_char(text, position);
    if ((before == '-') || (before == '+') || (before == '(')) {
        --position;
        before = previous_char(text, position);
    }
    return (before == '^') || ((before == '*') && (position >= 2U) &&
                               (text[position - 2U] == '*'));
}

[[nodiscard]] std::optional<double> parse_double(std::string_view token) {
    const std::string owned(token);
    double value = 0.0;
    const auto* const end = std::to_address(owned.end());
    const auto [stop, error] = std::from_chars(owned.data(), end, value);
    if ((error != std::errc{}) || (stop != end)) {
        return std::nullopt;
    }
    return value;
}

// A slot index written by this file: digits only.
[[nodiscard]] std::size_t to_index(std::string_view digits) {
    return std::accumulate(digits.begin(), digits.end(), std::size_t{0U},
                           [](std::size_t value, char ch) {
                               return (value * 10U) +
                                      static_cast<std::size_t>(ch - '0');
                           });
}

[[nodiscard]] std::string upper(std::string_view text) {
    std::string result(text);
    std::ranges::transform(result, result.begin(), [](char ch) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    });
    return result;
}

// The type of a one-node entity token (T, C, QS, ...), any case.
[[nodiscard]] std::optional<EntityType> one_node_type(std::string_view token) {
    const auto type = lookup_entity_type(upper(token));
    if (!type.has_value() || (entity_ops(*type).node_count != 1U)) {
        return std::nullopt;
    }
    return type;
}

[[nodiscard]] std::string ref_text(const NodeRef& ref) {
    return ref.slot ? "{" + std::to_string(ref.value) + "}"
                    : std::to_string(ref.value);
}

[[nodiscard]] std::string entity_text(const EntityPattern& pattern) {
    const auto& ops = entity_ops(pattern.type);
    std::string text(ops.token);
    if (ops.node_count == 1U) {
        return text + ref_text(pattern.node_1);
    }
    if (ops.node_count == 2U) {
        return text + "(" + ref_text(pattern.node_1) + "," +
               ref_text(pattern.node_2) + ")";
    }
    return text;
}

[[nodiscard]] std::string format_number(double value) {
    auto text = std::format("{}", value);
    return value < 0.0 ? "(" + text + ")" : text;
}

// The node token of a coupling written GL(a, b) / GR(a, b): digits or,
// when `named`, {name}. Returns where the token ends (== position when
// there is none).
[[nodiscard]] std::size_t node_token_end(std::string_view text,
                                         std::size_t position, bool named) {
    if (named && (position < text.size()) && (text[position] == '{')) {
        const auto close = text.find('}', position);
        return close == std::string_view::npos ? position : close + 1U;
    }
    return scan_digits(text, position);
}

struct CouplingCall {
    EntityType type{EntityType::gl};
    std::string_view node_1;
    std::string_view node_2;
    std::size_t end{0U};
};

// One node of a coupling call at `cursor` (just after '(' or ','), followed
// by `separator`. Advances `cursor` past the separator.
[[nodiscard]] std::optional<std::string_view> coupling_node(
    std::string_view text, std::size_t& cursor, bool named, char separator) {
    const auto begin = skip_blanks(text, cursor);
    const auto end = node_token_end(text, begin, named);
    if (end == begin) {
        return std::nullopt;
    }
    const auto after = skip_blanks(text, end);
    if ((after >= text.size()) || (text[after] != separator)) {
        return std::nullopt;
    }
    cursor = after + 1U;
    return text.substr(begin, end - begin);
}

// A coupling GL(a, b) / GR(a, b) whose name is text[position, name_end).
[[nodiscard]] std::optional<CouplingCall> parse_coupling(std::string_view text,
                                                         std::size_t position,
                                                         std::size_t name_end,
                                                         bool named) {
    const auto name = upper(text.substr(position, name_end - position));
    if ((name != "GL") && (name != "GR")) {
        return std::nullopt;
    }
    std::size_t cursor = skip_blanks(text, name_end);
    if ((cursor >= text.size()) || (text[cursor] != '(')) {
        return std::nullopt;
    }
    ++cursor;
    const auto node_1 = coupling_node(text, cursor, named, ',');
    if (!node_1.has_value()) {
        return std::nullopt;
    }
    const auto node_2 = coupling_node(text, cursor, named, ')');
    if (!node_2.has_value()) {
        return std::nullopt;
    }
    return CouplingCall{.type = name == "GL" ? EntityType::gl : EntityType::gr,
                        .node_1 = *node_1,
                        .node_2 = *node_2,
                        .end = cursor};
}

// Writes the two texts of a shape as the formula is read.
class ShapeWriter {
  public:
    explicit ShapeWriter(std::string prefix) : _prefix(std::move(prefix)) {}

  protected:
    void copy(std::string_view piece) {
        _text.print += piece;
        _text.expression += piece;
    }

    void add_entity(const EntityPattern& pattern) {
        auto found =
            std::ranges::find(_text.entities, pattern,
                              &std::pair<std::string, EntityPattern>::second);
        if (found == _text.entities.end()) {
            _text.entities.emplace_back(
                _prefix + "e" + std::to_string(_text.entities.size()), pattern);
            found = std::prev(_text.entities.end());
        }
        _text.expression += found->first;
        _text.print += entity_text(pattern);
    }

    // Number slot `slot`, read at this point.
    void add_number(std::size_t slot) {
        while (_text.numbers.size() <= slot) {
            _text.numbers.push_back(_prefix + "n" +
                                    std::to_string(_text.numbers.size()));
        }
        _text.expression += _text.numbers[slot];
        _text.print += "{#" + std::to_string(slot) + "}";
    }

    // Notes a name of the formula, which placeholders must not start with.
    void note_name(std::string_view name) {
        if (name.starts_with(_prefix)) {
            _collided = true;
        }
    }

    [[nodiscard]] bool collided() const noexcept { return _collided; }
    [[nodiscard]] ShapeText& text() noexcept { return _text; }

  private:
    std::string _prefix;
    ShapeText _text;
    bool _collided{false};
};

// Reads a formula, taking its numbers and node numbers out.
class Templatizer : public ShapeWriter {
  public:
    Templatizer(std::string_view expression, const Parameters* parameters,
                ThermalNetwork* network, std::string prefix)
        : ShapeWriter(std::move(prefix)),
          _expression(expression),
          _parameters(parameters),
          _network(network) {}

    // False when a name of the formula starts with the placeholder prefix.
    [[nodiscard]] bool run(const Entity& target) {
        text().target = target_pattern(target);
        std::size_t position = 0U;
        while (position < _expression.size()) {
            const char ch = _expression[position];
            if (is_digit(ch) ||
                ((ch == '.') && (position + 1U < _expression.size()) &&
                 is_digit(_expression[position + 1U]))) {
                position = number(position);
            } else if (starts_identifier(ch)) {
                position = identifier(position);
            } else {
                copy(_expression.substr(position, 1U));
                ++position;
            }
        }
        text().node_slots = to_u32(_values.nodes.size());
        return !collided();
    }

    [[nodiscard]] TemplatedFormula take() {
        return {.text = std::move(text()), .values = std::move(_values)};
    }

    [[nodiscard]] EntityPattern target_pattern(const Entity& target) {
        EntityPattern pattern{
            .type = target.type(), .node_1 = {}, .node_2 = {}};
        if (target.node_count() >= 1) {
            pattern.node_1 = slot_of(target.node_1());
        }
        if (target.node_count() == 2) {
            pattern.node_2 = slot_of(target.node_2());
        }
        return pattern;
    }

  private:
    // Equal node numbers share a slot, so a formula reading its target's
    // node refers to the same slot as the target.
    [[nodiscard]] NodeRef slot_of(NodeNum node) {
        const auto found = std::ranges::find(_values.nodes, node);
        const auto slot = static_cast<std::size_t>(
            std::ranges::distance(_values.nodes.begin(), found));
        if (found == _values.nodes.end()) {
            _values.nodes.push_back(node);
        }
        return {.value = static_cast<std::int32_t>(slot), .slot = true};
    }

    [[nodiscard]] std::size_t number(std::size_t position) {
        const auto end = scan_number(_expression, position);
        const auto token = _expression.substr(position, end - position);
        const bool glued =
            (end < _expression.size()) &&
            (is_symbol_char(_expression[end]) || (_expression[end] == '(') ||
             (_expression[end] == '.'));
        const auto value = (glued || is_exponent(_expression, position))
                               ? std::nullopt
                               : parse_double(token);
        if (!value.has_value()) {
            copy(token);
            return end;
        }
        add_number(_values.numbers.size());
        _values.numbers.push_back(*value);
        return end;
    }

    [[nodiscard]] std::size_t identifier(std::size_t position) {
        const auto end = scan_identifier(_expression, position);
        const auto name = _expression.substr(position, end - position);
        note_name(name);
        if (_network != nullptr) {
            if (const auto call = parse_coupling(_expression, position, end,
                                                 /*named=*/false);
                call.has_value() && add_coupling(*call)) {
                return call->end;
            }
        }
        const auto next = skip_blanks(_expression, end);
        const bool called =
            (next < _expression.size()) && (_expression[next] == '(');
        if (called || !lift_entity(name)) {
            copy(name);
        }
        return end;
    }

    // False when a node number does not fit, which leaves the call as
    // written.
    [[nodiscard]] bool add_coupling(const CouplingCall& call) {
        const auto node_1 = parse_node_num(call.node_1);
        const auto node_2 = parse_node_num(call.node_2);
        if (!node_1.has_value() || !node_2.has_value()) {
            return false;
        }
        add_entity({.type = call.type,
                    .node_1 = slot_of(*node_1),
                    .node_2 = slot_of(*node_2)});
        return true;
    }

    // An entity name (not a parameter) becomes an entity with node slots.
    [[nodiscard]] bool lift_entity(std::string_view name) {
        if (_network == nullptr) {
            return false;
        }
        const std::string owned(name);
        if ((_parameters != nullptr) && _parameters->contains(owned)) {
            return false;
        }
        const auto entity = Entity::from_internal_symbol(*_network, owned);
        if (!entity.has_value() || (entity->node_count() == 0)) {
            return false;
        }
        add_entity(target_pattern(*entity));
        return true;
    }

    std::string_view _expression;
    const Parameters* _parameters;
    ThermalNetwork* _network;
    ShapeValues _values;
};

// Reads a template with named slots.
class NamedParser : public ShapeWriter {
  public:
    explicit NamedParser(std::string prefix) : ShapeWriter(std::move(prefix)) {}

    // False when a name of the template starts with the placeholder prefix.
    [[nodiscard]] bool run(std::string_view target,
                           std::string_view expression) {
        text().target = parse_target(target);
        std::size_t position = 0U;
        while (position < expression.size()) {
            const char ch = expression[position];
            if (ch == '{') {
                position = number_slot(expression, position);
            } else if (starts_identifier(ch)) {
                position = identifier(expression, position);
            } else {
                copy(expression.substr(position, 1U));
                ++position;
            }
        }
        text().node_slots = to_u32(_node_slots.size());
        return !collided();
    }

    [[nodiscard]] NamedTemplate take() {
        return {.text = std::move(text()),
                .node_slots = std::move(_node_slots),
                .number_slots = std::move(_number_slots)};
    }

  private:
    // The name inside {name} at `position`; throws when it is malformed.
    [[nodiscard]] static std::string_view slot_name(std::string_view text,
                                                    std::size_t position,
                                                    std::size_t& end) {
        const auto close = text.find('}', position);
        if (close == std::string_view::npos) {
            throw std::invalid_argument("Unclosed slot in formula template '" +
                                        std::string(text) + "'");
        }
        const auto name = text.substr(position + 1U, close - position - 1U);
        if (name.empty() || !starts_identifier(name.front()) ||
            (scan_identifier(name, 0U) != name.size())) {
            throw std::invalid_argument("Slot '{" + std::string(name) +
                                        "}' must be a name, in '" +
                                        std::string(text) + "'");
        }
        end = close + 1U;
        return name;
    }

    [[nodiscard]] NodeRef node_slot(std::string_view name) {
        if (std::ranges::find(_number_slots, name) != _number_slots.end()) {
            throw std::invalid_argument(
                "Slot '" + std::string(name) +
                "' is used both as a node number and as a number");
        }
        const auto found = std::ranges::find(_node_slots, name);
        const auto slot = static_cast<std::size_t>(
            std::ranges::distance(_node_slots.begin(), found));
        if (found == _node_slots.end()) {
            _node_slots.emplace_back(name);
        }
        return {.value = static_cast<std::int32_t>(slot), .slot = true};
    }

    // A node written as {name} or as a node number.
    [[nodiscard]] NodeRef node_ref(std::string_view token) {
        if (token.starts_with('{')) {
            std::size_t end = 0U;
            return node_slot(slot_name(token, 0U, end));
        }
        const auto node = parse_node_num(token);
        if (!node.has_value()) {
            throw std::invalid_argument("'" + std::string(token) +
                                        "' is not a node number");
        }
        return {.value = *node, .slot = false};
    }

    [[nodiscard]] EntityPattern parse_target(std::string_view target) {
        const auto start = skip_blanks(target, 0U);
        const auto name_end = scan_identifier(target, start);
        const auto name = target.substr(start, name_end - start);
        std::optional<EntityPattern> pattern;
        std::size_t end = name_end;
        if (const auto call =
                parse_coupling(target, start, name_end, /*named=*/true);
            call.has_value()) {
            pattern = EntityPattern{.type = call->type,
                                    .node_1 = node_ref(call->node_1),
                                    .node_2 = node_ref(call->node_2)};
            end = call->end;
        } else if ((name_end < target.size()) && (target[name_end] == '{')) {
            pattern = single_node(name, target, name_end, end);
        } else {
            pattern = fixed_single_node(name);
        }
        if (!pattern.has_value() ||
            (skip_blanks(target, end) != target.size())) {
            throw std::invalid_argument(
                "Formula template target '" + std::string(target) +
                "' must be one entity, such as T{a} or GL({a},{b})");
        }
        return *pattern;
    }

    // T{a}, C{a}, QI{a}, ...: the name is text[.., brace) and `end` is set
    // past the closing brace.
    [[nodiscard]] EntityPattern single_node(std::string_view name,
                                            std::string_view text,
                                            std::size_t brace,
                                            std::size_t& end) {
        const auto type = one_node_type(name);
        if (!type.has_value()) {
            throw std::invalid_argument(
                "Only T, C, QS, QA, QE, QI and QR take a node slot, not '" +
                std::string(name) + "'");
        }
        return EntityPattern{.type = *type,
                             .node_1 = node_slot(slot_name(text, brace, end)),
                             .node_2 = {}};
    }

    // A fixed one-node entity such as T7.
    [[nodiscard]] static std::optional<EntityPattern> fixed_single_node(
        std::string_view name) {
        std::size_t letters = 0U;
        while ((letters < name.size()) &&
               (std::isalpha(static_cast<unsigned char>(name[letters])) != 0)) {
            ++letters;
        }
        const auto type = one_node_type(name.substr(0U, letters));
        const auto node = parse_node_num(name.substr(letters));
        if (!type.has_value() || !node.has_value()) {
            return std::nullopt;
        }
        return EntityPattern{.type = *type,
                             .node_1 = {.value = *node, .slot = false},
                             .node_2 = {}};
    }

    [[nodiscard]] std::size_t number_slot(std::string_view expression,
                                          std::size_t position) {
        std::size_t end = 0U;
        const auto name = slot_name(expression, position, end);
        if (std::ranges::find(_node_slots, name) != _node_slots.end()) {
            throw std::invalid_argument(
                "Slot '" + std::string(name) +
                "' is used both as a node number and as a number");
        }
        const auto found = std::ranges::find(_number_slots, name);
        const auto slot = static_cast<std::size_t>(
            std::ranges::distance(_number_slots.begin(), found));
        if (found == _number_slots.end()) {
            _number_slots.emplace_back(name);
        }
        add_number(slot);
        return end;
    }

    [[nodiscard]] std::size_t identifier(std::string_view expression,
                                         std::size_t position) {
        const auto end = scan_identifier(expression, position);
        const auto name = expression.substr(position, end - position);
        note_name(name);
        if (const auto call =
                parse_coupling(expression, position, end, /*named=*/true);
            call.has_value()) {
            add_entity({.type = call->type,
                        .node_1 = node_ref(call->node_1),
                        .node_2 = node_ref(call->node_2)});
            return call->end;
        }
        if ((end < expression.size()) && (expression[end] == '{')) {
            std::size_t after = end;
            add_entity(single_node(name, expression, end, after));
            return after;
        }
        copy(name);
        return end;
    }

    std::vector<std::string> _node_slots;
    std::vector<std::string> _number_slots;
};

// How a symbol of a parsed shape is read: as a parameter (its index), or
// else as an entity.
struct SymbolRole {
    std::optional<Index> parameter;
    EntityPattern pattern;
};

[[nodiscard]] SymbolRole symbol_role(const ShapeText& text,
                                     const std::string& name,
                                     const Parameters* parameters,
                                     ThermalNetwork* network) {
    const auto entity = std::ranges::find(
        text.entities, name, &std::pair<std::string, EntityPattern>::first);
    if (entity != text.entities.end()) {
        return {.parameter = std::nullopt, .pattern = entity->second};
    }
    if ((parameters != nullptr) && parameters->contains(name)) {
        const auto index = parameters->get_idx(name);
        if (!index.has_value() ||
            (parameters->get_double_ptr(*index) == nullptr)) {
            throw std::invalid_argument(
                "ExpressionFormula expects parameter '" + name +
                "' to store a double");
        }
        return {.parameter = index, .pattern = {}};
    }
    if ((parameters != nullptr) && parameters->has_table(name)) {
        throw std::invalid_argument(
            std::format("'{0}' is a table; call it as {0}(argument)", name));
    }
    if (network == nullptr) {
        throw std::invalid_argument(
            "ExpressionFormula references unknown parameter '" + name + "'");
    }
    const auto input = Entity::from_internal_symbol(*network, name);
    if (!input.has_value() || !input->exists()) {
        throw std::invalid_argument(
            "ExpressionFormula references unknown symbol '" + name + "'");
    }
    EntityPattern pattern{.type = input->type(),
                          .node_1 = {.value = input->node_1(), .slot = false},
                          .node_2 = {}};
    if (input->node_count() == 2) {
        pattern.node_2 = {.value = input->node_2(), .slot = false};
    }
    return {.parameter = std::nullopt, .pattern = pattern};
}

// The symbols a shape loads (every symbol but its numbers), in LOAD order.
[[nodiscard]] std::vector<SymbolNode> input_symbols(
    const ShapeText& text, const ParsedExpression& parsed) {
    std::vector<SymbolNode> inputs;
    std::ranges::copy_if(parsed.symbols, std::back_inserter(inputs),
                         [&text](const SymbolNode& symbol) {
                             return std::ranges::find(text.numbers,
                                                      symbol->get_name()) ==
                                    text.numbers.end();
                         });
    return inputs;
}

[[nodiscard]] FormulaKind kind_of(const FormulaShape& shape) {
    if (!shape.entities.empty()) {
        return FormulaKind::Entities;
    }
    if (!shape.parameter_names.empty() || !shape.lowered.tables.empty()) {
        return FormulaKind::Parameters;
    }
    return FormulaKind::Constant;
}

}  // namespace

ParsedExpression parse_expression(const std::string& expression,
                                  std::string_view display) {
    const auto normalized =
        preprocess_expression(preprocess_formula_symbols(expression));

    ParsedExpression parsed;
    try {
        parsed.tree = SymEngine::parse(normalized);
    } catch (const std::exception& exception) {
        throw std::invalid_argument(
            "Formula could not parse '" +
            (display.empty() ? expression : std::string(display)) +
            "': " + exception.what());
    }
    parsed.symbols = collect_symbols(parsed.tree);
    return parsed;
}

Entity EntityPattern::resolve(ThermalNetwork& network,
                              std::span<const NodeNum> nodes) const {
    return Entity::make(network, type, node_1.node(nodes), node_2.node(nodes));
}

std::string ShapeText::key() const {
    std::string result = entity_text(target);
    result += '=';
    std::ranges::copy_if(print, std::back_inserter(result),
                         [](char ch) { return !is_blank(ch); });
    return result;
}

TemplatedFormula templatize(const Entity& target, std::string_view expression,
                            const Parameters* parameters,
                            ThermalNetwork* network) {
    std::string prefix(default_prefix);
    while (true) {
        Templatizer reader(expression, parameters, network, prefix);
        if (reader.run(target)) {
            return reader.take();
        }
        prefix.insert(0U, "_");
    }
}

TemplatedFormula constant_formula(const Entity& target, double value) {
    Templatizer reader({}, nullptr, nullptr, std::string(default_prefix));
    static_cast<void>(reader.run(target));
    auto formula = reader.take();
    formula.text.numbers.push_back(std::string(default_prefix) + "n0");
    formula.text.expression = formula.text.numbers.front();
    formula.text.print = "{#0}";
    formula.values.numbers.push_back(value);
    return formula;
}

NamedTemplate parse_template(std::string_view target,
                             std::string_view expression) {
    std::string prefix(default_prefix);
    while (true) {
        NamedParser reader(prefix);
        if (reader.run(target, expression)) {
            return reader.take();
        }
        prefix.insert(0U, "_");
    }
}

std::shared_ptr<const FormulaShape> build_shape(ShapeText text,
                                                const Parameters* parameters,
                                                ThermalNetwork* network,
                                                std::string_view display) {
    const auto parsed = parse_expression(text.expression, display);
    auto shape = std::make_shared<FormulaShape>();
    const auto inputs = input_symbols(text, parsed);
    for (const auto& symbol : inputs) {
        const auto role =
            symbol_role(text, symbol->get_name(), parameters, network);
        if (const auto& index = role.parameter; index.has_value()) {
            shape->inputs.push_back(
                {.kind = ShapeInput::Kind::Parameter,
                 .index = to_u32(shape->parameter_names.size())});
            shape->parameter_names.push_back(
                parameters->get_parameter_name(*index).value_or(
                    symbol->get_name()));
            shape->parameter_indices.push_back(*index);
            continue;
        }
        shape->inputs.push_back({.kind = ShapeInput::Kind::Entity,
                                 .index = to_u32(shape->entities.size())});
        shape->entities.push_back(role.pattern);
    }

    try {
        shape->lowered =
            lower_expression(*parsed.tree, inputs, text.numbers, parameters);
    } catch (const std::invalid_argument& exception) {
        throw std::invalid_argument("Formula '" + std::string(display) +
                                    "': " + exception.what());
    }
    shape->kind = kind_of(*shape);
    shape->reads_target = std::ranges::find(shape->entities, text.target) !=
                          shape->entities.end();
    shape->parameters = parameters;
    shape->parameters_version =
        parameters == nullptr ? 0U : parameters->get_structure_version();
    shape->key = text.key();
    shape->text = std::move(text);
    return shape;
}

const std::vector<LoweredExpression>& shape_derivatives(
    const FormulaShape& shape) {
    if (shape.derivatives != nullptr) {
        return *shape.derivatives;
    }
    // Parsed again: derivatives are rarely needed, so nothing is kept for
    // them until they are.
    const auto parsed = parse_expression(shape.text.expression);
    const auto inputs = input_symbols(shape.text, parsed);
    auto derivatives = std::make_shared<std::vector<LoweredExpression>>();
    derivatives->reserve(inputs.size());
    for (const auto& symbol : inputs) {
        const auto derivative = SymEngine::diff(parsed.tree, symbol);
        try {
            derivatives->push_back(lower_expression(
                *derivative, inputs, shape.text.numbers, shape.parameters));
        } catch (const std::invalid_argument& exception) {
            throw std::runtime_error("Formula '" + shape.key +
                                     "' has no derivative with respect to '" +
                                     symbol->get_name() +
                                     "': " + exception.what());
        }
    }
    shape.derivatives = std::move(derivatives);
    return *shape.derivatives;
}

std::string print_expression(const FormulaShape& shape,
                             std::span<const NodeNum> nodes,
                             std::span<const double> numbers) {
    const std::string_view print = shape.text.print;
    std::string result;
    result.reserve(print.size() + 16U);
    std::size_t position = 0U;
    while (position < print.size()) {
        const auto open = print.find('{', position);
        result += print.substr(position, open - position);
        if (open == std::string_view::npos) {
            break;
        }
        const auto close = print.find('}', open);
        const auto inside = print.substr(open + 1U, close - open - 1U);
        if (inside.starts_with('#')) {
            result += format_number(numbers[to_index(inside.substr(1U))]);
        } else {
            result += std::to_string(nodes[to_index(inside)]);
        }
        position = close + 1U;
    }
    return result;
}

}  // namespace pycanha::detail
