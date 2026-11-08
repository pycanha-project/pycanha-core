#pragma once

#include <symengine/basic.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula_kind.hpp"
#include "pycanha-core/parameters/formula_program.hpp"

namespace pycanha {

class Parameters;

namespace detail {

// A formula expression after parsing: the expression tree and its free
// symbols, unique and sorted by name.
struct ParsedExpression {
    SymEngine::RCP<const SymEngine::Basic> tree;
    std::vector<SymEngine::RCP<const SymEngine::Symbol>> symbols;
};

// Parses a formula expression. Python's `**` is accepted for powers and
// GL(i,j) / GR(i,j) for couplings. Throws std::invalid_argument when the text
// does not parse or uses array access; the message quotes `display`, or the
// expression itself when `display` is empty.
[[nodiscard]] ParsedExpression parse_expression(const std::string& expression,
                                                std::string_view display = {});

// A node of an entity in a formula shape: one of the formula's own node
// numbers (`slot` true, `value` is the slot), or a node fixed by the shape.
struct NodeRef {
    std::int32_t value{0};
    bool slot{false};

    [[nodiscard]] bool operator==(const NodeRef&) const = default;
    [[nodiscard]] NodeNum node(std::span<const NodeNum> nodes) const {
        return slot ? nodes[static_cast<std::size_t>(value)] : value;
    }
};

// An entity in a formula shape: its type and its nodes (node_2 for the
// couplings only).
struct EntityPattern {
    EntityType type{EntityType::t};
    NodeRef node_1;
    NodeRef node_2;

    [[nodiscard]] bool operator==(const EntityPattern&) const = default;
    // The entity of a formula with these node numbers.
    [[nodiscard]] Entity resolve(ThermalNetwork& network,
                                 std::span<const NodeNum> nodes) const;
};

// A formula's text with the parts that differ between formulas of the same
// shape taken out: its node numbers and numbers.
struct ShapeText {
    EntityPattern target;
    // The expression to parse, with placeholder symbols for the entities
    // and numbers of the formula.
    std::string expression;
    // The expression to print: "{k}" stands for node slot k and "{#j}" for
    // number slot j.
    std::string print;
    // The placeholder symbols of the entities read, and of the numbers by
    // slot.
    std::vector<std::pair<std::string, EntityPattern>> entities;
    std::vector<std::string> numbers;
    std::uint32_t node_slots{0U};

    // Identifies the shape: its target and printed expression without
    // blanks.
    [[nodiscard]] std::string key() const;
};

// The node numbers and numbers of one formula of a shape.
struct ShapeValues {
    std::vector<NodeNum> nodes;
    std::vector<double> numbers;
};

struct TemplatedFormula {
    ShapeText text;
    ShapeValues values;
};

// Takes the numbers and node numbers out of a formula's expression, so that
// every formula written the same way, whatever its numbers, shares one
// parse. Names are read as the parser reads them: a parameter or a function
// stays in the shape, an entity becomes node slots. A number that is an
// exponent stays too, so integer powers keep their own operation.
[[nodiscard]] TemplatedFormula templatize(const Entity& target,
                                          std::string_view expression,
                                          const Parameters* parameters,
                                          ThermalNetwork* network);

// A formula that is a number.
[[nodiscard]] TemplatedFormula constant_formula(const Entity& target,
                                                double value);

// A formula template with named slots: `{a}` inside an entity
// (T{a}, GL({a},{b})) is a node slot, `{g}` on its own a number slot. The
// names are listed in slot order.
struct NamedTemplate {
    ShapeText text;
    std::vector<std::string> node_slots;
    std::vector<std::string> number_slots;
};
// Throws std::invalid_argument when the target is not one entity, a slot is
// malformed or a name is both a node and a number slot.
[[nodiscard]] NamedTemplate parse_template(std::string_view target,
                                           std::string_view expression);

// One input of a shape, read by LOAD.
struct ShapeInput {
    enum class Kind : std::uint8_t { Parameter, Entity };
    Kind kind{Kind::Parameter};
    // Into parameter_names / parameter_indices, or into entities.
    std::uint32_t index{0U};
};

// A parsed and lowered formula shape, shared by the formulas of that shape.
struct FormulaShape {
    ShapeText text;
    std::string key;
    // In LOAD order.
    std::vector<ShapeInput> inputs;
    std::vector<std::string> parameter_names;
    std::vector<Index> parameter_indices;
    std::vector<EntityPattern> entities;
    LoweredExpression lowered;
    FormulaKind kind{FormulaKind::Constant};
    // An entity read is the target itself.
    bool reads_target{false};
    const Parameters* parameters{nullptr};
    // The parameters' structure version the shape was built against.
    std::uint64_t parameters_version{0U};
    // The derivatives with respect to each input, lowered on first use.
    mutable std::shared_ptr<const std::vector<LoweredExpression>> derivatives;
};

// Parses and lowers a shape. Every name that is not a placeholder must be a
// parameter or, with a network, an existing entity. `display` is the formula
// as written, for the messages. Throws std::invalid_argument.
[[nodiscard]] std::shared_ptr<const FormulaShape> build_shape(
    ShapeText text, const Parameters* parameters, ThermalNetwork* network,
    std::string_view display);

// Throws std::runtime_error for an input without a derivative.
[[nodiscard]] const std::vector<LoweredExpression>& shape_derivatives(
    const FormulaShape& shape);

// The expression of one formula of `shape`. Negative numbers are put in
// parentheses.
[[nodiscard]] std::string print_expression(const FormulaShape& shape,
                                           std::span<const NodeNum> nodes,
                                           std::span<const double> numbers);

}  // namespace detail

}  // namespace pycanha
