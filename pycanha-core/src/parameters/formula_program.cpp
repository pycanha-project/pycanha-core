#include "pycanha-core/parameters/formula_program.hpp"

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/dict.h>
#include <symengine/eval_double.h>
#include <symengine/functions.h>
#include <symengine/lambda_double.h>
#include <symengine/mul.h>
#include <symengine/number.h>
#include <symengine/pow.h>
#include <symengine/symbol.h>
#include <symengine/symengine_casts.h>
#include <symengine/symengine_rcp.h>
#include <symengine/visitor.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/table.hpp"

namespace pycanha::detail {

namespace {

using SymEngine::Basic;

constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();
// Larger integer powers go through std::pow.
constexpr double max_multiplied_power = 64.0;

[[nodiscard]] Op make_op(OpCode code, std::uint32_t arg = 0U) {
    return Op{.code = code, .arg = arg};
}

[[nodiscard]] Op power_int_op(std::int32_t exponent) {
    return make_op(OpCode::POWER_INT, std::bit_cast<std::uint32_t>(exponent));
}

[[nodiscard]] std::uint32_t to_u32(std::size_t value) {
    return static_cast<std::uint32_t>(value);
}

// One step of the lowering: visit a node, visit base ^ exponent without
// building a Pow node, or emit an operation.
struct Action {
    enum class Kind : std::uint8_t { VISIT, POWER, EMIT };
    Kind kind{Kind::EMIT};
    const Basic* node{nullptr};
    const Basic* exponent{nullptr};
    Op op{};
};

[[nodiscard]] Action visit(const Basic& node) {
    return {.kind = Action::Kind::VISIT, .node = &node};
}

[[nodiscard]] Action emit(Op op) {
    return {.kind = Action::Kind::EMIT, .op = op};
}

// Unary functions with an operation, possibly through a reciprocal
// (cot x = 1 / tan x, acot x = atan(1 / x)). Appends the operations that
// follow the argument, or returns false for any other node.
[[nodiscard]] bool unary_ops(SymEngine::TypeID type, std::vector<Action>& out) {
    const auto inverse = emit(power_int_op(-1));
    const auto then = [&out](OpCode code) {
        out.push_back(emit(make_op(code)));
    };
    switch (type) {
        case SymEngine::SYMENGINE_SIN:
            then(OpCode::SIN);
            return true;
        case SymEngine::SYMENGINE_COS:
            then(OpCode::COS);
            return true;
        case SymEngine::SYMENGINE_TAN:
            then(OpCode::TAN);
            return true;
        case SymEngine::SYMENGINE_COT:
            then(OpCode::TAN);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_CSC:
            then(OpCode::SIN);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_SEC:
            then(OpCode::COS);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_ASIN:
            then(OpCode::ASIN);
            return true;
        case SymEngine::SYMENGINE_ACOS:
            then(OpCode::ACOS);
            return true;
        case SymEngine::SYMENGINE_ATAN:
            then(OpCode::ATAN);
            return true;
        case SymEngine::SYMENGINE_ACOT:
            out.push_back(inverse);
            then(OpCode::ATAN);
            return true;
        case SymEngine::SYMENGINE_ACSC:
            out.push_back(inverse);
            then(OpCode::ASIN);
            return true;
        case SymEngine::SYMENGINE_ASEC:
            out.push_back(inverse);
            then(OpCode::ACOS);
            return true;
        case SymEngine::SYMENGINE_SINH:
            then(OpCode::SINH);
            return true;
        case SymEngine::SYMENGINE_COSH:
            then(OpCode::COSH);
            return true;
        case SymEngine::SYMENGINE_TANH:
            then(OpCode::TANH);
            return true;
        case SymEngine::SYMENGINE_COTH:
            then(OpCode::TANH);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_CSCH:
            then(OpCode::SINH);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_SECH:
            then(OpCode::COSH);
            out.push_back(inverse);
            return true;
        case SymEngine::SYMENGINE_ASINH:
            then(OpCode::ASINH);
            return true;
        case SymEngine::SYMENGINE_ACOSH:
            then(OpCode::ACOSH);
            return true;
        case SymEngine::SYMENGINE_ATANH:
            then(OpCode::ATANH);
            return true;
        case SymEngine::SYMENGINE_ACOTH:
            out.push_back(inverse);
            then(OpCode::ATANH);
            return true;
        case SymEngine::SYMENGINE_ACSCH:
            out.push_back(inverse);
            then(OpCode::ASINH);
            return true;
        case SymEngine::SYMENGINE_ASECH:
            out.push_back(inverse);
            then(OpCode::ACOSH);
            return true;
        case SymEngine::SYMENGINE_LOG:
            then(OpCode::LOG);
            return true;
        case SymEngine::SYMENGINE_ABS:
            then(OpCode::ABS);
            return true;
        case SymEngine::SYMENGINE_SIGN:
            then(OpCode::SIGN);
            return true;
        case SymEngine::SYMENGINE_FLOOR:
            then(OpCode::FLOOR);
            return true;
        case SymEngine::SYMENGINE_CEILING:
            then(OpCode::CEILING);
            return true;
        default:
            return false;
    }
}

// Whether `tree` calls a function the evaluator resolves itself (a table
// call or its derivative), which SymEngine's evaluator cannot run.
[[nodiscard]] bool has_table_call(const Basic& tree) {
    const auto is_call = [](const Basic& node) {
        const auto type = node.get_type_code();
        return (type == SymEngine::SYMENGINE_FUNCTIONSYMBOL) ||
               (type == SymEngine::SYMENGINE_DERIVATIVE) ||
               (type == SymEngine::SYMENGINE_SUBS);
    };
    if (is_call(tree)) {
        return true;
    }
    // Owning pointers: get_args() builds some arguments on the spot (the
    // powers of a product), which would be gone by the time they are visited.
    SymEngine::vec_basic pending = tree.get_args();
    while (!pending.empty()) {
        const auto current = pending.back();
        pending.pop_back();
        if (is_call(*current)) {
            return true;
        }
        const auto arguments = current->get_args();
        pending.insert(pending.end(), arguments.begin(), arguments.end());
    }
    return false;
}

[[nodiscard]] Op coefficient_op(double coefficient, std::uint32_t number) {
    return coefficient == -1.0 ? make_op(OpCode::NEGATE)
                               : make_op(OpCode::MULTIPLY_NUMBER, number);
}

class Lowering {
  public:
    Lowering(const std::vector<SymEngine::RCP<const SymEngine::Symbol>>& inputs,
             std::span<const std::string> slots, const Parameters* parameters)
        : _parameters(parameters) {
        std::uint32_t index = 0U;
        for (const auto& input : inputs) {
            _inputs.emplace(input->get_name(), index++);
        }
        index = 0U;
        for (const auto& slot : slots) {
            _numbers.emplace(slot, index++);
        }
    }

    LoweredExpression run(const Basic& tree) {
        std::vector<Action> pending{visit(tree)};
        std::vector<Action> expansion;
        while (!pending.empty()) {
            const Action action = pending.back();
            pending.pop_back();
            if (action.kind == Action::Kind::EMIT) {
                _result.ops.push_back(action.op);
                continue;
            }
            expansion.clear();
            if (action.kind == Action::Kind::POWER) {
                expand_power(*action.node, *action.exponent, expansion);
            } else {
                expand(*action.node, expansion);
            }
            pending.insert(pending.end(), expansion.rbegin(), expansion.rend());
        }
        _result.stack_depth = stack_depth(_result.ops);
        return std::move(_result);
    }

  private:
    [[nodiscard]] Action number(double value) {
        _result.numbers.push_back(value);
        return emit(
            make_op(OpCode::NUMBER, to_u32(_result.numbers.size() - 1U)));
    }

    [[nodiscard]] Action with_number(OpCode code, double value) {
        _result.numbers.push_back(value);
        return emit(make_op(code, to_u32(_result.numbers.size() - 1U)));
    }

    // Multiplies the top by `coefficient`, if it is not 1.
    void scale(double coefficient, std::vector<Action>& out) {
        if (coefficient == 1.0) {
            return;
        }
        if (coefficient != -1.0) {
            _result.numbers.push_back(coefficient);
        }
        out.push_back(emit(
            coefficient_op(coefficient, to_u32(_result.numbers.size() - 1U))));
    }

    // The slot of a symbol that is one of the formula's numbers.
    [[nodiscard]] std::optional<std::uint32_t> slot_of(
        const Basic& node) const {
        if (node.get_type_code() != SymEngine::SYMENGINE_SYMBOL) {
            return std::nullopt;
        }
        const auto found = _numbers.find(
            SymEngine::down_cast<const SymEngine::Symbol&>(node).get_name());
        if (found == _numbers.end()) {
            return std::nullopt;
        }
        return found->second;
    }

    [[nodiscard]] std::uint32_t table_slot(const std::string& name) {
        const auto found = std::ranges::find(_result.tables, name);
        if (found != _result.tables.end()) {
            return to_u32(
                static_cast<std::size_t>(found - _result.tables.begin()));
        }
        _result.tables.push_back(name);
        return to_u32(_result.tables.size() - 1U);
    }

    void expand(const Basic& node, std::vector<Action>& out) {
        if (SymEngine::is_a_Number(node) ||
            (node.get_type_code() == SymEngine::SYMENGINE_CONSTANT)) {
            out.push_back(number(SymEngine::eval_double(node)));
            return;
        }
        switch (node.get_type_code()) {
            case SymEngine::SYMENGINE_SYMBOL:
            case SymEngine::SYMENGINE_DUMMY:
                out.push_back(load(node));
                return;
            case SymEngine::SYMENGINE_ADD:
                expand_add(SymEngine::down_cast<const SymEngine::Add&>(node),
                           out);
                return;
            case SymEngine::SYMENGINE_MUL:
                expand_mul(SymEngine::down_cast<const SymEngine::Mul&>(node),
                           out);
                return;
            case SymEngine::SYMENGINE_POW: {
                const auto& power =
                    SymEngine::down_cast<const SymEngine::Pow&>(node);
                expand_power(*power.get_base(), *power.get_exp(), out);
                return;
            }
            case SymEngine::SYMENGINE_FUNCTIONSYMBOL:
                expand_table_call(node, out);
                return;
            case SymEngine::SYMENGINE_DERIVATIVE:
            case SymEngine::SYMENGINE_SUBS:
                expand_table_slope(node, out);
                return;
            case SymEngine::SYMENGINE_ATAN2:
            case SymEngine::SYMENGINE_MAX:
            case SymEngine::SYMENGINE_MIN:
                expand_multi_argument(node, out);
                return;
            default:
                break;
        }
        const auto arguments = node.get_args();
        std::vector<Action> unary;
        if ((arguments.size() == 1U) &&
            unary_ops(node.get_type_code(), unary)) {
            out.push_back(visit(*arguments.front()));
            out.insert(out.end(), unary.begin(), unary.end());
            return;
        }
        out.push_back(escape(node));
    }

    [[nodiscard]] Action load(const Basic& symbol) const {
        if (const auto slot = slot_of(symbol); slot.has_value()) {
            return emit(make_op(OpCode::SLOT, *slot));
        }
        const auto& name =
            SymEngine::down_cast<const SymEngine::Symbol&>(symbol).get_name();
        const auto input = _inputs.find(name);
        if (input == _inputs.end()) {
            throw std::invalid_argument("Unknown symbol '" + name + "'");
        }
        return emit(make_op(OpCode::LOAD, input->second));
    }

    // One term k * t of a sum: the first one starts the sum, the others are
    // added, with k = 1 and k = -1 folded into ADD and SUBTRACT.
    void append_term(const Basic& term, double factor, bool first,
                     std::vector<Action>& out) {
        out.push_back(visit(term));
        if (first) {
            scale(factor, out);
            return;
        }
        if (factor == -1.0) {
            out.push_back(emit(make_op(OpCode::SUBTRACT)));
            return;
        }
        scale(factor, out);
        out.push_back(emit(make_op(OpCode::ADD)));
    }

    // c + k1 * t1 + k2 * t2 + .... Terms that are numbers of the formula
    // (k = +-1) come last, as ADD_SLOT / SUBTRACT_SLOT.
    void expand_add(const SymEngine::Add& add, std::vector<Action>& out) {
        std::vector<Op> slot_terms;
        bool first = true;
        for (const auto& [term, coefficient] : add.get_dict()) {
            const double factor = SymEngine::eval_double(*coefficient);
            const auto slot = slot_of(*term);
            if (slot.has_value() && (std::abs(factor) == 1.0)) {
                slot_terms.push_back(make_op(
                    factor > 0.0 ? OpCode::ADD_SLOT : OpCode::SUBTRACT_SLOT,
                    *slot));
                continue;
            }
            append_term(*term, factor, first, out);
            first = false;
        }
        for (const auto& op : slot_terms) {
            if (first) {
                first = false;
                out.push_back(emit(make_op(OpCode::SLOT, op.arg)));
                if (op.code == OpCode::SUBTRACT_SLOT) {
                    out.push_back(emit(make_op(OpCode::NEGATE)));
                }
                continue;
            }
            out.push_back(emit(op));
        }
        const double constant = SymEngine::eval_double(*add.get_coef());
        if (constant != 0.0) {
            out.push_back(with_number(OpCode::ADD_NUMBER, constant));
        }
    }

    // The value of c * (numerator factors): the factors, or the first
    // multiplying number of the formula when there are none, then c.
    void append_numerator(
        const std::vector<std::pair<const Basic*, const Basic*>>& numerator,
        double coefficient, std::vector<Op>& slot_factors,
        std::vector<Action>& out) {
        if (!numerator.empty()) {
            append_product(numerator, out);
            scale(coefficient, out);
            return;
        }
        const auto first_slot =
            std::ranges::find(slot_factors, OpCode::MULTIPLY_SLOT, &Op::code);
        if (first_slot == slot_factors.end()) {
            out.push_back(number(coefficient));
            return;
        }
        out.push_back(emit(make_op(OpCode::SLOT, first_slot->arg)));
        slot_factors.erase(first_slot);
        scale(coefficient, out);
    }

    // c * (numerator factors) / (denominator factors): factors with a
    // negative number exponent are divided once at the end instead of each
    // being inverted. Numbers of the formula (exponent +-1) are applied
    // with MULTIPLY_SLOT / DIVIDE_SLOT.
    void expand_mul(const SymEngine::Mul& mul, std::vector<Action>& out) {
        std::vector<std::pair<const Basic*, const Basic*>> numerator;
        std::vector<std::pair<const Basic*, const Basic*>> denominator;
        std::vector<Op> slot_factors;
        for (const auto& [base, exponent] : mul.get_dict()) {
            const bool number_exponent = SymEngine::is_a_Number(*exponent);
            const double value =
                number_exponent ? SymEngine::eval_double(*exponent) : 0.0;
            if (const auto slot = slot_of(*base); slot.has_value() &&
                                                  number_exponent &&
                                                  (std::abs(value) == 1.0)) {
                slot_factors.push_back(make_op(
                    value > 0.0 ? OpCode::MULTIPLY_SLOT : OpCode::DIVIDE_SLOT,
                    *slot));
            } else if (number_exponent && (value < 0.0)) {
                _kept.push_back(SymEngine::neg(exponent));
                denominator.emplace_back(base.get(), _kept.back().get());
            } else {
                numerator.emplace_back(base.get(), exponent.get());
            }
        }

        append_numerator(numerator, SymEngine::eval_double(*mul.get_coef()),
                         slot_factors, out);
        std::ranges::transform(slot_factors, std::back_inserter(out), emit);
        if (!denominator.empty()) {
            append_product(denominator, out);
            out.push_back(emit(make_op(OpCode::DIVIDE)));
        }
    }

    static void append_product(
        const std::vector<std::pair<const Basic*, const Basic*>>& factors,
        std::vector<Action>& out) {
        bool first = true;
        for (const auto& [base, exponent] : factors) {
            out.push_back({.kind = Action::Kind::POWER,
                           .node = base,
                           .exponent = exponent});
            if (!first) {
                out.push_back(emit(make_op(OpCode::MULTIPLY)));
            }
            first = false;
        }
    }

    // base ^ exponent: exp() for base e, the base alone for 1, repeated
    // multiplication for small integers, sqrt for +-1/2.
    void expand_power(const Basic& base, const Basic& exponent,
                      std::vector<Action>& out) {
        if (SymEngine::eq(base, *SymEngine::E)) {
            out.push_back(visit(exponent));
            out.push_back(emit(make_op(OpCode::EXP)));
            return;
        }
        if (!SymEngine::is_a_Number(exponent)) {
            out.push_back(visit(base));
            out.push_back(visit(exponent));
            out.push_back(emit(make_op(OpCode::POWER)));
            return;
        }

        const double value = SymEngine::eval_double(exponent);
        out.push_back(visit(base));
        if (value == 1.0) {
            return;
        }
        if ((std::trunc(value) == value) &&
            (std::abs(value) <= max_multiplied_power)) {
            out.push_back(emit(power_int_op(static_cast<std::int32_t>(value))));
            return;
        }
        if (std::abs(value) == 0.5) {
            out.push_back(emit(make_op(OpCode::SQRT)));
            if (value < 0.0) {
                out.push_back(emit(power_int_op(-1)));
            }
            return;
        }
        out.push_back(number(value));
        out.push_back(emit(make_op(OpCode::POWER)));
    }

    static void expand_multi_argument(const Basic& node,
                                      std::vector<Action>& out) {
        const auto arguments = node.get_args();
        OpCode code = OpCode::ATAN2;
        if (node.get_type_code() == SymEngine::SYMENGINE_MAX) {
            code = OpCode::MAX;
        } else if (node.get_type_code() == SymEngine::SYMENGINE_MIN) {
            code = OpCode::MIN;
        }
        bool first = true;
        for (const auto& argument : arguments) {
            out.push_back(visit(*argument));
            if (!first) {
                out.push_back(emit(make_op(code)));
            }
            first = false;
        }
    }

    [[nodiscard]] const std::string& table_name(const Basic& call) const {
        const auto& name =
            SymEngine::down_cast<const SymEngine::FunctionSymbol&>(call)
                .get_name();
        if ((_parameters == nullptr) || !_parameters->has_table(name)) {
            throw std::invalid_argument("Unknown function '" + name +
                                        "': no table has that name");
        }
        if (call.get_args().size() != 1U) {
            throw std::invalid_argument("Table '" + name +
                                        "' takes exactly one argument, not " +
                                        std::to_string(call.get_args().size()));
        }
        return name;
    }

    void expand_table_call(const Basic& call, std::vector<Action>& out) {
        const auto slot = table_slot(table_name(call));
        out.push_back(visit(*call.get_args().front()));
        out.push_back(emit(make_op(OpCode::TABLE, slot)));
    }

    // The derivative of a table call, as differentiation writes it:
    // Derivative(f(x), x) for a symbol argument and
    // Subs(Derivative(f(_x), _x), _x, argument) otherwise.
    void expand_table_slope(const Basic& node, std::vector<Action>& out) {
        const Basic* derivative = &node;
        const Basic* argument = nullptr;
        if (node.get_type_code() == SymEngine::SYMENGINE_SUBS) {
            const auto& subs =
                SymEngine::down_cast<const SymEngine::Subs&>(node);
            if (subs.get_dict().size() == 1U) {
                derivative = subs.get_arg().get();
                argument = subs.get_dict().begin()->second.get();
            }
        }
        const Basic* call = nullptr;
        if (derivative->get_type_code() == SymEngine::SYMENGINE_DERIVATIVE) {
            const auto& of =
                SymEngine::down_cast<const SymEngine::Derivative&>(*derivative);
            const Basic* candidate = of.get_arg().get();
            if ((of.get_symbols().size() == 1U) &&
                (candidate->get_type_code() ==
                 SymEngine::SYMENGINE_FUNCTIONSYMBOL) &&
                (candidate->get_args().size() == 1U) &&
                SymEngine::eq(*candidate->get_args().front(),
                              **of.get_symbols().begin())) {
                call = candidate;
            }
        }
        if (call == nullptr) {
            throw std::invalid_argument("Cannot evaluate '" + node.__str__() +
                                        "'");
        }
        const auto slot = table_slot(table_name(*call));
        out.push_back(
            visit(argument == nullptr ? *call->get_args().front() : *argument));
        out.push_back(emit(make_op(OpCode::TABLE_SLOPE, slot)));
    }

    // A node without an operation: SymEngine's double evaluator runs it from
    // the expression inputs it reads.
    [[nodiscard]] Action escape(const Basic& node) {
        if (has_table_call(node)) {
            throw std::invalid_argument(
                "Cannot evaluate '" + node.__str__() +
                "': a table call is not supported inside this function");
        }
        LoweredEscape lowered;
        SymEngine::vec_basic symbols;
        for (const auto& symbol : SymEngine::free_symbols(node)) {
            if (const auto slot = slot_of(*symbol); slot.has_value()) {
                lowered.inputs.push_back(escape_slot_flag | *slot);
                symbols.push_back(symbol);
                continue;
            }
            const auto& name =
                SymEngine::down_cast<const SymEngine::Symbol&>(*symbol)
                    .get_name();
            const auto input = _inputs.find(name);
            if (input == _inputs.end()) {
                throw std::invalid_argument("Unknown symbol '" + name + "'");
            }
            lowered.inputs.push_back(input->second);
            symbols.push_back(symbol);
        }
        lowered.visitor =
            std::make_shared<SymEngine::LambdaRealDoubleVisitor>();
        try {
            lowered.visitor->init(symbols, node, false);
        } catch (const std::exception& exception) {
            throw std::invalid_argument("Cannot evaluate '" + node.__str__() +
                                        "': " + exception.what());
        }
        _result.escapes.push_back(std::move(lowered));
        return emit(
            make_op(OpCode::ESCAPE, to_u32(_result.escapes.size() - 1U)));
    }

    [[nodiscard]] static std::uint32_t stack_depth(std::span<const Op> ops) {
        std::int64_t depth = 0;
        std::int64_t deepest = 0;
        for (const auto& op : ops) {
            switch (op.code) {
                case OpCode::NUMBER:
                case OpCode::LOAD:
                case OpCode::SLOT:
                case OpCode::ESCAPE:
                    ++depth;
                    break;
                case OpCode::ADD:
                case OpCode::SUBTRACT:
                case OpCode::MULTIPLY:
                case OpCode::DIVIDE:
                case OpCode::POWER:
                case OpCode::ATAN2:
                case OpCode::MAX:
                case OpCode::MIN:
                case OpCode::STORE:
                    --depth;
                    break;
                default:
                    break;
            }
            deepest = std::max(deepest, depth);
        }
        return static_cast<std::uint32_t>(deepest);
    }

    const Parameters* _parameters;
    // Input symbols by name -> LOAD index; number symbols -> slot.
    std::unordered_map<std::string, std::uint32_t> _inputs;
    std::unordered_map<std::string, std::uint32_t> _numbers;
    // Negated exponents built while lowering, kept alive until it ends.
    std::vector<SymEngine::RCP<const Basic>> _kept;
    LoweredExpression _result;
};

[[nodiscard]] double power_int(double base, std::int32_t exponent) {
    const bool invert = exponent < 0;
    auto remaining = static_cast<std::uint32_t>(invert ? -exponent : exponent);
    double result = 1.0;
    double factor = base;
    while (remaining != 0U) {
        if ((remaining & 1U) != 0U) {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1U;
    }
    return invert ? 1.0 / result : result;
}

[[nodiscard]] double sign_of(double value) {
    if (value > 0.0) {
        return 1.0;
    }
    return value < 0.0 ? -1.0 : 0.0;
}

// The operations that take and give one value, kept out of run_ops so its
// loop stays small; they cost far more than this second dispatch.
[[nodiscard]] double unary(OpCode code, double value) {
    switch (code) {
        case OpCode::NEGATE:
            return -value;
        case OpCode::SQRT:
            return std::sqrt(value);
        case OpCode::EXP:
            return std::exp(value);
        case OpCode::LOG:
            return std::log(value);
        case OpCode::SIN:
            return std::sin(value);
        case OpCode::COS:
            return std::cos(value);
        case OpCode::TAN:
            return std::tan(value);
        case OpCode::ASIN:
            return std::asin(value);
        case OpCode::ACOS:
            return std::acos(value);
        case OpCode::ATAN:
            return std::atan(value);
        case OpCode::SINH:
            return std::sinh(value);
        case OpCode::COSH:
            return std::cosh(value);
        case OpCode::TANH:
            return std::tanh(value);
        case OpCode::ASINH:
            return std::asinh(value);
        case OpCode::ACOSH:
            return std::acosh(value);
        case OpCode::ATANH:
            return std::atanh(value);
        case OpCode::ABS:
            return std::abs(value);
        case OpCode::SIGN:
            return sign_of(value);
        case OpCode::FLOOR:
            return std::floor(value);
        case OpCode::CEILING:
            return std::ceil(value);
        default:
            return nan_value;
    }
}

[[nodiscard]] double binary(OpCode code, double left, double right) {
    switch (code) {
        case OpCode::POWER:
            return std::pow(left, right);
        case OpCode::ATAN2:
            return std::atan2(left, right);
        case OpCode::MAX:
            return std::max(left, right);
        case OpCode::MIN:
            return std::min(left, right);
        default:
            return nan_value;
    }
}

// `inputs` and `slots` start at the formula's own.
void run_escape(const ProgramView& program, const EscapeView& escape,
                std::span<const double* const> inputs,
                std::span<const double> slots, std::span<double> scratch,
                double* out) {
    for (std::uint32_t index = 0U; index < escape.input_count; ++index) {
        const std::uint32_t entry =
            program.escape_inputs[escape.first_input + index];
        scratch[index] = (entry & escape_slot_flag) != 0U
                             ? slots[entry & ~escape_slot_flag]
                             : *inputs[entry];
    }
    escape.visitor->call(out, scratch.data());
}

// The one-value operations over a column: the switch is outside the loop.
void unary_column(OpCode code, std::span<double> values) {
    const auto each = [values](auto function) {
        for (auto& value : values) {
            value = function(value);
        }
    };
    switch (code) {
        case OpCode::NEGATE:
            each([](double value) { return -value; });
            break;
        case OpCode::SQRT:
            each([](double value) { return std::sqrt(value); });
            break;
        case OpCode::EXP:
            each([](double value) { return std::exp(value); });
            break;
        case OpCode::LOG:
            each([](double value) { return std::log(value); });
            break;
        case OpCode::ABS:
            each([](double value) { return std::abs(value); });
            break;
        default:
            each([code](double value) { return unary(code, value); });
            break;
    }
}

// Applies `function` to every value of a column.
template <typename Function>
void each_value(std::span<double> values, Function function) {
    std::ranges::transform(values, values.begin(), function);
}

// values[k] = *inputs[k * stride] for every formula k of a block.
void load_column(std::span<double> values,
                 std::span<const double* const> inputs, std::size_t stride) {
    for (std::size_t formula = 0U; formula < values.size(); ++formula) {
        values[formula] = *inputs[formula * stride];
    }
}

// values[k] = numbers[k * stride] for every formula k of a block.
void number_column(std::span<double> values, std::span<const double> numbers,
                   std::size_t stride) {
    for (std::size_t formula = 0U; formula < values.size(); ++formula) {
        values[formula] = numbers[formula * stride];
    }
}

// left[k] = left[k] op right[k]: the switch is outside the loop.
void binary_columns(OpCode code, std::span<double> left,
                    std::span<const double> right) {
    const auto each = [left, right](auto function) {
        std::ranges::transform(left, right, left.begin(), function);
    };
    switch (code) {
        case OpCode::ADD:
            each([](double a, double b) { return a + b; });
            break;
        case OpCode::SUBTRACT:
            each([](double a, double b) { return a - b; });
            break;
        case OpCode::MULTIPLY:
            each([](double a, double b) { return a * b; });
            break;
        case OpCode::DIVIDE:
            each([](double a, double b) { return a / b; });
            break;
        default:
            each([code](double a, double b) { return binary(code, a, b); });
            break;
    }
}

// values[k] = values[k] op numbers[k * stride], for the *_SLOT operations.
void slot_columns(OpCode code, std::span<double> values,
                  std::span<const double> numbers, std::size_t stride) {
    const auto each = [values, numbers, stride](auto function) {
        for (std::size_t formula = 0U; formula < values.size(); ++formula) {
            values[formula] =
                function(values[formula], numbers[formula * stride]);
        }
    };
    switch (code) {
        case OpCode::ADD_SLOT:
            each([](double a, double b) { return a + b; });
            break;
        case OpCode::SUBTRACT_SLOT:
            each([](double a, double b) { return a - b; });
            break;
        case OpCode::MULTIPLY_SLOT:
            each([](double a, double b) { return a * b; });
            break;
        default:
            each([](double a, double b) { return a / b; });
            break;
    }
}

}  // namespace

LoweredExpression lower_expression(
    const Basic& tree,
    const std::vector<SymEngine::RCP<const SymEngine::Symbol>>& inputs,
    std::span<const std::string> slots, const Parameters* parameters) {
    return Lowering(inputs, slots, parameters).run(tree);
}

double run_formulas(const ProgramView& program, const ProgramRun& run) {
    // Few values stay live through the loop, so the calls inside it (math
    // functions, a table lookup's slow path) have little to save around
    // them: the top of the stack is a local, the values under it are in
    // `below`, and each formula's inputs, numbers and target are reached
    // through views that STORE moves on to the next formula.
    const auto ops = program.ops.subspan(run.op_begin);
    const auto numbers = program.numbers;
    const auto below = program.stack;
    auto inputs = program.inputs.subspan(run.input_begin);
    auto slots = program.slots.subspan(run.slot_begin);
    auto targets = program.targets.subspan(run.first, run.count);
    double top = nan_value;
    std::size_t depth = 0U;
    // One dispatch loop for the whole run: STORE moves on to the next
    // formula and starts the operations again.
    std::size_t position = 0U;
    while (true) {
        const Op op = ops[position++];
        switch (op.code) {
            case OpCode::NUMBER:
                below[depth++] = top;
                top = numbers[op.arg];
                break;
            case OpCode::LOAD:
                below[depth++] = top;
                top = *inputs[op.arg];
                break;
            case OpCode::SLOT:
                below[depth++] = top;
                top = slots[op.arg];
                break;
            case OpCode::ADD:
                top = below[--depth] + top;
                break;
            case OpCode::SUBTRACT:
                top = below[--depth] - top;
                break;
            case OpCode::MULTIPLY:
                top = below[--depth] * top;
                break;
            case OpCode::DIVIDE:
                top = below[--depth] / top;
                break;
            case OpCode::POWER:
            case OpCode::ATAN2:
            case OpCode::MAX:
            case OpCode::MIN:
                top = binary(op.code, below[--depth], top);
                break;
            case OpCode::ADD_NUMBER:
                top += numbers[op.arg];
                break;
            case OpCode::MULTIPLY_NUMBER:
                top *= numbers[op.arg];
                break;
            case OpCode::ADD_SLOT:
                top += slots[op.arg];
                break;
            case OpCode::SUBTRACT_SLOT:
                top -= slots[op.arg];
                break;
            case OpCode::MULTIPLY_SLOT:
                top *= slots[op.arg];
                break;
            case OpCode::DIVIDE_SLOT:
                top /= slots[op.arg];
                break;
            case OpCode::POWER_INT:
                top = power_int(top, std::bit_cast<std::int32_t>(op.arg));
                break;
            case OpCode::TABLE:
                top = table_value(program.tables[op.arg], top);
                break;
            case OpCode::TABLE_SLOPE:
                top = table_slope(program.tables[op.arg], top);
                break;
            case OpCode::ESCAPE:
                below[depth++] = top;
                // The visitor writes into storage the analyzers cannot
                // bound, rather than into a local.
                run_escape(program, program.escapes[op.arg], inputs, slots,
                           program.scratch, &below[depth]);
                top = below[depth];
                break;
            case OpCode::STORE:
                *targets.front() = top;
                if (targets.size() == 1U) {
                    return top;
                }
                targets = targets.subspan(1U);
                inputs = inputs.subspan(run.input_stride);
                slots = slots.subspan(run.slot_stride);
                position = 0U;
                depth = 0U;
                break;
            default:
                top = unary(op.code, top);
                break;
        }
    }
}

void run_block(const ProgramView& program, const ProgramRun& run,
               std::size_t begin, std::size_t count, std::span<double> columns,
               std::span<double> scratch) {
    const auto ops = program.ops.subspan(run.op_begin);
    const std::size_t input_stride = run.input_stride;
    const std::size_t slot_stride = run.slot_stride;
    const auto inputs =
        program.inputs.subspan(run.input_begin + (begin * input_stride));
    const auto slots =
        program.slots.subspan(run.slot_begin + (begin * slot_stride));
    const auto targets = program.targets.subspan(run.first + begin, count);
    // Column i holds stack level i of every formula of the block.
    const auto column = [columns, count](std::size_t level) {
        return columns.subspan(level * block_formulas, count);
    };
    std::size_t depth = 0U;
    for (std::size_t position = 0U;; ++position) {
        const Op op = ops[position];
        switch (op.code) {
            case OpCode::NUMBER:
                std::ranges::fill(column(depth++), program.numbers[op.arg]);
                break;
            case OpCode::LOAD:
                load_column(column(depth++), inputs.subspan(op.arg),
                            input_stride);
                break;
            case OpCode::SLOT:
                number_column(column(depth++), slots.subspan(op.arg),
                              slot_stride);
                break;
            case OpCode::ADD:
            case OpCode::SUBTRACT:
            case OpCode::MULTIPLY:
            case OpCode::DIVIDE:
            case OpCode::POWER:
            case OpCode::ATAN2:
            case OpCode::MAX:
            case OpCode::MIN:
                --depth;
                binary_columns(op.code, column(depth - 1U), column(depth));
                break;
            case OpCode::ADD_NUMBER: {
                const double number = program.numbers[op.arg];
                each_value(column(depth - 1U),
                           [number](double value) { return value + number; });
                break;
            }
            case OpCode::MULTIPLY_NUMBER: {
                const double number = program.numbers[op.arg];
                each_value(column(depth - 1U),
                           [number](double value) { return value * number; });
                break;
            }
            case OpCode::ADD_SLOT:
            case OpCode::SUBTRACT_SLOT:
            case OpCode::MULTIPLY_SLOT:
            case OpCode::DIVIDE_SLOT:
                slot_columns(op.code, column(depth - 1U), slots.subspan(op.arg),
                             slot_stride);
                break;
            case OpCode::POWER_INT: {
                const auto exponent = std::bit_cast<std::int32_t>(op.arg);
                each_value(column(depth - 1U), [exponent](double value) {
                    return power_int(value, exponent);
                });
                break;
            }
            case OpCode::TABLE: {
                // A copy: the stores to the column would otherwise make the
                // compiler read the view again for every formula.
                const TableView table = program.tables[op.arg];
                each_value(column(depth - 1U), [table](double value) {
                    return table_value(table, value);
                });
                break;
            }
            case OpCode::TABLE_SLOPE: {
                const TableView table = program.tables[op.arg];
                each_value(column(depth - 1U), [table](double value) {
                    return table_slope(table, value);
                });
                break;
            }
            case OpCode::ESCAPE: {
                const auto values = column(depth++);
                const auto& escape = program.escapes[op.arg];
                for (std::size_t formula = 0U; formula < count; ++formula) {
                    run_escape(program, escape,
                               inputs.subspan(formula * input_stride),
                               slots.subspan(formula * slot_stride), scratch,
                               &values[formula]);
                }
                break;
            }
            case OpCode::STORE: {
                const auto values = column(depth - 1U);
                for (std::size_t formula = 0U; formula < count; ++formula) {
                    *targets[formula] = values[formula];
                }
                return;
            }
            default:
                unary_column(op.code, column(depth - 1U));
                break;
        }
    }
}

double evaluate_lowered(const LoweredExpression& lowered,
                        std::span<const double> input_values,
                        std::span<const double> slot_values,
                        std::span<const TableView> tables) {
    std::vector<double> values(input_values.begin(), input_values.end());
    std::vector<const double*> inputs(values.size());
    std::ranges::transform(values, inputs.begin(),
                           [](double& value) { return &value; });
    std::vector<Op> ops(lowered.ops.begin(), lowered.ops.end());
    ops.push_back(make_op(OpCode::STORE));
    // STORE writes through targets; the result lands in the vector.
    std::vector<double> result(1U, nan_value);
    const std::vector<double*> targets{result.data()};

    std::vector<EscapeView> escapes;
    std::vector<std::uint32_t> escape_inputs;
    std::size_t widest = 0U;
    for (const auto& escape : lowered.escapes) {
        escapes.push_back({.visitor = escape.visitor.get(),
                           .first_input = to_u32(escape_inputs.size()),
                           .input_count = to_u32(escape.inputs.size())});
        escape_inputs.insert(escape_inputs.end(), escape.inputs.begin(),
                             escape.inputs.end());
        widest = std::max(widest, escape.inputs.size());
    }

    std::vector<double> stack(std::size_t{lowered.stack_depth} + 1U);
    std::vector<double> scratch(widest);
    const ProgramView program{.ops = ops,
                              .numbers = lowered.numbers,
                              .inputs = inputs,
                              .slots = slot_values,
                              .targets = targets,
                              .tables = tables,
                              .escapes = escapes,
                              .escape_inputs = escape_inputs,
                              .callables = {},
                              .stack = stack,
                              .scratch = scratch,
                              .columns = {},
                              .scratch_count = scratch.size(),
                              .column_count = 0U};
    const ProgramRun run{.op_begin = 0U,
                         .first = 0U,
                         .count = 1U,
                         .input_begin = 0U,
                         .input_stride = 0U,
                         .slot_begin = 0U,
                         .slot_stride = 0U,
                         .callable_begin = 0U,
                         .callables = false,
                         .blocks = false,
                         .watch = no_watch};
    return run_formulas(program, run);
}

ExecutionOrder execution_order(const FormulaPointers& pointers) {
    const std::size_t count = pointers.targets.size();
    std::vector<const double*> sorted_targets(pointers.targets.begin(),
                                              pointers.targets.end());
    std::ranges::sort(sorted_targets);
    auto reads = pointers.entity_reads;
    std::ranges::sort(reads);

    const auto reads_a_target = [&](std::size_t index) {
        const auto own = std::span(pointers.inputs)
                             .subspan(pointers.first_input[index],
                                      pointers.first_input[index + 1U] -
                                          pointers.first_input[index]);
        return std::ranges::any_of(own, [&](const double* input) {
            return std::ranges::binary_search(sorted_targets, input);
        });
    };
    // The first reader of a target, by (pointer, formula) order.
    const auto read_earlier = [&](std::size_t index) {
        const std::pair<const double*, std::size_t> first_possible{
            pointers.targets[index], 0U};
        const auto reader = std::ranges::lower_bound(reads, first_possible);
        return (reader != reads.end()) &&
               (reader->first == pointers.targets[index]) &&
               (reader->second < index);
    };

    ExecutionOrder result;
    result.order.reserve(count);
    std::vector<std::size_t> dependent;
    for (std::size_t index = 0U; index < count; ++index) {
        const bool independent = (index < pointers.first_callable) &&
                                 !reads_a_target(index) && !read_earlier(index);
        (independent ? result.order : dependent).push_back(index);
    }
    result.independent = result.order.size();
    result.order.insert(result.order.end(), dependent.begin(), dependent.end());
    return result;
}

void FormulaProgram::reserve(std::size_t formulas, std::size_t inputs,
                             std::size_t slots) {
    _targets.reserve(formulas);
    _formulas.reserve(formulas);
    _inputs.reserve(inputs);
    _slots.reserve(slots);
}

std::uint32_t FormulaProgram::table_slot(const TableView& table) {
    // Each table has its own range record, so it identifies the table.
    const auto found =
        std::ranges::find(_tables, table.range, &TableView::range);
    if (found != _tables.end()) {
        return to_u32(static_cast<std::size_t>(found - _tables.begin()));
    }
    _tables.push_back(table);
    return to_u32(_tables.size() - 1U);
}

Op FormulaProgram::fold_slot(Op op,
                             std::span<const std::optional<double>> shared,
                             std::span<const std::int64_t> stored_index) {
    const auto stored = stored_index[op.arg];
    const auto& shared_value = shared[op.arg];
    if ((stored >= 0) || !shared_value.has_value()) {
        op.arg = static_cast<std::uint32_t>(stored);
        return op;
    }
    const double value = *shared_value;
    const auto number = [this](double folded) {
        _numbers.push_back(folded);
        return to_u32(_numbers.size() - 1U);
    };
    switch (op.code) {
        case OpCode::SLOT:
            return make_op(OpCode::NUMBER, number(value));
        case OpCode::ADD_SLOT:
            return make_op(OpCode::ADD_NUMBER, number(value));
        case OpCode::SUBTRACT_SLOT:
            return make_op(OpCode::ADD_NUMBER, number(-value));
        case OpCode::MULTIPLY_SLOT:
            return make_op(OpCode::MULTIPLY_NUMBER, number(value));
        default:
            return make_op(OpCode::MULTIPLY_NUMBER, number(1.0 / value));
    }
}

void FormulaProgram::push_op(Op op, std::uint32_t run_begin) {
    // Folded numbers often leave two multiplications (or additions) by a
    // number in a row, e.g. 0.01 * (Ti + Tj) / 2: they become one.
    if ((_ops.size() > run_begin) &&
        ((op.code == OpCode::MULTIPLY_NUMBER) ||
         (op.code == OpCode::ADD_NUMBER)) &&
        (_ops.back().code == op.code)) {
        const double previous = _numbers[_ops.back().arg];
        _numbers.push_back(op.code == OpCode::MULTIPLY_NUMBER
                               ? previous * _numbers[op.arg]
                               : previous + _numbers[op.arg]);
        _ops.back().arg = to_u32(_numbers.size() - 1U);
        return;
    }
    _ops.push_back(op);
}

void FormulaProgram::open_run(const LoweredExpression& lowered,
                              std::uint32_t input_count,
                              std::span<const std::optional<double>> shared,
                              std::span<const TableView> tables) {
    close_run();
    // An escape reads its numbers from the formula's own, so those stay
    // stored even when shared.
    std::vector<bool> escaped(shared.size(), false);
    for (const auto& escape : lowered.escapes) {
        for (const auto input : escape.inputs) {
            if ((input & escape_slot_flag) != 0U) {
                escaped[input & ~escape_slot_flag] = true;
            }
        }
    }
    std::vector<std::int64_t> stored_index(shared.size(), -1);
    _stored_slots.clear();
    for (std::size_t slot = 0U; slot < shared.size(); ++slot) {
        if (!shared[slot].has_value() || escaped[slot]) {
            stored_index[slot] =
                static_cast<std::int64_t>(_stored_slots.size());
            _stored_slots.push_back(to_u32(slot));
        }
    }

    const auto number_base = to_u32(_numbers.size());
    const auto escape_base = to_u32(_escapes.size());
    _numbers.insert(_numbers.end(), lowered.numbers.begin(),
                    lowered.numbers.end());
    std::vector<std::uint32_t> table_slots(tables.size());
    std::ranges::transform(
        tables, table_slots.begin(),
        [this](const TableView& table) { return table_slot(table); });
    for (const auto& escape : lowered.escapes) {
        _escapes.push_back({.visitor = escape.visitor.get(),
                            .first_input = to_u32(_escape_inputs.size()),
                            .input_count = to_u32(escape.inputs.size())});
        std::ranges::transform(
            escape.inputs, std::back_inserter(_escape_inputs),
            [&stored_index](std::uint32_t input) {
                if ((input & escape_slot_flag) == 0U) {
                    return input;
                }
                return escape_slot_flag |
                       static_cast<std::uint32_t>(
                           stored_index[input & ~escape_slot_flag]);
            });
        _widest = std::max(_widest, escape.inputs.size());
    }

    const ProgramRun run{.op_begin = to_u32(_ops.size()),
                         .first = to_u32(_targets.size()),
                         .count = 0U,
                         .input_begin = to_u32(_inputs.size()),
                         .input_stride = input_count,
                         .slot_begin = to_u32(_slots.size()),
                         .slot_stride = to_u32(_stored_slots.size()),
                         .callable_begin = 0U,
                         .callables = false,
                         .blocks = false,
                         .watch = no_watch};
    for (auto op : lowered.ops) {
        switch (op.code) {
            case OpCode::NUMBER:
            case OpCode::ADD_NUMBER:
            case OpCode::MULTIPLY_NUMBER:
                op.arg += number_base;
                break;
            case OpCode::SLOT:
            case OpCode::ADD_SLOT:
            case OpCode::SUBTRACT_SLOT:
            case OpCode::MULTIPLY_SLOT:
            case OpCode::DIVIDE_SLOT:
                op = fold_slot(op, shared, stored_index);
                break;
            case OpCode::TABLE:
            case OpCode::TABLE_SLOPE:
                op.arg = table_slots[op.arg];
                break;
            case OpCode::ESCAPE:
                op.arg += escape_base;
                break;
            default:
                break;
        }
        push_op(op, run.op_begin);
    }
    _ops.push_back(make_op(OpCode::STORE));
    _runs.push_back(run);
    _depth = std::max<std::size_t>(_depth, lowered.stack_depth);
}

void FormulaProgram::add_formula(FormulaRef formula, double* target,
                                 std::span<const double* const> inputs,
                                 std::span<const double> slots) {
    _targets.push_back(target);
    _formulas.push_back(formula);
    _inputs.insert(_inputs.end(), inputs.begin(), inputs.end());
    std::ranges::transform(_stored_slots, std::back_inserter(_slots),
                           [slots](std::uint32_t slot) { return slots[slot]; });
    ++_runs.back().count;
}

void FormulaProgram::add_callable(CallableFormula& callable, FormulaRef formula,
                                  double* target) {
    if (_runs.empty() || !_runs.back().callables) {
        close_run();
        _runs.push_back(ProgramRun{.op_begin = to_u32(_ops.size()),
                                   .first = to_u32(_targets.size()),
                                   .count = 0U,
                                   .input_begin = to_u32(_inputs.size()),
                                   .input_stride = 0U,
                                   .slot_begin = to_u32(_slots.size()),
                                   .slot_stride = 0U,
                                   .callable_begin = to_u32(_callables.size()),
                                   .callables = true,
                                   .blocks = false,
                                   .watch = no_watch});
    }
    ++_runs.back().count;
    _callables.push_back(&callable);
    _targets.push_back(target);
    _formulas.push_back(formula);
}

void FormulaProgram::close_run() {
    if (_runs.empty() || _runs.back().callables) {
        return;
    }
    auto& run = _runs.back();
    // Independent formulas read no formula target: nothing to check.
    if (!_independent.has_value() || (_runs.size() <= *_independent)) {
        run.blocks = true;
        return;
    }
    // A formula reading the target of another formula of the run must see
    // it written first, which a block (all loads before all stores) would
    // not do. Its own target is fine: it is read before it is written
    // either way.
    const auto targets = std::span(_targets).subspan(run.first, run.count);
    std::vector<const double*> sorted(targets.begin(), targets.end());
    std::ranges::sort(sorted);
    const auto inputs = std::span(_inputs).subspan(
        run.input_begin, std::size_t{run.count} * run.input_stride);
    run.blocks = true;
    for (std::size_t formula = 0U; formula < run.count; ++formula) {
        const auto own =
            inputs.subspan(formula * run.input_stride, run.input_stride);
        const bool reads_other =
            std::ranges::any_of(own, [&](const double* input) {
                return (input != targets[formula]) &&
                       std::ranges::binary_search(sorted, input);
            });
        if (reads_other) {
            run.blocks = false;
            return;
        }
    }
}

void FormulaProgram::watch_run(std::span<const std::uint64_t* const> changed) {
    auto& run = _runs.back();
    run.watch = to_u32(_watches.size());
    _watches.push_back({.first_changed = to_u32(_changed.size()),
                        .changed_count = to_u32(changed.size()),
                        .first_value = to_u32(_values.size()),
                        .evaluated = false,
                        .seen = 0U});
    _changed.insert(_changed.end(), changed.begin(), changed.end());
    _values.resize(_values.size() + run.count, 0.0);
}

bool FormulaProgram::replay(const ProgramRun& run, std::uint64_t changes) {
    auto& watch = _watches[run.watch];
    if (!watch.evaluated) {
        return false;
    }
    if (changes != watch.seen) {
        // Something changed since; the run only cares about what it reads.
        const auto changed = std::span(_changed).subspan(watch.first_changed,
                                                         watch.changed_count);
        const auto seen = watch.seen;
        if (std::ranges::any_of(changed, [seen](const std::uint64_t* last) {
                return *last > seen;
            })) {
            return false;
        }
        watch.seen = changes;
    }
    const auto values =
        std::span(_values).subspan(watch.first_value, run.count);
    const auto targets = std::span(_targets).subspan(run.first, run.count);
    for (std::size_t formula = 0U; formula < run.count; ++formula) {
        *targets[formula] = values[formula];
    }
    return true;
}

void FormulaProgram::record(const ProgramRun& run, std::uint64_t changes) {
    auto& watch = _watches[run.watch];
    const auto values =
        std::span(_values).subspan(watch.first_value, run.count);
    const auto targets = std::span(_targets).subspan(run.first, run.count);
    for (std::size_t formula = 0U; formula < run.count; ++formula) {
        values[formula] = *targets[formula];
    }
    watch.seen = changes;
    watch.evaluated = true;
}

void FormulaProgram::finish() {
    close_run();
    // One more than the deepest expression: an escape writes its result
    // just above the values under the top.
    _stack.assign(_depth + 1U, 0.0);
    _scratch.assign(scratch_stride(), 0.0);
    _plans.assign(_runs.size(), GroupPlan{});
    _columns.assign((_depth + 1U) * block_formulas, 0.0);
}

void FormulaProgram::end_independent() { _independent = _runs.size(); }

void FormulaProgram::reserve_workers(std::size_t workers) {
    const std::size_t columns = (_depth + 1U) * block_formulas;
    if (_columns.size() < workers * columns) {
        _columns.resize(workers * columns, 0.0);
        _scratch.resize(workers * scratch_stride(), 0.0);
    }
}

std::size_t FormulaProgram::scratch_stride() const noexcept {
    // A cache line of padding after each worker's values, so two workers
    // never write to the same line.
    constexpr std::size_t line = 64U / sizeof(double);
    return _widest == 0U ? 0U : (((_widest + line - 1U) / line) + 1U) * line;
}

ProgramView FormulaProgram::view() {
    return {.ops = _ops,
            .numbers = _numbers,
            .inputs = _inputs,
            .slots = _slots,
            .targets = _targets,
            .tables = _tables,
            .escapes = _escapes,
            .escape_inputs = _escape_inputs,
            .callables = _callables,
            .stack = _stack,
            .scratch = _scratch,
            .columns = _columns,
            .scratch_count = scratch_stride(),
            .column_count = (_depth + 1U) * block_formulas};
}

}  // namespace pycanha::detail
