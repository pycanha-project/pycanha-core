#pragma once

#include <symengine/basic.h>
#include <symengine/lambda_double.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/table.hpp"

namespace pycanha {

class Parameters;
class Formula;
class CallableFormula;

namespace detail {

// The operations of the formula evaluator. A formula is a postfix sequence
// of them working on a small stack of doubles.
enum class OpCode : std::uint8_t {
    // Push numbers[arg].
    NUMBER,
    // Push *inputs[arg].
    LOAD,
    // Pop b, then a; push a op b.
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
    POWER,
    ATAN2,
    MAX,
    MIN,
    // Top op= numbers[arg].
    ADD_NUMBER,
    MULTIPLY_NUMBER,
    // Top = top ^ arg, arg read as a signed integer; computed by
    // multiplication.
    POWER_INT,
    // Top = f(top).
    NEGATE,
    SQRT,
    EXP,
    LOG,
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    SINH,
    COSH,
    TANH,
    ASINH,
    ACOSH,
    ATANH,
    ABS,
    SIGN,
    FLOOR,
    CEILING,
    // Top = table value or slope at top, tables[arg].
    TABLE,
    TABLE_SLOPE,
    // Push the value of escapes[arg], an expression the evaluator has no
    // operation for, evaluated by SymEngine from its own inputs.
    ESCAPE,
    // Pop into *targets[arg].
    STORE,
};

struct Op {
    OpCode code{OpCode::NUMBER};
    std::uint32_t arg{0U};
};

// An expression the evaluator has no operation for. `inputs` are input
// slots of the expression, in the order the visitor expects them.
struct LoweredEscape {
    std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> visitor;
    std::vector<std::uint32_t> inputs;
};

// A formula expression lowered to evaluator operations. LOAD reads input
// slots (the expression's symbols in name order), TABLE / TABLE_SLOPE index
// `tables` and ESCAPE indexes `escapes`.
struct LoweredExpression {
    std::vector<Op> ops;
    std::vector<double> numbers;
    // The tables called, by name as written.
    std::vector<std::string> tables;
    std::vector<LoweredEscape> escapes;
    std::uint32_t stack_depth{0U};
};

// Lowers `tree`, whose symbols are `inputs`. A call `name(argument)` must
// name a table of `parameters` and take exactly one argument. Derivatives of
// table calls (as produced by differentiation) lower to TABLE_SLOPE. Throws
// std::invalid_argument for an unknown function, a table call with another
// number of arguments, and anything neither the evaluator nor SymEngine's
// double evaluator can run.
[[nodiscard]] LoweredExpression lower_expression(
    const SymEngine::Basic& tree,
    const std::vector<SymEngine::RCP<const SymEngine::Symbol>>& inputs,
    const Parameters& parameters);

// An escape inside a program: its inputs are escape_inputs[first_input,
// first_input + input_count), each an index into the program's inputs.
struct EscapeView {
    SymEngine::LambdaRealDoubleVisitor* visitor{nullptr};
    std::uint32_t first_input{0U};
    std::uint32_t input_count{0U};
};

// What the evaluator reads and writes while it runs.
struct ProgramView {
    std::span<const Op> ops;
    std::span<const double> numbers;
    std::span<double* const> inputs;
    std::span<double* const> targets;
    std::span<const TableView> tables;
    std::span<const EscapeView> escapes;
    std::span<const std::uint32_t> escape_inputs;
    // Sized for the deepest expression and the widest escape.
    std::span<double> stack;
    std::span<double> scratch;
};

// Runs ops[begin, end) and returns what is left on top of the stack, or NaN
// when nothing is.
double run_ops(const ProgramView& program, std::size_t begin, std::size_t end);

// Evaluates one lowered expression on the given input values, with the
// tables it calls in `tables` (same order as lowered.tables).
[[nodiscard]] double evaluate_lowered(const LoweredExpression& lowered,
                                      std::span<const double> input_values,
                                      std::span<const TableView> tables);

// The value pointers of a collection's formulas, in insertion order, as a
// compile collects them.
struct FormulaPointers {
    std::vector<double*> targets;
    // Inputs of every formula, flat: formula i reads
    // inputs[first_input[i], first_input[i + 1]).
    std::vector<double*> inputs;
    std::vector<std::size_t> first_input;
    // (input, formula) for every entity input.
    std::vector<std::pair<const double*, std::size_t>> entity_reads;
    // The position of the first callable formula, or the number of formulas.
    std::size_t first_callable{0U};
};

// The order a program runs the formulas in, giving the results of insertion
// order: first the formulas that read no formula target and whose target no
// earlier formula reads, then the others in insertion order. A callable
// reads everything, so no formula after the first one runs early.
[[nodiscard]] std::vector<std::size_t> execution_order(
    const FormulaPointers& pointers);

// A run of evaluator operations, or one callable formula.
struct ProgramRun {
    std::uint32_t op_begin{0U};
    std::uint32_t op_end{0U};
    CallableFormula* callable{nullptr};
    std::uint32_t target{0U};
};

// Formulas compiled into one sequence of operations, in execution order.
// Each formula ends with a STORE to its target.
class FormulaProgram {
  public:
    void reserve(std::size_t formulas, std::size_t inputs);
    void add_constant(const Formula& formula, double* target, double value);
    void add_callable(CallableFormula& formula, double* target);
    // `inputs` are the expression's input pointers and `tables` the views
    // of the tables it calls, in the order of lowered.tables.
    void add_expression(const Formula& formula, double* target,
                        const LoweredExpression& lowered,
                        std::span<double* const> inputs,
                        std::span<const TableView> tables);
    // Sizes the stack and scratch space for what was added.
    void finish();

    [[nodiscard]] ProgramView view();
    [[nodiscard]] std::span<const ProgramRun> runs() const noexcept {
        return _runs;
    }
    [[nodiscard]] std::span<const Op> ops() const noexcept { return _ops; }
    // The formula that writes targets[target], and the value it wrote.
    [[nodiscard]] const Formula& formula_of(std::uint32_t target) const {
        return *_target_formulas[target];
    }
    [[nodiscard]] double value_of(std::uint32_t target) const {
        return *_targets[target];
    }

  private:
    [[nodiscard]] std::uint32_t add_target(const Formula& formula,
                                           double* target);
    void open_run();
    [[nodiscard]] std::uint32_t table_slot(const TableView& table);
    void append_lowered_ops(const LoweredExpression& lowered,
                            std::uint32_t input_base, std::uint32_t number_base,
                            std::uint32_t escape_base,
                            std::span<const std::uint32_t> table_slots);

    std::vector<Op> _ops;
    std::vector<double> _numbers;
    std::vector<double*> _inputs;
    std::vector<double*> _targets;
    std::vector<TableView> _tables;
    std::vector<EscapeView> _escapes;
    std::vector<std::uint32_t> _escape_inputs;
    std::vector<ProgramRun> _runs;
    std::vector<const Formula*> _target_formulas;
    std::vector<double> _stack;
    std::vector<double> _scratch;
    std::size_t _depth{1U};
    std::size_t _widest{0U};
};

}  // namespace detail

}  // namespace pycanha
