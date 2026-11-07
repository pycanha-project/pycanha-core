#pragma once

#include <symengine/basic.h>
#include <symengine/lambda_double.h>
#include <symengine/symbol.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/parameters/table.hpp"

namespace pycanha {

class Parameters;
class CallableFormula;

namespace detail {

// The operations of the formula evaluator. A formula is a postfix sequence
// of them working on a small stack of doubles. Formulas of the same shape
// share their operations: NUMBER reads a number of the shape, SLOT a number
// of the formula itself.
enum class OpCode : std::uint8_t {
    // Push numbers[arg].
    NUMBER,
    // Push *inputs[arg].
    LOAD,
    // Push slots[arg].
    SLOT,
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
    // Top op= slots[arg].
    ADD_SLOT,
    SUBTRACT_SLOT,
    MULTIPLY_SLOT,
    DIVIDE_SLOT,
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
    // Pop into the formula's target.
    STORE,
};

struct Op {
    OpCode code{OpCode::NUMBER};
    std::uint32_t arg{0U};
};

// Marks an escape input that is a slot (a number of the formula) rather than
// an input slot.
inline constexpr std::uint32_t escape_slot_flag = 0x80000000U;

// An expression the evaluator has no operation for. `inputs` are input
// slots of the expression, or escape_slot_flag | slot for its numbers, in
// the order the visitor expects them.
struct LoweredEscape {
    std::shared_ptr<SymEngine::LambdaRealDoubleVisitor> visitor;
    std::vector<std::uint32_t> inputs;
};

// A formula expression lowered to evaluator operations. LOAD reads input
// slots (the expression's input symbols, in the order given), SLOT and the
// *_SLOT operations read the formula's own numbers, TABLE / TABLE_SLOPE
// index `tables` and ESCAPE indexes `escapes`.
struct LoweredExpression {
    std::vector<Op> ops;
    std::vector<double> numbers;
    // The tables called, by name as written.
    std::vector<std::string> tables;
    std::vector<LoweredEscape> escapes;
    std::uint32_t stack_depth{0U};
};

// Lowers `tree`. Its symbols are `inputs` (read with LOAD) and the symbols
// named in `slots` (read with SLOT, by position in `slots`). A call
// `name(argument)` must name a table of `parameters` and take exactly one
// argument. Derivatives of table calls (as produced by differentiation)
// lower to TABLE_SLOPE. Throws std::invalid_argument for an unknown
// function, a table call with another number of arguments, and anything
// neither the evaluator nor SymEngine's double evaluator can run.
[[nodiscard]] LoweredExpression lower_expression(
    const SymEngine::Basic& tree,
    const std::vector<SymEngine::RCP<const SymEngine::Symbol>>& inputs,
    std::span<const std::string> slots, const Parameters* parameters);

// An escape inside a program: its inputs are escape_inputs[first_input,
// first_input + input_count), each an input of the formula or one of its
// slots (escape_slot_flag).
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
    std::span<const double> slots;
    std::span<double* const> targets;
    std::span<const TableView> tables;
    std::span<const EscapeView> escapes;
    std::span<const std::uint32_t> escape_inputs;
    // Sized for the deepest expression and the widest escape.
    std::span<double> stack;
    std::span<double> scratch;
};

// Evaluates one lowered expression on the given input values and numbers,
// with the tables it calls in `tables` (same order as lowered.tables).
[[nodiscard]] double evaluate_lowered(const LoweredExpression& lowered,
                                      std::span<const double> input_values,
                                      std::span<const double> slot_values,
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
// order: first the `independent` formulas, which read no formula target and
// whose target no earlier formula reads, then the others in insertion
// order. A callable reads everything, so no formula after the first one
// runs early. Independent formulas may run in any order among themselves.
struct ExecutionOrder {
    std::vector<std::size_t> order;
    std::size_t independent{0U};
};
[[nodiscard]] ExecutionOrder execution_order(const FormulaPointers& pointers);

// Where a formula is stored in a Formulas collection: its batch and its
// position in the batch.
struct FormulaRef {
    std::uint32_t batch{0U};
    std::uint32_t instance{0U};

    [[nodiscard]] bool operator==(const FormulaRef&) const = default;
};

// Formulas that run the same operations, one after another, or one callable
// formula. Formula k of the run (from 0) is formula first + k of the
// program; its inputs start at input_begin + k * input_stride and its
// numbers at slot_begin + k * slot_stride.
struct ProgramRun {
    std::uint32_t op_begin{0U};
    std::uint32_t op_end{0U};
    std::uint32_t first{0U};
    std::uint32_t count{0U};
    std::uint32_t input_begin{0U};
    std::uint32_t input_stride{0U};
    std::uint32_t slot_begin{0U};
    std::uint32_t slot_stride{0U};
    CallableFormula* callable{nullptr};
};

// Runs every formula of a run of operations (not a callable run): STORE
// writes formula k's value to targets[first + k]. Returns what is left on
// top of the stack after the last formula, or NaN when nothing is.
double run_formulas(const ProgramView& program, const ProgramRun& run);

// Formulas compiled into runs, in execution order. The operations of a run
// are stored once and end with a STORE to the formula's target.
class FormulaProgram {
  public:
    void reserve(std::size_t formulas, std::size_t inputs, std::size_t slots);
    // Starts a run of formulas that evaluate `lowered`, each with
    // `input_count` inputs and its numbers. `tables` are the views of the
    // tables it calls, in the order of lowered.tables. `shared` has one
    // entry per number slot: a value every formula of the run has there, or
    // none. Shared numbers become operations on a number of the run, so the
    // formulas do not store them.
    void open_run(const LoweredExpression& lowered, std::uint32_t input_count,
                  std::span<const std::optional<double>> shared,
                  std::span<const TableView> tables);
    // Adds a formula to the run opened last; `slots` are all its numbers.
    void add_formula(FormulaRef formula, double* target,
                     std::span<double* const> inputs,
                     std::span<const double> slots);
    void add_callable(CallableFormula& callable, FormulaRef formula,
                      double* target);
    // Sizes the stack and scratch space for what was added.
    void finish();

    [[nodiscard]] ProgramView view();
    [[nodiscard]] std::span<const ProgramRun> runs() const noexcept {
        return _runs;
    }
    // The formula compiled at `position` and the value it wrote.
    [[nodiscard]] FormulaRef formula_at(std::size_t position) const {
        return _formulas[position];
    }
    [[nodiscard]] double value_at(std::size_t position) const {
        return *_targets[position];
    }

  private:
    [[nodiscard]] std::uint32_t table_slot(const TableView& table);
    // Appends `op` to the run starting at `run_begin`.
    void push_op(Op op, std::uint32_t run_begin);
    // The operation of the run for `op`, with the shared numbers folded in
    // and the stored ones renumbered (`stored_index`, -1 when shared).
    [[nodiscard]] Op fold_slot(Op op,
                               std::span<const std::optional<double>> shared,
                               std::span<const std::int64_t> stored_index);

    std::vector<Op> _ops;
    std::vector<double> _numbers;
    std::vector<double*> _inputs;
    std::vector<double> _slots;
    std::vector<double*> _targets;
    std::vector<FormulaRef> _formulas;
    std::vector<TableView> _tables;
    std::vector<EscapeView> _escapes;
    std::vector<std::uint32_t> _escape_inputs;
    std::vector<ProgramRun> _runs;
    // The number slots the formulas of the run opened last store.
    std::vector<std::uint32_t> _stored_slots;
    std::vector<double> _stack;
    std::vector<double> _scratch;
    std::size_t _depth{1U};
    std::size_t _widest{0U};
};

}  // namespace detail

}  // namespace pycanha
