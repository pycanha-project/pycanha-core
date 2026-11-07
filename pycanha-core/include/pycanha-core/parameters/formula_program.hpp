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
    std::span<const double* const> inputs;
    std::span<const double> slots;
    std::span<double* const> targets;
    std::span<const TableView> tables;
    std::span<const EscapeView> escapes;
    std::span<const std::uint32_t> escape_inputs;
    std::span<CallableFormula* const> callables;
    // The values under the top of the stack: one more than the deepest
    // expression needs, for an escape's result.
    std::span<double> stack;
    // One slice of scratch_count values (the widest escape's inputs, padded)
    // and one of column_count values (run_block's columns) for every worker
    // thread; run_formulas uses the first scratch slice.
    std::span<double> scratch;
    std::span<double> columns;
    std::size_t scratch_count{0U};
    std::size_t column_count{0U};
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
    std::vector<const double*> inputs;
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

// ProgramRun::watch of a run that is not watched.
inline constexpr std::uint32_t no_watch = 0xFFFFFFFFU;

// Formulas that run the same operations, one after another, or callable
// formulas that follow each other. Formula k of the run (from 0) is formula
// first + k of the program. In a run of operations its inputs start at
// input_begin + k * input_stride and its numbers at
// slot_begin + k * slot_stride; in a callable run it calls
// callables[callable_begin + k].
struct ProgramRun {
    std::uint32_t op_begin{0U};
    std::uint32_t first{0U};
    std::uint32_t count{0U};
    std::uint32_t input_begin{0U};
    std::uint32_t input_stride{0U};
    std::uint32_t slot_begin{0U};
    std::uint32_t slot_stride{0U};
    std::uint32_t callable_begin{0U};
    bool callables{false};
    // No formula of the run reads the target of another formula of the run,
    // so the run may be evaluated a block of formulas at a time.
    bool blocks{false};
    // For a run that reads only parameters (and tables): its entry in the
    // program's watched runs, else no_watch.
    std::uint32_t watch{no_watch};
};

// Runs every formula of a run of operations (not a callable run). The
// operations of each formula end with STORE, which writes formula k's value
// to targets[first + k]; the run ends after the last one. Returns the value
// the last formula stored.
double run_formulas(const ProgramView& program, const ProgramRun& run);

// The most formulas run_block evaluates at once.
inline constexpr std::size_t block_formulas = 256U;

// Evaluates formulas [begin, begin + count) of a run of operations (count at
// most block_formulas) one operation at a time over all of them: each
// operation is one tight loop over the formulas instead of one dispatch per
// formula. Valid only for a run whose `blocks` is set. `columns` holds
// (stack depth + 1) * block_formulas values and `scratch` the widest
// escape's inputs.
void run_block(const ProgramView& program, const ProgramRun& run,
               std::size_t begin, std::size_t count, std::span<double> columns,
               std::span<double> scratch);

// Formulas [begin, begin + count) of a run, for run_block.
struct BlockTask {
    const ProgramRun* run{nullptr};
    std::size_t begin{0U};
    std::size_t count{0U};
};

// How the blocks of a group of runs (runs that may be evaluated at the same
// time) are shared among threads. Decided from one timed apply on the
// calling thread, after one apply that warms the caches.
struct GroupPlan {
    std::uint8_t applies{0U};
    // The thread count available when it was decided, and the count chosen
    // (1: the calling thread alone).
    int decided_for{0};
    int threads{1};
};

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
                     std::span<const double* const> inputs,
                     std::span<const double> slots);
    // Adds a callable formula; one that follows another callable joins its
    // run.
    void add_callable(CallableFormula& callable, FormulaRef formula,
                      double* target);
    // Marks the run opened last as reading only parameters and tables whose
    // last-change numbers are behind `changed` (see
    // detail::ParameterAccess). replay() can then write its stored values
    // instead of evaluating it again.
    void watch_run(std::span<const std::uint64_t* const> changed);
    // For a watched run whose parameters and tables did not change since it
    // was last evaluated (`changes` is the parameters' change counter):
    // writes the values it stored then to its targets and returns true.
    // Otherwise returns false, and the run must be evaluated and recorded.
    [[nodiscard]] bool replay(const ProgramRun& run, std::uint64_t changes);
    // Stores the values a watched run just wrote.
    void record(const ProgramRun& run, std::uint64_t changes);
    // Marks the runs added so far as the independent ones: they read no
    // formula target, so they may run in any order and at the same time.
    // Without a call, every run is independent.
    void end_independent();
    // Sizes the stack and scratch space for what was added.
    void finish();
    // Makes room for `workers` threads in the view's scratch and columns.
    // Invalidates views taken before.
    void reserve_workers(std::size_t workers);

    [[nodiscard]] ProgramView view();
    [[nodiscard]] std::span<const ProgramRun> runs() const noexcept {
        return _runs;
    }
    [[nodiscard]] std::size_t independent_runs() const noexcept {
        return _independent.value_or(_runs.size());
    }
    // The plan of the group that starts at run `first_run`.
    [[nodiscard]] GroupPlan& plan(std::size_t first_run) {
        return _plans[first_run];
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
    // The scratch values of one worker, padded.
    [[nodiscard]] std::size_t scratch_stride() const noexcept;
    // Appends `op` to the run starting at `run_begin`.
    void push_op(Op op, std::uint32_t run_begin);
    // Sets `blocks` on the run opened last once all its formulas are in.
    void close_run();
    // The operation of the run for `op`, with the shared numbers folded in
    // and the stored ones renumbered (`stored_index`, -1 when shared).
    [[nodiscard]] Op fold_slot(Op op,
                               std::span<const std::optional<double>> shared,
                               std::span<const std::int64_t> stored_index);

    std::vector<Op> _ops;
    std::vector<double> _numbers;
    std::vector<const double*> _inputs;
    std::vector<double> _slots;
    std::vector<double*> _targets;
    std::vector<FormulaRef> _formulas;
    std::vector<TableView> _tables;
    std::vector<EscapeView> _escapes;
    std::vector<std::uint32_t> _escape_inputs;
    std::vector<CallableFormula*> _callables;
    std::vector<ProgramRun> _runs;
    std::optional<std::size_t> _independent;
    std::vector<GroupPlan> _plans;
    // A run that reads only parameters: what it reads, the values it wrote
    // and the change counter when it was evaluated.
    struct Watch {
        std::uint32_t first_changed{0U};
        std::uint32_t changed_count{0U};
        std::uint32_t first_value{0U};
        bool evaluated{false};
        std::uint64_t seen{0U};
    };
    std::vector<Watch> _watches;
    std::vector<const std::uint64_t*> _changed;
    std::vector<double> _values;
    // The number slots the formulas of the run opened last store.
    std::vector<std::uint32_t> _stored_slots;
    std::vector<double> _stack;
    std::vector<double> _scratch;
    std::vector<double> _columns;
    std::size_t _depth{1U};
    std::size_t _widest{0U};
};

}  // namespace detail

}  // namespace pycanha
