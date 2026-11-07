# Formulas

A formula sets one target entity: a temperature `T<n>`, a capacity `C<n>`, a load `QS<n>`, `QA<n>`, `QE<n>`, `QI<n>`, `QR<n>`, or a coupling `GL(i,j)` / `GR(i,j)`. Each target has at most one formula. `tm.formulas()` holds them.

```cpp
tm.parameters().add_parameter("heater_power", 50.0);
tm.parameters().add_table("cp_al", LookupTable1D(T_K, cp));       // [K] -> [J/(kg K)]

tm.formulas().add_formula("QI1", 12.5);                           // constant
tm.formulas().add_formula("GL(1,2)", "contact_conductance");      // parameters
tm.formulas().add_formula("C1", "0.12 * (1 + 2e-3 * T1)");        // entities
tm.formulas().add_formula("C2", "0.5 * cp_al(T2)");               // table call
tm.formulas().add_callable_formula("QI2", [&] { return power(); }); // callable
```

Formulas are stored by target. `find(target)` and `contains(target)` look one up in constant time, and `formulas()` goes through all of them in insertion order. `find()` and `formulas()` build a `Formula` object for each formula they return; the collection keeps no such objects, so two calls give two objects with the same content.

## Evaluation order

Formulas run in insertion order, so a formula may read what an earlier one set. During a solve the model applies all formulas at every callback, after updating `time` and the tables evaluated at a parameter, and before the C and Python user callbacks. A removed formula leaves the order of the others unchanged; a formula added again goes to the end.

The compiled program runs some formulas earlier than their position, with exactly the results of insertion order. A formula is *independent* when it reads no formula target and no earlier formula reads its target; independent formulas run first, in any order, then the others in insertion order. A callable formula counts as reading everything, so no formula after one is independent.

`move_formula(target, before)` moves a formula to just before another one, and `move_formula_to_end(target)` to the end. Both return `false` when a target has no formula. The new position counts as the formula's insertion position. A move goes through the whole list once; it is meant for occasional changes, not for building a model.

## Formula kinds

`Formula::kind()` follows from what the expression reads:

| Kind | Reads | Notes |
|---|---|---|
| `Constant` | nothing | evaluated once, when the formula is created |
| `Parameters` | parameters only, including `time`, the values of tables evaluated at a parameter, and table calls | the only kind with derivatives |
| `Entities` | at least one entity | |
| `Callable` | anything | a C++ callable, see below |

## Expressions

Expressions accept numbers, parameters, entities, table calls and these operations:

- `+`, `-`, `*`, `/` and powers, written `**` or `^`;
- `exp`, `log`, `sqrt`, `abs`, `sign`, `floor`, `ceiling`, `max`, `min` (any number of arguments), `atan2`;
- `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, their inverses (`asin`, ..., `acsc`), the hyperbolic functions and their inverses;
- the constants `pi` and `E`.

Every other symbol must be a parameter or an entity of the model, and every other call `name(argument)` must call a table (see below). An unknown function, a table call without exactly one argument, or a table named without a call is refused when the formula is added.

Expressions are lowered to a short sequence of evaluator operations. Integer powers up to 64 are computed by multiplication and `x**1` costs nothing. Functions without an operation of their own (`gamma`, `erf`, `loggamma`, ...) are evaluated by SymEngine's double evaluator; a table call inside such a function is refused.

## Shapes

Before a formula is parsed, its numbers and node numbers are taken out of the text. What is left is its *shape*:

```
GL(12,13) = 0.0125 * k_al((T12 + T13) / 2)
  shape:  GL({0},{1}) = {#0} * k_al((T{0} + T{1}) / {#1})
  values: nodes 12, 13; numbers 0.0125, 2
```

Formulas of the same shape share one parse and one list of evaluator operations, and each formula stores only its own node numbers and numbers. A model read from a file with a million couplings written in two ways has two shapes, whatever order the formulas come in. Only the first formula of a shape pays for the parse.

- Names are read as the parser reads them: parameters, table names and functions stay in the shape; `T12`, `QI3`, `GL(12,13)` become node slots. A node number used twice (the target's nodes and the temperatures it reads) is one slot.
- A number written as an exponent stays in the shape, so `T1**2` keeps the integer power operation, and `T1**2` and `T1**3` are two shapes.
- Two formulas share a shape only when their text matches apart from numbers, node numbers and blanks: `g*k(T1)` and `k(T1)*g` are two shapes.
- Expressions are printed from the shape and the formula's numbers when asked for (`ExpressionFormula::expression()`). The printed text keeps the blanks of the first formula of the shape and writes numbers in their shortest exact form (`2.5e-3` prints as `0.0025`, a negative number in parentheses). It is meant for reading, not for parsing back.

## Templates

A generator that knows its shape passes it once, with named slots, and one array of values per slot:

```cpp
const std::vector<NodeNum> a{1, 2, 3}, b{2, 3, 4};
const std::vector<double> g{0.0125, 0.0130, 0.0125};
tm.formulas().add_generated_formulas(
    Origin::conduction_builder,
    FormulaTemplate{.target = "GL({a},{b})", .expression = "{g} * k_al((T{a} + T{b}) / 2)"},
    std::vector<NodeSlot>{{.name = "a", .values = a}, {.name = "b", .values = b}},
    std::vector<NumberSlot>{{.name = "g", .values = g}});
```

- `{a}` inside an entity (`T{a}`, `QI{a}`, `GL({a},{b})`) takes node numbers, `{g}` on its own takes numbers.
- The target must be one entity. The table and every other name are fixed by the template; numbers written in the template stay as written.
- A malformed call throws `std::invalid_argument` and adds nothing: a slot without values, values for a slot the template does not use or uses as the other kind, slots of different lengths, a target that is not one entity, an expression that does not parse.
- The collision rules of generated formulas apply to each formula (see below). A formula reading an entity that does not exist is reported in `failed`.
- `add_formulas(shape, nodes, numbers)` is the same call for user formulas. It checks every formula first and adds none when a target already has a formula, appears twice in the call or reads an entity that does not exist.

Each formula added by a template is an ordinary formula: it is found, printed, removed and moved one at a time.

## Tables

A table is a named 1D function of one argument, stored with the parameters:

```cpp
tm.parameters().add_table("k_al", LookupTable1D(T_K, k, InterpolationMethod::Linear,
                                                ExtrapolationMethod::Constant));
tm.formulas().add_formula("GL(1,2)", "2.5e-3 * k_al((T1 + T2) / 2)");
```

- The argument is any expression of entities and parameters.
- Interpolation: `Linear`, `NearestLower`, `NearestUpper` or `Step`. Extrapolation: `Constant` holds the end value, `Linear` extends the end segment, `Throw` throws `std::out_of_range`.
- Table names share the parameters' name space, ignoring case: a table cannot take the name of a parameter, an entity token (`T`, `C`, `QS`, `QA`, `QE`, `QI`, `QR`, `GL`, `GR`), `time`, or a built-in function or constant.
- Adding, removing or replacing a table is a change of the parameters' structure, refused while a solve runs.
- `set_table_values(name, y)` and `set_table_x(name, x)` overwrite the values in place: the size stays and `x` must stay strictly increasing. They change no structure, so a callback may edit a table during a solve and the next lookup reads the new values.
- Removing a table that formulas call is allowed. The next validation names those formulas, as for a removed node.

**Tables evaluated at a parameter.** `add_table(name, data, "time")` looks the table up at the parameter `time` once per callback and publishes the value as a parameter with the table's name. Formulas then read `name` as a parameter: one lookup per callback however many formulas read it, and those formulas keep their derivatives. The same table can still be called with another argument, e.g. `name(time - 60)`. Any parameter that stores a double can serve as the argument.

**Generated tables.** A generator writes its tables with `add_generated_table(origin, name, data)`. A `user` table with that name stays and the call logs a WARN (`kept_user`). A generated table of any origin has its data and origin replaced (`overwritten`), with a WARN when the origin differs. Tables are never removed by a regeneration.

### Out of range

A query outside a table's range is recorded per table: the time of the first one (from the `time` parameter), its argument, and the smallest and largest argument seen. After each solve, one WARN lists every table queried outside its range and the same line is added to the solver report's `warnings`. `Parameters::table_ranges()` returns the records until the next solve starts.

- `Linear` extrapolation never reports: extrapolating was asked for.
- `Throw` throws at the first such query.
- A NaN argument is reported and gives NaN.

Recording costs nothing in range: only the out-of-range branch of a lookup writes the record.

## Derivatives

Sensitivity solves (`JacobianPropagation`) use the derivatives of the `Parameters` formulas with respect to the parameters they read. They are built once per shape, on the first `calculate_derivatives()` call of one of its formulas, by parsing and differentiating the shape. The values are kept on the `Formula` object the call was made on. `Formulas::calculate_derivatives()` builds the derivatives of every `Parameters` shape, so a shape without derivatives is reported at once; `calculate_derivatives()` on an `Entities` formula throws.

A table call is differentiated through the table's slope. For `Linear` interpolation that is the slope of the segment that holds the argument, and of the segment above it at a breakpoint, so a derivative jumps where the argument crosses a breakpoint. The stepped interpolations and `Constant` extrapolation have a slope of zero.

## Callable formulas

`add_callable_formula(target, function)` takes a `std::function<double()>`. The formula calls it on every apply, at its position in the insertion order, and writes the returned value to the target. The Python bindings use it for formulas written as Python functions.

What a callable reads cannot be known, so a callable formula counts as reading every entity:

- every formula added after it is treated as depending on its result, so no later formula is ever evaluated before it;
- it lists no parameter or entity dependencies;
- it has no derivatives (`get_derivative_values()` returns null);
- `get_value()` returns the last value written, without calling the function, and NaN before the first apply.

Add callable formulas after the expression formulas that do not need their result: an early callable makes everything after it depend on it.

## Origins and generated formulas

Every formula and table records who wrote it, `origin()`:

| Origin | Writer |
|---|---|
| `user` | `add_formula`, `add_callable_formula`, `add_table` |
| `conduction_builder` | the conduction builder |
| `etms_reader` | the `.erg` / `.etms` geometry reader |
| `esatan_tmm_reader` | the thermal model input (`.d`) reader |
| `steptas_gmm_reader` | the STEP-TAS geometry reader |

A formula added one at a time is always a `user` formula, also when it is a copy of a generated one. Generators add their formulas with `add_generated_formulas(origin, formulas)` (a list of targets and expressions) or with a template (see above); both return a `GenerationReport`. On a target that already has a formula:

- a `user` formula stays, and the generated one is dropped (`kept_user`);
- a generated formula, of any origin, is removed and the new one goes to the end of the list (`replaced`);
- an expression that is refused is skipped (`failed`, with the message), and a formula already on the target stays.

One WARN per call sums up the collisions and failures, with the first few targets; DEBUG lists each one. A generator that writes its formulas again first removes all of its own with `remove_formulas(origin)`, so formulas it no longer produces are gone. `add_formula` still throws on a target that has a formula: remove it first.

A removed formula of a template leaves a hole that costs one byte until more than half of the formulas stored with it are removed; the next compile then packs them. Adding a formula back on that target makes a new formula at the end of the list.

`dependencies()` groups the targets of the formulas by the parameters, entities and tables they read.

## Compilation and model changes

`compile_formulas()` builds one program for all formulas. Formulas of the same shape that run one after another share one run of operations, with their own value pointers and numbers; independent formulas are grouped by shape, so a template compiles to one run. `apply_formulas()` runs it, and compiles first when something the program points into changed since the last compile: a formula added or removed, a parameter or table added, removed or replaced, a node added or removed, or a coupling added. Values can change freely, table values edited in place included. On an unchanged model a solve does no formula work beyond comparing these counters, and `compile_formulas()` can be called ahead of a solve to pay the cost earlier.

Removing a node or a table is never refused, even when formulas use it. The next validation (`validate_for_execution()`, called by `compile_formulas()` and by a solve) checks every formula's target, inputs and tables, and throws `std::runtime_error` with the number of formulas that cannot run and the first few, so a solve never starts on a broken formula. Remove or fix them, then solve again.

`debug_formulas = true` applies the formulas one at a time and logs each target and value at INFO. It is slower and meant for tracing.
