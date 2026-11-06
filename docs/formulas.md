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

## Evaluation order

Formulas run in insertion order, so a formula may read what an earlier one set. During a solve the model applies all formulas at every callback, after updating `time` and the tables evaluated at a parameter, and before the C and Python user callbacks. A removed formula leaves the order of the others unchanged; a formula added again goes to the end.

The compiled program runs some formulas earlier than their position, with exactly the results of insertion order. A formula is *independent* when it reads no formula target and no earlier formula reads its target; independent formulas run first, then the others in insertion order. A callable formula counts as reading everything, so no formula after one is independent.

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

When a formula is added, its expression is parsed once and lowered to a short sequence of evaluator operations; the expression text is kept for printing and for derivatives. Integer powers up to 64 are computed by multiplication and `x**1` costs nothing. Functions without an operation of their own (`gamma`, `erf`, `loggamma`, ...) are evaluated by SymEngine's double evaluator; a table call inside such a function is refused.

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

Sensitivity solves (`JacobianPropagation`) use the derivatives of the `Parameters` formulas with respect to the parameters they read. They are built on the first `calculate_derivatives()` call, by parsing and differentiating the expression text. `Formulas::calculate_derivatives()` skips the other kinds; `calculate_derivatives()` on an `Entities` formula throws.

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

A formula added one at a time is always a `user` formula, also when it is a copy of a generated one. Generators add their formulas in batches with `add_generated_formulas(origin, formulas)`, which returns a `GenerationReport`. On a target that already has a formula:

- a `user` formula stays, and the generated one is dropped (`kept_user`);
- a generated formula, of any origin, is removed and the new one goes to the end of the list (`replaced`);
- an expression that is refused is skipped (`failed`, with the message), and a formula already on the target stays.

One WARN per call sums up the collisions and failures, with the first few targets; DEBUG lists each one. A generator that writes its formulas again first removes all of its own with `remove_formulas(origin)`, so formulas it no longer produces are gone. `add_formula` still throws on a target that has a formula: remove it first.

`contains(target)` and `find(target)` look a formula up by target in constant time.

## Compilation and model changes

`compile_formulas()` builds one program for all formulas: their operations in one array, with the value pointers of every target and input and a view of every table called. `apply_formulas()` runs it, and compiles first when something the program points into changed since the last compile: a formula added or removed, a parameter or table added, removed or replaced, a node added or removed, or a coupling added. Values can change freely, table values edited in place included. On an unchanged model a solve does no formula work beyond comparing these counters, and `compile_formulas()` can be called ahead of a solve to pay the cost earlier.

Removing a node or a table is never refused, even when formulas use it. The next validation (`validate_for_execution()`, called by `compile_formulas()` and by a solve) checks every formula's target, inputs and tables, and throws `std::runtime_error` with the number of formulas that cannot run and the first few, so a solve never starts on a broken formula. Remove or fix them, then solve again.

`debug_formulas = true` applies the formulas one at a time and logs each target and value at INFO. It is slower and meant for tracing.
