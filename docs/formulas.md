# Formulas

A formula sets one target entity: a temperature `T<n>`, a capacity `C<n>`, a load `QS<n>`, `QA<n>`, `QE<n>`, `QI<n>`, `QR<n>`, or a coupling `GL(i,j)` / `GR(i,j)`. Each target has at most one formula. `tm.formulas()` holds them.

```cpp
tm.parameters().add_parameter("heater_power", 50.0);

tm.formulas().add_formula("QI1", 12.5);                           // constant
tm.formulas().add_formula("GL(1,2)", "contact_conductance");      // parameters
tm.formulas().add_formula("C1", "0.12 * (1 + 2e-3 * T1)");        // entities
tm.formulas().add_callable_formula("QI2", [&] { return power(); }); // callable
```

## Evaluation order

Formulas run in insertion order, so a formula may read what an earlier one set. During a solve the model applies all formulas at every callback, after updating `time` and the time variables, and before the C and Python user callbacks. A removed formula leaves the order of the others unchanged; a formula added again goes to the end.

## Formula kinds

`Formula::kind()` follows from what the expression reads:

| Kind | Reads | Notes |
|---|---|---|
| `Constant` | nothing | evaluated once, when the formula is created |
| `Parameters` | parameters only, including `time` and the time variables | the only kind with derivatives |
| `Entities` | at least one entity | |
| `Callable` | anything | a C++ callable, see below |

Expressions accept Python's `**` for powers and the usual functions (`sin`, `exp`, `log`, `sqrt`, ...). Every symbol must be a parameter or an entity of the model.

## Derivatives

Sensitivity solves (`JacobianPropagation`) use the derivatives of the `Parameters` formulas with respect to the parameters they read. They are built on the first `calculate_derivatives()` call. `Formulas::calculate_derivatives()` skips the other kinds; `calculate_derivatives()` on an `Entities` formula throws.

## Callable formulas

`add_callable_formula(target, function)` takes a `std::function<double()>`. The formula calls it on every apply, at its position in the insertion order, and writes the returned value to the target. The Python bindings use it for formulas written as Python functions.

What a callable reads cannot be known, so a callable formula counts as reading every entity:

- every formula added after it is treated as depending on its result, so no later formula is ever evaluated before it;
- it lists no parameter or entity dependencies;
- it has no derivatives (`get_derivative_values()` returns null);
- `get_value()` returns the last value written, without calling the function, and NaN before the first apply.

Add callable formulas after the expression formulas that do not need their result: an early callable makes everything after it depend on it.

## Origins and generated formulas

Every formula records who wrote it, `Formula::origin()`:

| Origin | Writer |
|---|---|
| `user` | `add_formula`, `add_callable_formula` |
| `conduction_builder` | the conduction builder |
| `etms_reader` | the `.erg` / `.etms` geometry reader |
| `esatan_tmm_reader` | the thermal model input (`.d`) reader |
| `steptas_gmm_reader` | the STEP-TAS geometry reader |

A formula added one at a time is always a `user` formula, also when it is a copy of a generated one. Generators add their formulas in batches with `add_generated_formulas(origin, formulas)`, which returns a `GenerationReport`. On a target that already has a formula:

- a `user` formula stays, and the generated one is dropped (`kept_user`);
- a generated formula, of any origin, is removed and the new one goes to the end of the list (`replaced`);
- an expression that does not parse is skipped (`failed`, with the message), and a formula already on the target stays.

One WARN per call sums up the collisions and failures, with the first few targets; DEBUG lists each one. A generator that writes its formulas again first removes all of its own with `remove_formulas(origin)`, so formulas it no longer produces are gone. `add_formula` still throws on a target that has a formula: remove it first.

`contains(target)` and `find(target)` look a formula up by target in constant time.

## Compilation and model changes

`compile_formulas()` takes the value pointer of every target and input. A solve compiles the formulas only when something they point into changed since the last compile: a formula added or removed, a parameter added or removed, a node added or removed, or a coupling added. Values can change freely. On an unchanged model a solve does no formula work beyond comparing these counters, and `compile_formulas()` can be called ahead of a solve to pay the cost earlier.

Removing a node is never refused, even when formulas use it. The next validation (`validate_for_execution()`, called by `compile_formulas()` and by a solve) checks every formula's target and inputs and throws `std::runtime_error` with the number of formulas that cannot run and the first few, so a solve never starts on a broken formula. Remove or fix them, then solve again.
