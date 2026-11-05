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
