# Temperature-dependent IR emissivity

`radiative::RadiativeRetracer` keeps the radiative couplings (`GR`) of a thermal model in step with IR emissivities that depend on temperature. During a solve it traces the geometry again on the GPU when the temperatures have moved, writes the new couplings into the model and tells the solver when the couplings it used are recent enough to accept convergence. The couplings at convergence are those of a trace at the final temperatures.

Use it when the emissivity of a coating changes enough over the run to change the couplings between faces: variable-emittance coatings, or surfaces whose emissivity drifts strongly with temperature. When a coupling only depends on the emissivity of one node and the geometry does not matter, for example a node that sees only space, a formula is enough and needs no GPU:

```cpp
tm.formulas().add_formula("GR(1,1000)", "0.5 * eps_paint(T1)");   // 0.5 m² facing space
```

## Emissivity tables

The IR emissivity of an `gmm::OpticalMaterial` can name a table of the model's parameters, giving it against the temperature [K] of the face's node:

```cpp
tmm->parameters().add_table(
    "eps_paint", pycanha::LookupTable1D(Eigen::Vector2d(250.0, 350.0),
                                        Eigen::Vector2d(0.15, 0.75)));

auto paint = std::make_shared<gmm::OpticalMaterial>("paint", 0.5, 0.3);
paint->set_emissivity_ir_table("eps_paint");   // "" removes it
```

- Only the IR emissivity can be a table. Solar properties and the IR specular reflectance and transmissivity stay numbers.
- The material keeps its number (`emissivity_ir()`), and everything except the retracer reads it: a plain trace, the material table of the geometry model, the solar and view-factor paths. `GeometryModel::material_table()` lists the table names in `MaterialTable::emissivity_ir_tables`, one per row (empty without a table).
- The table must extrapolate with `Constant` (the default) or `Throw`. `Linear` extrapolation is refused, because it can leave [0, 1].
- At every breakpoint the emissivity must be in [0, 1] and, with the IR specular reflectance and transmissivity of the material, add up to at most 1.
- The table is looked up and checked by `RadiativeRetracer::initialize()`, not when it is named. A lookup outside the table during a solve is reported with the other out-of-range table calls of that solve (a WARN and `SolverReport::warnings`).

## Construction

```cpp
RadiativeRetracer(Device& device, const gmm::GeometryModel& gmm,
                  std::shared_ptr<ThermalMathematicalModel> tmm,
                  NodeNum space_node,
                  TraceSettings trace = {}, AccumConfig accum = {},
                  std::set<PatternState> pattern_states = {LOW, MID, HIGH},
                  RetraceThreshold retrace_dT = 0.0, bool write_back = true);
```

| Argument | Meaning |
|---|---|
| `device` | The GPU device that traces. It must outlive the retracer. |
| `gmm` | Read once, at construction, into the retracer's own scene. Later changes to the geometry are not seen: build a new retracer. |
| `tmm` | The thermal model whose radiative couplings the retracer owns. |
| `space_node` | The node that receives the couplings to space. It must be in the model when `initialize()` runs. |
| `trace` | Rays per face, seed and the other trace settings. The seed is fixed, so two runs of the same model give the same couplings. |
| `accum` | Accumulator layout (`AccumLayout::Dense` by default, or `Tiled` with its `tile_rows`) and sparse threshold, as for any exchange trace. |
| `pattern_states` | Optical states traced by `initialize()` to find which node pairs need a coupling (see below). |
| `retrace_dT` | Temperature change [K] that triggers a new trace: one value for every node owning an active face, or a `std::map<NodeNum, double>` checked on the listed nodes only. `0` traces again on any change. Must not be negative. |
| `write_back` | Whether the couplings that had no place in the model are inserted into it after the solve. |

A face takes the optical properties of its material, with the IR emissivity read from the table at the temperature of the face's node. An active face without a node is taken at 273.15 K and its couplings are dropped (one WARN). A face without a material is a black body. Only the IR band is traced: solar loads are not touched.

## What it owns

`initialize()` writes **every** radiative coupling of the model: the value of the trace at the current temperatures, or 0 where the trace has none. A radiative coupling that is not produced by the geometry, for example to a node without faces, is therefore set to 0. A coupling that must exist besides the traced ones has to be a formula or a callback: at each callback the solver calls the retracer first, then the formulas, then the user callbacks, so a formula on a coupling the retracer also writes wins.

```cpp
tmm->formulas().add_formula("GR(1,50)", 0.1);   // kept at 0.1 during the solve
```

Conductive couplings, loads and capacities are not touched.

## Life cycle

```cpp
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/retracer.hpp"
#include "pycanha-core/solvers/steady_state.hpp"

namespace rad = pycanha::radiative;

rad::Device device = rad::Device::create();

rad::TraceSettings trace;
trace.rays_per_face = 2'000;
trace.seed = 1;

auto retracer = std::make_shared<rad::RadiativeRetracer>(
    device, *gmm, tmm, /*space_node=*/1000, trace);
retracer->initialize();           // before the solver's initialize()

pycanha::solvers::steady::SteadyState solver(tmm);
solver.set_retracer(retracer);
solver.initialize();
solver.solve();

const auto& report = solver.report();   // report.retraces, report.overflow_couplings
```

The same retracer works with `transient::CrankNicolson`. `JacobianPropagation` refuses a retracer (its sensitivities do not include couplings that change with temperature), and so does a solver with `fixed_couplings = true`.

- `initialize()` checks the model and the tables, traces at the pattern states and at the current temperatures, inserts the node pairs it found and writes every coupling. It must run before the solver's `initialize()`, since it can add couplings to the model.
- A solve throws when the model's nodes, couplings or tables changed since `initialize()`: call `initialize()` again, then the solver's `initialize()`.
- A solve that wrote couplings back (see below) leaves both the retracer and the solver uninitialised: initialise both again before the next solve.
- `statistics()` returns a `RetracerStatistics`: traces made, couplings of the pattern, faces without a node, the overflow of the last trace, the totals to space, to inactive faces and lost, and the time spent retracing.

## Pattern and overflow

The solver's matrices have a fixed structure: a coupling can change value during a solve, but no coupling can be added. `initialize()` therefore decides up front which node pairs get a coupling, from the union of a few traces:

| `PatternState` | Each tabled material takes |
|---|---|
| `LOW` | the lowest emissivity of its table |
| `MID` | the mean of its lowest and highest emissivity |
| `HIGH` | the highest emissivity of its table |

plus a trace at the current temperatures. When no material has a table, only the trace at the current temperatures is made. For coatings whose emissivity stays above 0, the pairs do not depend on the emissivity and an empty set (`std::set<PatternState>{}`) gives the same pattern with fewer traces. The states matter when an emissivity reaches 0, which removes pairs from a trace.

A traced coupling without a place in the model (the *overflow*) is not lost: it is carried as an internal load, added to the node loads in every load sum, which makes the result exact at convergence. The transient output `KR` includes it. The report gives `overflow_couplings` and `overflow_share`, the overflow's share of the sum of all couplings. With `write_back = true`, the overflow of the last trace is inserted into the model after the solve, so the next solve has those pairs in its pattern.

## Retrace trigger and convergence

Before each callback the retracer traces again when a checked node moved more than its `retrace_dT` since the last trace. After each linear solve whose own test passed, it accepts convergence only when every checked node is within `max(retrace_dT, tolerance)` of the temperatures of the trace that solve used. With `retrace_dT = 0` the couplings at convergence belong to a trace at the final temperatures.

Tracing has a cost comparable to a solver iteration: with 1000 rays per face, a retrace costs about one Crank–Nicolson iteration above a few hundred nodes, and more than that on smaller models. With `retrace_dT = 0` a transient step needs at least one trace per iteration, and tracing can take most of the run. A positive threshold traces less, at the price of couplings up to the threshold old. Measured on a 32 × 32 plate (1024 nodes) with an emissivity going from 0.15 to 0.75 over 250-350 K, 300 steps of 10 s:

| `retrace_dT` | retraces | solve time | max difference from `retrace_dT = 0` |
|---|---|---|---|
| 0 | 600 | 37.9 s | — |
| 0.1 K | 88 | 17.5 s | 0.020 K |
| 0.5 K | 22 | 15.0 s | 0.072 K |
| 2 K | 6 | 14.8 s | 0.007 K |
| 5 K | 2 | 14.6 s | 1.36 K |

The error is not monotone in the threshold: it depends on when the retraces fall. Start from 0 for a reference and raise the threshold while the result stays within what the analysis needs. A per-node map limits the checks to the nodes whose emissivity matters.

The traced couplings carry the noise of single-precision sums, about 1e-6 relative. A steep emissivity amplifies it, so a solver `tolerance` far below 1e-4 K may never be met.

## Steady-state convergence

With a temperature-dependent emissivity, each iteration uses couplings from the previous iteration's temperatures. On a node that loses its heat mostly by radiation, the temperature error is multiplied at each iteration by about

    λ = -(T / 4) · (dε/dT) / ε · (1 - T_sink⁴ / T⁴)

With |λ| ≥ 1 the plain iteration oscillates and never converges: for the coating above, λ ≈ -1.1 to -1.3 near 280-290 K. Raising `retrace_dT` does not fix this: it freezes the couplings once the swing fits inside the threshold, and the answer is then off by about the threshold.

The solver settings `relaxation` and `relaxation_mode` damp the update to `T ← T + ω·ΔT`, which multiplies the error by `1 + ω(λ - 1)`:

- `RelaxationMode::ADAPTIVE` (default) starts at ω = `relaxation` and estimates ω from the last two changes (Aitken's method). A loop that converges without damping keeps ω = 1 and gives the same iterations and temperatures as before.
- `RelaxationMode::FIXED` uses `relaxation` at every iteration. A fixed ω < 1 also slows down models that did not need it.
- The iteration that meets the tolerance always takes the full step, and the convergence test always uses the full change.
- Both apply to the steady state and to the Crank–Nicolson inner iterations. In a transient the capacities damp the lag, but long time steps, thin parts and nodes without capacity can still oscillate.

On the 32 × 32 plate the plain steady state did not converge in 200 iterations (1.36 K off); with `ADAPTIVE` it converged in 7 iterations. The default `max_iterations` is 10: a model with steep emissivities may need more. The report gives `min_relaxation` and `last_relaxation` when ω went below 1, which shows how hard the model was to converge.

The same applies without a GPU to any coupling written as a formula of a temperature, such as `GR(1,1000) = eps_paint(T1)`.

## Report

`SolverReport` fields filled during a solve with a retracer:

| Field | Meaning |
|---|---|
| `retraces` | traces made during the solve |
| `overflow_couplings` | couplings of the last trace without a place in the model |
| `overflow_share` | their share of the sum of all couplings of the last trace |
| `min_relaxation`, `last_relaxation` | smallest and last relaxation factor (1 when no damping was needed) |

All appear in `to_string()` (the relaxation line only when ω went below 1) and `to_dict()`.

## Restrictions

- IR only: solar optical properties and solar loads are not recomputed.
- Editing a table's values during a solve does not trigger a retrace.
- The geometry is read at construction.
- With `AccumLayout::Tiled`, needed above roughly 16 000 faces where the dense layout passes 2 GB, each retrace is much slower than the first trace.
- A GPU with ray-tracing support is required (`radiative::is_available()`).
