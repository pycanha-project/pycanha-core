# Logging and Profiling

`pycanha-core` uses `spdlog` with two logger families:

1. General logging
    `pycanha-core` is the C++ library logger.
    `pycanha-python` is a second logger intended for Python-originated messages.
    These two loggers share the same sink set and the same formatter so their
    messages appear interleaved in one output stream.

2. Profiling logging
    `pycanha-core.profiling` is a separate logger used only by
    `PYCANHA_PROFILE_SCOPE(...)`.
    It is intentionally independent from the Python-facing logger path.

## Output format

General loggers use this pattern:

```text
[%H:%M:%S.%e] [%n] [%^%l%$] %v
```

Example output:

```text
[12:15:03.124] [pycanha-core] [info] assembling matrix
[12:15:03.130] [pycanha-python] [info] starting parametric loop
[12:15:03.131] [profiling] solver step took 42ms
```

The `%n` field carries the logger name, which is the only source label needed
to distinguish C++ and Python-originated general logs.

## Logger accessors

The public logger API lives in `pycanha-core/utils/logger.hpp`.

```cpp
#include "pycanha-core/utils/logger.hpp"

auto core_logger = pycanha::get_logger();
auto python_logger = pycanha::get_python_logger();
auto profiling_logger = pycanha::get_profiling_logger();
```

Behavior summary:

- `get_logger()` lazily creates `pycanha-core`.
- `get_python_logger()` lazily creates `pycanha-python` using the sink objects
   already owned by `pycanha-core`.
- `get_profiling_logger()` lazily creates `pycanha-core.profiling` with its own
   sink set and a fixed profiling-only format.

Because `pycanha-core` and `pycanha-python` share sink instances, they must
keep the same formatting pattern. In `spdlog`, formatter state lives on the
sink, not on the logger object.

## Runtime level control

```cpp
pycanha::set_logger_level(spdlog::level::info);
pycanha::set_python_logger_level(spdlog::level::info);
```

General logger level control has an explicit guard:

- `set_logger_level()` throws `std::invalid_argument` if the requested level is
   more verbose than the compile-time `SPDLOG_ACTIVE_LEVEL`.
- `set_python_logger_level()` applies the same rule.

This is important for Python bindings. If a build compiled away `trace` or
`debug` log calls, Python must not be allowed to request those levels and act
as if the missing C++ instrumentation still exists.

The profiling logger is different:

- Profiling output does not expose runtime level control.
- Profiling messages always use the fixed output form
   `[HH:MM:SS.mmm] [profiling] message`.
- Profiling output is controlled by `PYCANHA_PROFILING`, which enables or
   removes the profiling scope machinery itself.

## Compile-time log stripping

`SPDLOG_ACTIVE_LEVEL` controls which `SPDLOG_LOGGER_TRACE`,
`SPDLOG_LOGGER_DEBUG`, and higher macro-based log calls are compiled into the
binary.

Default behavior:

- `Debug` builds map to `TRACE`
- non-`Debug` builds map to `INFO`

If you need all macro-based logs in a release build, enable
`PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE`.

### CMake example

```bash
cmake -S . -B build -DPYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE=ON
```

### Conan profile example

```ini
[options]
PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE = True
```

## Relationship between runtime and compile-time control

Both controls matter:

1. Build configuration plus `PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE` decide
    which macro-based log calls exist in the binary.
2. Runtime logger levels decide which of those compiled log calls are emitted.

For example, if `SPDLOG_ACTIVE_LEVEL=INFO`, then `TRACE` and `DEBUG` macro
calls do not exist in the binary. In that configuration,
`set_logger_level(spdlog::level::debug)` and
`set_python_logger_level(spdlog::level::debug)` throw instead of silently
pretending that more verbose C++ logs can be re-enabled.

## Profiling

The profiling helper is defined in `pycanha-core/utils/profiling.hpp`.

```cpp
void some_step() {
      PYCANHA_PROFILE_SCOPE("Some step");
      // work
}
```

Enable profiling with:

- `PYCANHA_OPTION_PROFILING = ON` in CMake
- `PYCANHA_OPTION_PROFILING = True` in a Conan profile

When profiling is disabled, `PYCANHA_PROFILE_SCOPE(...)` expands to nothing.
When profiling is enabled, it logs elapsed time in milliseconds through
`pycanha-core.profiling` using the fixed `[profiling]` label.

## Python integration contract

For the Python stack, the intended model is:

- `pycanha-core` and `pycanha-python` share one sink set and one output format
   for general logs.
- `pycanha-core.profiling` remains separate and is not exposed as a Python
   logging API.
- Python-side level changes must go through guarded setters so requests for
   compiled-away verbosity fail explicitly.