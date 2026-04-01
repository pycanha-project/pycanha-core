# Logging and profiling

`pycanha-core` uses `spdlog` for two independent output streams:

1. `pycanha-core`
   General library logs such as debug diagnostics, solver lifecycle messages, warnings and errors.

2. `pycanha-core.profiling`
   Lightweight timing output produced by `PYCANHA_PROFILE_SCOPE(...)`.

## Runtime logger control

The public logger API lives in `pycanha-core/utils/logger.hpp`.

```cpp
#include "pycanha-core/utils/logger.hpp"

pycanha::set_logger_level(spdlog::level::debug);
pycanha::set_profiling_logger_level(spdlog::level::info);
```

Available runtime levels are the normal `spdlog::level::level_enum` values.
This controls what the logger emits at runtime, but it cannot restore log calls
that were compiled out by `SPDLOG_ACTIVE_LEVEL`.

## Compile-time log stripping

`SPDLOG_ACTIVE_LEVEL` controls which `SPDLOG_LOGGER_TRACE` / `DEBUG` / `INFO`
calls are compiled into the binary.

Default behaviour:

- `Debug` builds map to `TRACE`
- non-`Debug` builds map to `INFO`

If you need all logs even in a release build, enable
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

## Profiling

The spdlog-style profiling helper is defined in `pycanha-core/utils/profiling.hpp`.

```cpp
void some_step() {
    PYCANHA_PROFILE_SCOPE("Some step");
    // work
}
```

Enable profiling with:

- `PYCANHA_OPTION_PROFILING = ON` in CMake, or
- `PYCANHA_OPTION_PROFILING = True` in a Conan profile.

When profiling is disabled, `PYCANHA_PROFILE_SCOPE(...)` expands to nothing.
When enabled, it logs elapsed time in milliseconds through
`pycanha-core.profiling`.

## Relationship between runtime and compile-time control

Both controls matter:

1. Build configuration plus `PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE` decide which log calls exist in the binary.
2. `set_logger_level()` and `set_profiling_logger_level()` decide which existing log calls are emitted.

For example, if `SPDLOG_ACTIVE_LEVEL=INFO`, then `TRACE` and `DEBUG` log calls
do not exist at runtime, even if you later call `set_logger_level(spdlog::level::trace)`.

## Notes for Python bindings

The current C++ side keeps logging sink selection simple. The intended Python
integration is to inject custom sinks from `pycanha-core-python` so main and
profiling logs can be routed independently to Python logging or Loguru.