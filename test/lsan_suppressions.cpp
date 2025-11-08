// Copyright (c) 2025, pycanha-project
// SPDX-License-Identifier: Apache-2.0

// LeakSanitizer support ------------------------------------------------
//
// Intel MKL allocates internal structures inside `mkl_pds_lp64_sfinit_pardiso`
// that are not released before program termination. This shows up as a leak
// when the test suite is executed with AddressSanitizer (which includes
// LeakSanitizer). The leak is benign and originates inside MKL, so we suppress
// it explicitly to keep the test run clean while still surfacing any other
// leaks that might appear in our own code.
//
// LeakSanitizer reads suppressions from the string returned by
// `__lsan_default_suppressions`. We guard the definition so that it is only
// provided when the binary is compiled with AddressSanitizer enabled; otherwise
// the function would be unused.

namespace {
[[maybe_unused]] constexpr const char* k_mkl_leak_suppression =
    "leak:mkl_pds_lp64_sfinit_pardiso\n";

constexpr bool k_with_asan() {
#if defined(__SANITIZE_ADDRESS__)
    return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    return true;
#else
    return false;
#endif
#else
    return false;
#endif
}
}  // namespace

// NOLINTNEXTLINE(readability-identifier-naming,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
extern "C" const char* __lsan_default_suppressions() noexcept {
    if (!k_with_asan()) {
        return nullptr;
    }
    return k_mkl_leak_suppression;
}
