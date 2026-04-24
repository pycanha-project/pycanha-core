#include "pycanha-core/solvers/solver_registry.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/solvers/tscnrlds_jacobian.hpp"

namespace pycanha {

SolverRegistry::SolverRegistry(std::shared_ptr<ThermalMathematicalModel> tmm)
    : _tmm(std::move(tmm)) {
    if (_tmm == nullptr) {
        throw std::invalid_argument(
            "SolverRegistry requires a ThermalMathematicalModel");
    }
}

SolverRegistry::~SolverRegistry() = default;

SSLU& SolverRegistry::sslu() {
    if (_sslu == nullptr) {
        _sslu = std::make_unique<SSLU>(_tmm);
    }

    return *_sslu;
}

TSCNRLDS& SolverRegistry::tscnrlds() {
    if (_tscnrlds == nullptr) {
        _tscnrlds = std::make_unique<TSCNRLDS>(_tmm);
    }

    return *_tscnrlds;
}

TSCNRLDS_JACOBIAN& SolverRegistry::tscnrlds_jacobian() {
    if (_tscnrlds_jacobian == nullptr) {
        _tscnrlds_jacobian = std::make_unique<TSCNRLDS_JACOBIAN>(_tmm);
    }

    return *_tscnrlds_jacobian;
}

}  // namespace pycanha
