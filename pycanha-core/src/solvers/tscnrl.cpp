#include "pycanha-core/solvers/tscnrl.hpp"

#include <memory>
#include <utility>

#include "pycanha-core/solvers/tscn.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

TSCNRL::TSCNRL(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : TSCN(std::move(tmm_shptr)) {}

void TSCNRL::initialize_common() { TSCN::initialize_common(); }

}  // namespace pycanha
