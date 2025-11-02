#pragma once

#include <memory>

#include "pycanha-core/solvers/tscn.hpp"

namespace pycanha {

class TSCNRL : public TSCN {
    friend class TSCNRLDS;

  public:
    explicit TSCNRL(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TSCNRL() override = default;

  protected:
    // TODO: Refactor initialize_common naming throughout the solver hierarchy
    // to avoid duplicate inherited member warnings once the API is stabilized.
    // cppcheck-suppress duplInheritedMember
    void initialize_common();
};

}  // namespace pycanha
