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
    void initialize_common();
};

}  // namespace pycanha
