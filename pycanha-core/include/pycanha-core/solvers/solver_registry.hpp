#pragma once

#include <memory>

namespace pycanha {

class SSLU;
class TSCNRLDS;
class TSCNRLDS_JACOBIAN;
class ThermalMathematicalModel;

class SolverRegistry {
  public:
    explicit SolverRegistry(std::shared_ptr<ThermalMathematicalModel> tmm);
    ~SolverRegistry();

    SolverRegistry(const SolverRegistry&) = delete;
    SolverRegistry& operator=(const SolverRegistry&) = delete;
    SolverRegistry(SolverRegistry&&) noexcept = delete;
    SolverRegistry& operator=(SolverRegistry&&) noexcept = delete;

    [[nodiscard]] SSLU& sslu();
    [[nodiscard]] TSCNRLDS& tscnrlds();
    [[nodiscard]] TSCNRLDS_JACOBIAN& tscnrlds_jacobian();

    [[nodiscard]] std::shared_ptr<ThermalMathematicalModel> tmm_ptr()
        const noexcept {
        return _tmm;
    }

  private:
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    std::unique_ptr<SSLU> _sslu;
    std::unique_ptr<TSCNRLDS> _tscnrlds;
    std::unique_ptr<TSCNRLDS_JACOBIAN> _tscnrlds_jacobian;
};

}  // namespace pycanha