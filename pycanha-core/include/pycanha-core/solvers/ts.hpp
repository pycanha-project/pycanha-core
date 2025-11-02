#pragma once

#include <memory>
#include <string>

#include "pycanha-core/solvers/solver.hpp"

namespace pycanha {

class TransientSolver : public Solver {
    friend class TSCN;

  public:
    explicit TransientSolver(
        std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~TransientSolver() override = default;

    void set_simulation_time(double start_time, double end_time, double dtime,
                             double output_stride);

  protected:
    // TODO: Refactor initialize_common naming throughout the solver hierarchy
    // to avoid duplicate inherited member warnings once the API is stabilized.
    // cppcheck-suppress duplInheritedMember
    void initialize_common();

    void save_temp_data();
    void outputs();
    void outputs_first_last();

    void restart_solve() override;

    double start_time = 0.0;
    double end_time = 0.0;
    double dtime = -1.0;
    double dtime_out = 0.0;

    double time = 0.0;
    int time_iter = 0;

    int num_time_steps = 0;
    int num_outputs = 0;
    int wait_n_dtimes = 0;
    int idata_out = 0;

    double* output_data = nullptr;
    std::string output_table_name;
};

}  // namespace pycanha
