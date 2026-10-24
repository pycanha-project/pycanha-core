#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"

namespace pycanha {
class DataModel;
}  // namespace pycanha

namespace pycanha::solvers::detail {
class Factorizer;
}  // namespace pycanha::solvers::detail

namespace pycanha::solvers::transient {

/// Crank-Nicolson transient with the radiation linearised at every inner
/// iteration, solved with a sparse direct factorization.
/**
 * The time and output settings (start_time, end_time, time_step,
 * output_interval, output_attributes, output_name) are read by initialize(),
 * which also creates the output model in the thermal data; time_step is
 * required. The factorization is always LU. A step whose inner iterations do
 * not converge logs an ERROR and the run goes on.
 */
class CrankNicolson : public Solver {
  public:
    explicit CrankNicolson(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~CrankNicolson() override;

    CrankNicolson(const CrankNicolson&) = delete;
    CrankNicolson& operator=(const CrankNicolson&) = delete;
    CrankNicolson(CrankNicolson&&) = delete;
    CrankNicolson& operator=(CrankNicolson&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "CrankNicolson";
    }

    /// Sets start_time, end_time, time_step and output_interval [s] at once.
    /// Throws std::invalid_argument for an end before the start, a time step
    /// that is not positive or a negative output interval.
    void set_simulation_time(double start_time, double end_time,
                             double time_step, double output_interval);

    /// The output model of the last initialize().
    [[nodiscard]] DataModel& output_model();
    [[nodiscard]] const DataModel& output_model() const;

  protected:
    CrankNicolson(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
                  SolverType type);

    // Time steps and outputs from the resolved settings; creates the output
    // model.
    void initialize_transient();
    void save_output_data();
    void outputs();
    void outputs_first_last();
    // Throws std::invalid_argument when the solver is not initialised;
    // otherwise rewinds the time to start_time.
    void restart_solve();

    void build_capacities();
    void build_conductance_matrix();
    void build_heat_flux();
    void store_heat_flux_at_n0();
    void add_capacities_to_matrix();
    void solve_step();
    void release() noexcept;

    // Two times closer than this are the same output time [s].
    static constexpr double eps_time = 1.0e-6;

    double time = 0.0;
    int time_iter = 0;
    int num_time_steps = 0;
    int num_outputs = 0;
    int wait_n_dtimes = 0;
    int idata_out = 0;

    SpMatRow _k_matrix;
    VectorXd _rhs;
    SpMatRow _boundary_matrix;

    VectorXd _heat_flux_n0;
    VectorXd _capacities;
    VectorXd _capacities_inverse;

    std::unique_ptr<detail::Factorizer> _factorizer;

    VectorXd _t3_domain;
    VectorXd _t3_boundary;
    VectorXd _t4_domain;
    VectorXd _t4_boundary;

    std::vector<int> _lower_kr_indices;
    std::vector<int> _upper_kr_indices;
    std::vector<int> _lower_kl_indices;
    std::vector<int> _upper_kl_indices;
    std::vector<int> _diagonal_indices;

    VectorXd _radiation_linear_term;
    VectorXd _kt_q_n0;
    VectorXd _ones_domain;
    VectorXd _ones_boundary;
};

}  // namespace pycanha::solvers::transient
