#include "pycanha-core/solvers/ts.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

TransientSolver::TransientSolver(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr)) {}

void TransientSolver::set_simulation_time(double start_time, double end_time,
                                          double dtime, double output_stride) {
    if (end_time < start_time) {
        throw std::invalid_argument("end_time must be greater than start_time");
    }
    if (dtime <= 0.0) {
        throw std::invalid_argument("dtime must be positive");
    }
    if (output_stride < 0.0) {
        throw std::invalid_argument("output stride must be non-negative");
    }

    this->start_time = start_time;
    this->end_time = end_time;
    this->dtime = dtime;
    dtime_out = output_stride;
}

void TransientSolver::initialize_common() {
    Solver::initialize_common();

    if (dtime <= 0.0) {
        throw std::invalid_argument(
            "Simulation time has not been set. Please set the simulation time"
            " before initializing the solver.");
    }

    const double total_sim_time = end_time - start_time;
    num_time_steps =
        static_cast<int>(std::ceil((total_sim_time - eps_time) / dtime));
    num_time_steps = std::max(num_time_steps, 1);

    if (dtime_out <= 0.0) {
        wait_n_dtimes = num_time_steps;
    } else {
        wait_n_dtimes =
            static_cast<int>(std::ceil((dtime_out - eps_time) / dtime));
        wait_n_dtimes = std::max(wait_n_dtimes, 1);
    }

    num_outputs = ((num_time_steps - 1) / wait_n_dtimes) + 2;
    num_outputs = std::max(num_outputs, 2);

    tmm.thermal_data.create_new_table(output_table_name, num_outputs, N + 1);
    output_data = tmm.thermal_data.get_table(output_table_name).data();
}

void TransientSolver::save_temp_data() {
    if (output_data == nullptr) {
        return;
    }

    const auto row_offset =
        static_cast<std::size_t>(N + 1) * static_cast<std::size_t>(idata_out);
    double* data_ptr_row =
        std::next(output_data, static_cast<std::ptrdiff_t>(row_offset));
    Eigen::Map<Eigen::VectorXd> row_map(data_ptr_row, N + 1);
    row_map[0] = time;
    row_map.tail(N) = T;
}

void TransientSolver::outputs_first_last() {
    SOLVER_PROFILE_SCOPE("Outputs");
    if (output_data == nullptr) {
        return;
    }

    if (idata_out == 0) {
        save_temp_data();
    } else if (idata_out < num_outputs - 1) {
        idata_out = num_outputs - 1;
        save_temp_data();
    }
}

void TransientSolver::outputs() {
    SOLVER_PROFILE_SCOPE("Outputs");
    if (output_data == nullptr) {
        return;
    }

    if (((time_iter + 1) % wait_n_dtimes) == 0) {
        ++idata_out;
        save_temp_data();
    }
}

void TransientSolver::restart_solve() {
    if (!solver_initialized) {
        throw std::invalid_argument(
            "Solver has not been initialized. Please call initialize() before"
            " attempting to solve the system.");
    }

    solver_converged = false;
    time = start_time;
    tmm.time = time;

    time_iter = -1;
    idata_out = 0;

    std::cout << "(Re)starting solve..." << '\n';
}

}  // namespace pycanha
