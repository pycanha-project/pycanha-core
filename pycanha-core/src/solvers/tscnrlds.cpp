#include "pycanha-core/solvers/tscnrlds.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/solvers/tscnrl.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha {

TSCNRLDS::TSCNRLDS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : TSCNRL(std::move(tmm_shptr)) {
    solver_name = "TSCNRLDS";
    output_model_name = "TSCNRLDS";
}

TSCNRLDS::~TSCNRLDS() {
    // Ensure MKL resources are freed even if deinitialize() wasn't called
    // Only call if we were initialized to avoid double-deinitialize
    release_solver_resources();
}

void TSCNRLDS::initialize() {
    release_solver_resources();
    // Throws before anything is built when the options are not available.
    const auto options = linear_solver_options(engine, solver_type);
    _linear_solver = make_linear_solver(options);
    TSCNRL::initialize_common();

    SPDLOG_LOGGER_INFO(get_logger(), "{} initializing ({} {})", solver_name,
                       to_string(engine),
                       to_string(resolve_solver_type(engine, solver_type)));

    sparse_utils::add_zero_diag_square(_k_matrix);
    sparse_utils::set_to_zero(_k_matrix);

    _boundary_matrix = KRdb + KLdb;
    sparse_utils::set_to_zero(_boundary_matrix);

    _upper_kl_indices.resize(KLdd.nonZeros());
    _lower_kl_indices.resize(KLdd.nonZeros());
    _upper_kr_indices.resize(KRdd.nonZeros());
    _lower_kr_indices.resize(KRdd.nonZeros());
    _diagonal_indices.resize(to_sizet(nd));

    _radiation_linear_term = VectorXd::Zero(nd);
    _kt_q_n0 = VectorXd::Zero(nd);
    _ones_domain = VectorXd::Ones(nd);
    _ones_boundary = VectorXd::Ones(nb);

    // TODO(PYC-402): Replace pointer arithmetic with Eigen iterators to avoid
    // manual index pointer math.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // NOLINTBEGIN(readability-suspicious-call-argument)
    for (Index row = 0; row < KLdd.outerSize(); ++row) {
        for (Index idx = KLdd.outerIndexPtr()[row];
             idx < KLdd.outerIndexPtr()[row + 1]; ++idx) {
            const Index col = KLdd.innerIndexPtr()[idx];
            _upper_kl_indices[idx] = static_cast<int>(
                &_k_matrix.coeffRef(row, col) - _k_matrix.valuePtr());
            // TODO(PYC-403): Verify symmetric accessor ordering.
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            _lower_kl_indices[idx] = static_cast<int>(
                &_k_matrix.coeffRef(col, row) - _k_matrix.valuePtr());
        }
    }

    for (Index row = 0; row < KRdd.outerSize(); ++row) {
        for (Index idx = KRdd.outerIndexPtr()[row];
             idx < KRdd.outerIndexPtr()[row + 1]; ++idx) {
            const Index col = KRdd.innerIndexPtr()[idx];
            _upper_kr_indices[idx] = static_cast<int>(
                &_k_matrix.coeffRef(row, col) - _k_matrix.valuePtr());
            // TODO(PYC-403): Verify symmetric accessor ordering.
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            _lower_kr_indices[idx] = static_cast<int>(
                &_k_matrix.coeffRef(col, row) - _k_matrix.valuePtr());
        }
    }
    // NOLINTEND(readability-suspicious-call-argument)
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    for (Index row = 0; row < nd; ++row) {
        _diagonal_indices[to_sizet(row)] = static_cast<int>(
            &_k_matrix.coeffRef(row, row) - _k_matrix.valuePtr());
    }

    _t3_domain = VectorXd::Zero(nd);
    _t3_boundary = VectorXd::Zero(nb);
    _t4_domain = VectorXd::Zero(nd);
    _t4_boundary = VectorXd::Zero(nb);

    build_conductance_matrix();

    _linear_solver->analyze_pattern(_k_matrix);
    if (!_linear_solver->succeeded()) {
        throw std::runtime_error(solver_name + ": analysis failed (" +
                                 _linear_solver->error_message() + ")");
    }
    suggest_min_degree_if_dense(*_linear_solver, options, nd, solver_name);
    solver_initialized = true;
}

void TSCNRLDS::solve() {
    SPDLOG_LOGGER_INFO(get_logger(), "TSCNRLDS solving...");

    const FormulaExecutionGuard formula_execution(*this);

    restart_solve();
    callback_transient_time_change();
    callback_solver_loop();
    outputs_first_last();

    build_capacities();

    for (time_iter = 0; time_iter < num_time_steps; ++time_iter) {
        callback_solver_loop();
        build_conductance_matrix();
        build_heat_flux();
        store_heat_flux_at_n0();

        time += dtime;
        tmm.time = time;
        callback_transient_time_change();

        for (solver_iter = 0; solver_iter < max_iters; ++solver_iter) {
            callback_solver_loop();
            build_conductance_matrix();
            build_heat_flux();
            add_capacities_to_matrix();
            solve_step();

            solver_converged = temperature_convergence_check();

            {
                PYCANHA_PROFILE_SCOPE("Write Td in TMM");
                Td = Td_solver;
            }

            if (solver_converged) {
                break;
            }
        }

        if (!solver_converged) {
            Index max_index = -1;
            dTd.cwiseAbs().maxCoeff(&max_index);
            SPDLOG_LOGGER_ERROR(
                get_logger(), "TSCNRLDS did not converge after {} iterations.",
                max_iters);
            SPDLOG_LOGGER_ERROR(get_logger(), "Time iter: {} Time: {} s",
                                time_iter, time);
            SPDLOG_LOGGER_ERROR(get_logger(), "Max. dT: {} K at index: {}",
                                max_dT, max_index);
        }

        callback_transient_after_timestep();
        outputs();
    }

    outputs_first_last();
}

void TSCNRLDS::deinitialize() { release_solver_resources(); }

void TSCNRLDS::release_solver_resources() {
    _linear_solver.reset();
    solver_initialized = false;
}

void TSCNRLDS::build_capacities() {
    PYCANHA_PROFILE_SCOPE("Build C");
    _capacities = Cd;
    _capacities.array() += eps_capacity;
    _capacities_inverse = _capacities.cwiseInverse();
}

void TSCNRLDS::build_conductance_matrix() {
    PYCANHA_PROFILE_SCOPE("Linearization");

    _t3_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
    _t3_boundary = (4.0 * STF_BOLTZ) * Tb.array().cube();
    _t4_domain = STF_BOLTZ * Td.array().square().square();
    _t4_boundary = STF_BOLTZ * Tb.array().square().square();

    sparse_utils::set_to_zero(_k_matrix);
    sparse_utils::set_to_zero(_boundary_matrix);

    sparse_utils::copy_2_values_with_idx(_k_matrix.valuePtr(), KRdd.valuePtr(),
                                         _lower_kr_indices, _upper_kr_indices);

    _rhs = -(KRdd.selfadjointView<Eigen::Upper>() * _ones_domain);
    _rhs.noalias() -= KRdb * _ones_boundary;

    sparse_utils::copy_values_with_idx(_k_matrix.valuePtr(), _rhs.data(),
                                       _diagonal_indices);

    // TODO(PYC-404): Replace pointer arithmetic with Eigen iterators within
    // conductance assembly.
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    for (Index idx = 0; idx < _k_matrix.nonZeros(); ++idx) {
        const Index column = _k_matrix.innerIndexPtr()[idx];
        _k_matrix.valuePtr()[idx] *= _t3_domain[column];
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

    static const double q_alpha = -0.75;

    _radiation_linear_term.noalias() = q_alpha * (_k_matrix * Td);

    sparse_utils::copy_sum_2_values_with_idx(_k_matrix.valuePtr(),
                                             KLdd.valuePtr(), _lower_kl_indices,
                                             _upper_kl_indices);

    _rhs =
        KLdd.selfadjointView<Eigen::Upper>() * (-VectorXd::Ones(KLdd.cols())) -
        KLdb * VectorXd::Ones(KLdb.cols());
    sparse_utils::copy_sum_values_with_idx(_k_matrix.valuePtr(), _rhs.data(),
                                           _diagonal_indices);
}

void TSCNRLDS::build_heat_flux() {
    PYCANHA_PROFILE_SCOPE("Build Q");
    Q = QI_sp + QS_sp + QA_sp + QE_sp + QR_sp;
    new (&Qd) WrappVectorXd(Q.data(), nd);
    Qd += _radiation_linear_term + KLdb * Tb + KRdb * _t4_boundary;
}

void TSCNRLDS::store_heat_flux_at_n0() {
    PYCANHA_PROFILE_SCOPE("Store Q_n0");
    _kt_q_n0 = _k_matrix * Td + Qd;
    _heat_flux_n0 = _kt_q_n0 + (2.0 / dtime) * _capacities.cwiseProduct(Td);
}

void TSCNRLDS::euler_step() {
    PYCANHA_PROFILE_SCOPE("Euler Step");
    dTd = dtime * _kt_q_n0.cwiseProduct(_capacities_inverse);
    Td += dTd;
}

void TSCNRLDS::add_capacities_to_matrix() {
    PYCANHA_PROFILE_SCOPE("Add C to K");
    _k_matrix.diagonal() -= (2.0 / dtime) * _capacities;
}

void TSCNRLDS::solve_step() {
    PYCANHA_PROFILE_SCOPE("Solver Step");
    _rhs = Qd + _heat_flux_n0;

    // The factorised matrix is -K + 2C/dt, with a positive diagonal. The
    // negated values stay in _k_matrix until the next assembly.
    _k_matrix.coeffs() *= -1.0;
    _linear_solver->factorize_and_solve(_k_matrix, _rhs, Td_solver);
    if (!_linear_solver->succeeded()) {
        throw std::runtime_error(solver_name + ": solve failed (" +
                                 _linear_solver->error_message() + ")");
    }
}

}  // namespace pycanha
