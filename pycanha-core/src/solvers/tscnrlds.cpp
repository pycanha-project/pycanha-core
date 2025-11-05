#include "pycanha-core/solvers/tscnrlds.hpp"

#include <cstddef>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/solvers/solver.hpp"

#if PYCANHA_USE_MKL
#include <mkl_cblas.h>
#include <mkl_types.h>
#include <mkl_pardiso.h>
#include <mkl_service.h>
#endif

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/tscnrl.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha {

TSCNRLDS::TSCNRLDS(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : TSCNRL(std::move(tmm_shptr)) {
    solver_name = "TSCNRLDS";
    output_table_name = "TSCNRLDS_OUTPUT";
}

TSCNRLDS::~TSCNRLDS() {
    // Ensure MKL resources are freed even if deinitialize() wasn't called
    // Only call if we were initialized to avoid double-deinitialize
    if (solver_initialized) {
        deinitialize();
    }
}

void TSCNRLDS::initialize() {
    TSCNRL::initialize_common();

#if PYCANHA_USE_MKL
    std::cout << "TSCNRLDS (MKL) initializing..." << '\n';
    std::cout << "MKL threads: " << mkl_get_max_threads() << '\n';
#else
    std::cout << "TSCNRLDS (Eigen) initializing..." << '\n';
#endif

    sparse_utils::add_zero_diag_square(_k_matrix);
    sparse_utils::set_to_zero(_k_matrix);

    _boundary_matrix = KRdb + KLdb;
    sparse_utils::set_to_zero(_boundary_matrix);

    _upper_kl_indices.resize(KLdd.nonZeros());
    _lower_kl_indices.resize(KLdd.nonZeros());
    _upper_kr_indices.resize(KRdd.nonZeros());
    _lower_kr_indices.resize(KRdd.nonZeros());
    _diagonal_indices.resize(static_cast<std::size_t>(nd));

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
        _diagonal_indices[static_cast<std::size_t>(row)] = static_cast<int>(
            &_k_matrix.coeffRef(row, row) - _k_matrix.valuePtr());
    }

    _t3_domain = VectorXd::Zero(nd);
    _t3_boundary = VectorXd::Zero(nb);
    _t4_domain = VectorXd::Zero(nd);
    _t4_boundary = VectorXd::Zero(nb);

    build_conductance_matrix();

#if PYCANHA_USE_MKL
    _pardiso_perm.assign(static_cast<std::size_t>(nd), 0);
    _pardiso_size = static_cast<MKL_INT>(nd);
    _pardiso_maxfct = 1;
    _pardiso_mnum = 1;
    _pardiso_phase = 0;
    _pardiso_msglvl = 0;
    _pardiso_nrhs = 1;
    _pardiso_error = 0;

    // Convert Eigen's int indices to MKL_INT (long long int)
    const auto outer_size = _k_matrix.outerSize() + 1;
    const auto inner_nnz = _k_matrix.nonZeros();
    _k_matrix_outer_index.resize(outer_size);
    _k_matrix_inner_index.resize(inner_nnz);

    std::copy(_k_matrix.outerIndexPtr(),
              _k_matrix.outerIndexPtr() + outer_size,
              _k_matrix_outer_index.begin());
    std::copy(_k_matrix.innerIndexPtr(),
              _k_matrix.innerIndexPtr() + inner_nnz,
              _k_matrix_inner_index.begin());

    // TODO(PYC-405): Wrap MKL control structure access to avoid reinterpret
    // casts once the solver interface is refactored.
    // TODO(PYC-405): Wrap MKL pointers with helper to drop reinterpret
    // casts.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    pardisoinit(reinterpret_cast<void*>(_pardiso_pt.data()), &_pardiso_mtype,
                _pardiso_iparm.data());
    _pardiso_iparm[34] = 1;
    _pardiso_iparm[18] = 0;
    _pardiso_iparm[17] = 0;
    _pardiso_iparm[1] = 3;
    _pardiso_iparm[3] = pardiso_iparm_3;
    _pardiso_iparm[23] = 0;

    _pardiso_phase = 11;
    pardiso(reinterpret_cast<void*>(_pardiso_pt.data()), &_pardiso_maxfct,
        &_pardiso_mnum, &_pardiso_mtype, &_pardiso_phase,
        &_pardiso_size, _k_matrix.valuePtr(), _k_matrix_outer_index.data(),
        _k_matrix_inner_index.data(), _pardiso_perm.data(), &_pardiso_nrhs,
        _pardiso_iparm.data(), &_pardiso_msglvl, _rhs.data(),
        Td_solver.data(), &_pardiso_error);

    if (_pardiso_error != 0) {
        throw std::runtime_error(
            "MKL PARDISO initialization failed with error " +
            std::to_string(_pardiso_error));
    }

    solver_initialized = true;
#else
    // Ensure matrix is in compressed format and properly structured
    _k_matrix.makeCompressed();
    
    // Validate matrix dimensions before analysis
    if (_k_matrix.rows() == 0 || _k_matrix.cols() == 0) {
        throw std::runtime_error(
            "Cannot analyze empty matrix in Eigen SparseLU initialization");
    }
    
    if (_k_matrix.rows() != _k_matrix.cols()) {
        throw std::runtime_error(
            "Matrix must be square for SparseLU factorization");
    }
    
    // Note: analyzePattern only analyzes the sparsity pattern, not values
    // It should not fail for a properly structured matrix
    _eigen_solver.analyzePattern(_k_matrix);
    
    // Check if pattern analysis succeeded
    // Note: info() might not be properly set after analyzePattern in some Eigen versions
    // The actual factorization happens in solve_step()
    solver_initialized = true;
#endif
}

void TSCNRLDS::solve() {
    if constexpr (PROFILING) {
        Instrumentor::get().begin_session("TSCNRLDS SOLVER");
    }
    std::cout << "TSCNRLDS solving..." << '\n';

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

        for (solver_iter = 0; solver_iter < MAX_ITERS; ++solver_iter) {
            callback_solver_loop();
            build_conductance_matrix();
            build_heat_flux();
            add_capacities_to_matrix();
            solve_step();

            solver_converged = temperature_convergence_check();

            {
                SOLVER_PROFILE_SCOPE("Write Td in TMM");
                Td = Td_solver;
            }

            if (solver_converged) {
                break;
            }
        }

        if (!solver_converged) {
            Index max_index = -1;
            dTd.cwiseAbs().maxCoeff(&max_index);
            std::cout << "ERROR: TSCNRLDS did not converge after " << MAX_ITERS
                      << " iterations." << '\n';
            std::cout << "Time iter: " << time_iter << " Time: " << time << " s"
                      << '\n';
            std::cout << "Max. dT: " << max_dT << " K at index: " << max_index
                      << '\n';
        }

        callback_transient_after_timestep();
        outputs();
    }

    outputs_first_last();
    if constexpr (PROFILING) {
        Instrumentor::get().end_session();
    }
}

void TSCNRLDS::deinitialize() {
#if PYCANHA_USE_MKL
    if (solver_initialized) {
        _pardiso_phase = -1;
        // TODO(PYC-405): Wrap MKL pointers with helper to drop reinterpret casts.
        // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
        pardiso(reinterpret_cast<void*>(_pardiso_pt.data()),
            &_pardiso_maxfct, &_pardiso_mnum, &_pardiso_mtype,
            &_pardiso_phase, &_pardiso_size, _k_matrix.valuePtr(),
            _k_matrix_outer_index.data(), _k_matrix_inner_index.data(),
            _pardiso_perm.data(), &_pardiso_nrhs, _pardiso_iparm.data(),
            &_pardiso_msglvl, _rhs.data(), Td_solver.data(),
            &_pardiso_error);
        // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    }
    
    // Always free MKL buffers, even if solver wasn't initialized
    mkl_thread_free_buffers();
    mkl_free_buffers();
    
    _pardiso_perm.clear();
    _k_matrix_outer_index.clear();
    _k_matrix_inner_index.clear();
#endif
    solver_initialized = false;
}

void TSCNRLDS::build_capacities() {
    SOLVER_PROFILE_SCOPE("Build C");
    _capacities = Cd;
    _capacities.array() += eps_capacity;
    _capacities_inverse = _capacities.cwiseInverse();
}

void TSCNRLDS::build_conductance_matrix() {
    SOLVER_PROFILE_SCOPE("Linearization");

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
    SOLVER_PROFILE_SCOPE("Build Q");
    Q = QI_sp + QS_sp + QA_sp + QE_sp + QR_sp;
    new (&Qd) WrappVectorXd(Q.data(), nd);
    Qd += _radiation_linear_term + KLdb * Tb + KRdb * _t4_boundary;
}

void TSCNRLDS::store_heat_flux_at_n0() {
    SOLVER_PROFILE_SCOPE("Store Q_n0");
    _kt_q_n0 = _k_matrix * Td + Qd;
    _heat_flux_n0 = _kt_q_n0 + (2.0 / dtime) * _capacities.cwiseProduct(Td);
}

void TSCNRLDS::euler_step() {
    SOLVER_PROFILE_SCOPE("Euler Step");
    dTd = dtime * _kt_q_n0.cwiseProduct(_capacities_inverse);
    Td += dTd;
}

void TSCNRLDS::add_capacities_to_matrix() {
    SOLVER_PROFILE_SCOPE("Add C to K");
    _k_matrix.diagonal() -= (2.0 / dtime) * _capacities;
}

void TSCNRLDS::solve_step() {
    SOLVER_PROFILE_SCOPE("Solver Step");
    _rhs = Qd + _heat_flux_n0;

#if PYCANHA_USE_MKL
    cblas_dscal(static_cast<int>(_k_matrix.nonZeros()), -1.0,
                _k_matrix.valuePtr(), 1);

    // Update MKL_INT index arrays if matrix structure changed
    const auto inner_nnz = _k_matrix.nonZeros();
    if (_k_matrix_inner_index.size() != static_cast<std::size_t>(inner_nnz)) {
        _k_matrix_inner_index.resize(inner_nnz);
        std::copy(_k_matrix.innerIndexPtr(),
                  _k_matrix.innerIndexPtr() + inner_nnz,
                  _k_matrix_inner_index.begin());
    }

    _pardiso_phase = 23;
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    pardiso(reinterpret_cast<void*>(_pardiso_pt.data()), &_pardiso_maxfct,
        &_pardiso_mnum, &_pardiso_mtype, &_pardiso_phase, &_pardiso_size,
        _k_matrix.valuePtr(), _k_matrix_outer_index.data(),
        _k_matrix_inner_index.data(), _pardiso_perm.data(), &_pardiso_nrhs,
        _pardiso_iparm.data(), &_pardiso_msglvl, _rhs.data(),
        Td_solver.data(), &_pardiso_error);
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

    if (_pardiso_error != 0) {
        throw std::runtime_error("MKL PARDISO solve failed with error " +
                                 std::to_string(_pardiso_error));
    }
#else
    _k_matrix *= -1.0;
    _k_matrix.makeCompressed();
    _eigen_solver.factorize(_k_matrix);
    if (_eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Eigen SparseLU factorization failed during solve");
    }
    Td_solver = _eigen_solver.solve(_rhs);
    if (_eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen SparseLU solve failed");
    }
    _k_matrix *= -1.0;
#endif
}

}  // namespace pycanha
