#include "pycanha-core/solvers/sslu.hpp"

#include <chrono>
#include <iostream>
#include <utility>

#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha {

SSLU::SSLU(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : SteadyStateSolver(std::move(tmm_shptr)),
      _t_cubed_domain(nullptr, Index{0}),
      _t_cubed_boundary(nullptr, Index{0}),
      _t_fourth_domain(nullptr, Index{0}),
      _t_fourth_boundary(nullptr, Index{0}) {}

void SSLU::initialize() {
    this->initialize_common();

    _k_matrix.conservativeResize(nd, nd);
    _k_matrix.setIdentity();
    _k_matrix += KRdd.selfadjointView<Eigen::Upper>();
    _k_matrix += KLdd.selfadjointView<Eigen::Upper>();
    sparse_utils::set_to_zero(_k_matrix);

    _t_cubed = VectorXd::Zero(nd + nb);
    _t_fourth = VectorXd::Zero(nd + nb);

    new (&_t_cubed_domain) WrappVectorXd(_t_cubed.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&_t_cubed_boundary) WrappVectorXd(_t_cubed.data() + nd, nb);
    new (&_t_fourth_domain) WrappVectorXd(_t_fourth.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&_t_fourth_boundary) WrappVectorXd(_t_fourth.data() + nd, nb);

    _linear_solver.analyzePattern(_k_matrix);
    solver_initialized = true;
}

void SSLU::solve() {
    if (!solver_initialized) {
        std::cout << "Solver has not been initialized. Please call initialize()"
                  << " before solve()." << '\n';
        return;
    }
    std::cout << "SSLU solving..." << '\n';

    solver_converged = false;

    for (solver_iter = 0; solver_iter < MAX_ITERS; ++solver_iter) {
        Q = -(QI_sp + QS_sp + QA_sp + QE_sp + QR_sp);
        new (&Qd) WrappVectorXd(Q.data(), nd);

        _t_cubed = T.array().cube();
        _t_fourth = T.array().pow(4);
        new (&_t_cubed_domain) WrappVectorXd(_t_cubed.data(), nd);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        new (&_t_cubed_boundary) WrappVectorXd(_t_cubed.data() + nd, nb);
        new (&_t_fourth_domain) WrappVectorXd(_t_fourth.data(), nd);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        new (&_t_fourth_boundary) WrappVectorXd(_t_fourth.data() + nd, nb);

        Qd -= (KLdb * Tb + STPH_BOLTZ * (KRdb * _t_fourth_boundary));

        sparse_utils::set_to_zero(_k_matrix);
        _k_matrix += KRdd.selfadjointView<Eigen::Upper>();
        add_radiative_diagonal_to_matrix();

        _k_matrix =
            STPH_BOLTZ * 4.0 * (_k_matrix * _t_cubed_domain.asDiagonal());
        Qd += (3.0 / 4.0) * (_k_matrix * Td);

        add_conductive_diagonal_to_matrix();
        _k_matrix += KLdd.selfadjointView<Eigen::Upper>();

        const auto start = std::chrono::high_resolution_clock::now();
        _linear_solver.factorize(_k_matrix);
        if (_linear_solver.info() != Eigen::ComputationInfo::Success) {
            std::cout << "SSLU ERROR: Factorization failed. Error code: "
                      << _linear_solver.info() << '\n';
            return;
        }

        Td_solver = _linear_solver.solve(Qd);
        if (_linear_solver.info() != Eigen::ComputationInfo::Success) {
            std::cout << "SSLU ERROR: Solver failed. Error code: "
                      << _linear_solver.info() << '\n';
            return;
        }

        dTd = Td_solver - Td;
        Td = Td_solver;
        max_dT = dTd.cwiseAbs().maxCoeff();

        this->callback_solver_loop();

        if (max_dT < abstol_temp) {
            std::cout << "SSLU converged. Num. iters: " << solver_iter + 1
                      << ". Max. dT = " << max_dT << " K." << '\n';
            solver_converged = true;
            break;
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "ITER TIME " << duration.count() << " ms" << '\n';
    }

    if (!solver_converged) {
        std::cout << "ERROR. SSLU did NOT converge after " << MAX_ITERS
                  << " iterations. Max. dT = " << max_dT << " K." << '\n';
    }
}

void SSLU::add_radiative_diagonal_to_matrix() {
    _k_matrix.diagonal() =
        KRdd.selfadjointView<Eigen::Upper>() * (-VectorXd::Ones(KRdd.cols())) -
        KRdb * VectorXd::Ones(KRdb.cols());
}

void SSLU::add_conductive_diagonal_to_matrix() {
    _k_matrix.diagonal() -=
        KLdd.selfadjointView<Eigen::Upper>() * VectorXd::Ones(KLdd.cols()) +
        KLdb * VectorXd::Ones(KRdb.cols());
}

void SSLU::deinitialize() {
    std::cout << "De-initializing SSLU..." << '\n';
    std::cout << "ERROR: Not implemented yet." << '\n';
}

}  // namespace pycanha
