#include "pycanha-core/solvers/tscnrlds_jacobian.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/solvers/tscnrlds.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

#if PYCANHA_USE_MKL
#include <mkl_pardiso.h>
#include <mkl_types.h>
#endif

namespace pycanha {

namespace {

[[nodiscard]] bool is_heat_flux_entity(const std::string& type) {
    return !type.empty() && type.front() == 'Q';
}

[[nodiscard]] Index require_node_index(const Nodes& nodes, int node_num) {
    const auto resolved =
        nodes.get_idx_from_node_num(static_cast<NodeNum>(node_num));
    if (!resolved.has_value()) {
        throw std::invalid_argument("Jacobian references a missing node");
    }
    return *resolved;
}

}  // namespace

TSCNRLDS_JACOBIAN::TSCNRLDS_JACOBIAN(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : TSCNRLDS(std::move(tmm_shptr)),
      output_jacobian_table_name("TSCNRLDS_JACOBIAN_OUTPUT") {
    solver_name = "TSCNRLDS_JACOBIAN";
    output_table_name = "TSCNRLDS_OUTPUT";
}

void TSCNRLDS_JACOBIAN::initialize() {
    TSCNRLDS::initialize();

    collect_parameter_names();
    if (_parameter_names.empty()) {
        throw std::invalid_argument(
            "TSCNRLDS_JACOBIAN requires at least one formula with derivative "
            "metadata");
    }

    tmm.thermal_data.create_reset_table(
        output_jacobian_table_name, num_outputs,
        nd * to_idx(_parameter_names.size()) + 1);

    _d_kl_dd_matrices.clear();
    _d_kl_db_matrices.clear();
    _d_kr_dd_matrices.clear();
    _d_kr_db_matrices.clear();

    _d_kl_dd_matrices.reserve(_parameter_names.size());
    _d_kl_db_matrices.reserve(_parameter_names.size());
    _d_kr_dd_matrices.reserve(_parameter_names.size());
    _d_kr_db_matrices.reserve(_parameter_names.size());

    for (std::size_t parameter_index = 0;
         parameter_index < _parameter_names.size(); ++parameter_index) {
        _d_kl_dd_matrices.emplace_back(nd, nd);
        _d_kl_db_matrices.emplace_back(nd, nb);
        _d_kr_dd_matrices.emplace_back(nd, nd);
        _d_kr_db_matrices.emplace_back(nd, nb);
    }

    _d_capacity_matrix =
        DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));
    _d_heat_flux_matrix =
        DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));

    _m_k = DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));
    _m_q = DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));
    _m_c = DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));
    _mt = DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));
    _mb = DenseJacobian::Zero(nd, to_idx(_parameter_names.size()));

    _sp_nd_diag.resize(nd, nd);
    sparse_utils::add_zero_diag_square(_sp_nd_diag);

    std::unordered_map<std::string, Index> parameter_lookup;
    parameter_lookup.reserve(_parameter_names.size());
    for (std::size_t parameter_index = 0;
         parameter_index < _parameter_names.size(); ++parameter_index) {
        parameter_lookup.emplace(_parameter_names[parameter_index],
                                 to_idx(parameter_index));
    }

    for (const auto& formula : tmm.formulas.formulas()) {
        const auto* derivatives = formula->get_derivative_values();
        if ((derivatives == nullptr) || derivatives->empty()) {
            continue;
        }

        const auto& dependencies = formula->parameter_dependencies();
        if (dependencies.size() != derivatives->size()) {
            throw std::invalid_argument(
                "Formula derivative metadata size mismatch for " +
                formula->entity().string_representation());
        }

        for (std::size_t dependency_index = 0;
             dependency_index < dependencies.size(); ++dependency_index) {
            const auto search =
                parameter_lookup.find(dependencies[dependency_index]);
            if (search == parameter_lookup.end()) {
                continue;
            }

            fill_matrices(formula->entity(), search->second,
                          derivatives->at(dependency_index));
        }
    }
}

void TSCNRLDS_JACOBIAN::collect_parameter_names() {
    _parameter_names.clear();

    std::unordered_set<std::string> seen_parameters;
    for (const auto& formula : tmm.formulas.formulas()) {
        const auto* derivatives = formula->get_derivative_values();
        if ((derivatives == nullptr) || derivatives->empty()) {
            continue;
        }

        const auto& dependencies = formula->parameter_dependencies();
        if (dependencies.size() != derivatives->size()) {
            throw std::invalid_argument(
                "Formula derivative metadata size mismatch for " +
                formula->entity().string_representation());
        }

        std::copy_if(dependencies.begin(), dependencies.end(),
                     std::back_inserter(_parameter_names),
                     [&seen_parameters](const std::string& dependency) {
                         return seen_parameters.insert(dependency).second;
                     });
    }
}

void TSCNRLDS_JACOBIAN::fill_matrices(ThermalEntity& entity,
                                      Index parameter_index,
                                      double derivative_value) {
    const auto& type = entity.type();

    if ((type == "GL") || (type == "GR")) {
        const Index index_1 =
            require_node_index(tmm.nodes(), entity.node_index_1());
        const Index index_2 =
            require_node_index(tmm.nodes(), entity.node_index_2());

        const bool first_is_domain = index_1 < nd;
        const bool second_is_domain = index_2 < nd;

        if (first_is_domain && second_is_domain) {
            const auto row = std::min(index_1, index_2);
            const auto col = std::max(index_1, index_2);
            auto& matrix = (type == "GL")
                               ? _d_kl_dd_matrices[to_sizet(parameter_index)]
                               : _d_kr_dd_matrices[to_sizet(parameter_index)];
            matrix.coeffRef(row, col) += derivative_value;
            return;
        }

        if (first_is_domain != second_is_domain) {
            const auto diff_index = first_is_domain ? index_1 : index_2;
            const auto bound_index = first_is_domain ? index_2 : index_1;
            auto& matrix = (type == "GL")
                               ? _d_kl_db_matrices[to_sizet(parameter_index)]
                               : _d_kr_db_matrices[to_sizet(parameter_index)];
            matrix.coeffRef(diff_index, bound_index - nd) += derivative_value;
        }

        return;
    }

    if (type == "C") {
        const Index index =
            require_node_index(tmm.nodes(), entity.node_index_1());
        if (index < nd) {
            _d_capacity_matrix(index, parameter_index) += derivative_value;
        }
        return;
    }

    if (is_heat_flux_entity(type)) {
        const Index index =
            require_node_index(tmm.nodes(), entity.node_index_1());
        if (index < nd) {
            _d_heat_flux_matrix(index, parameter_index) += derivative_value;
        }
        return;
    }

    SPDLOG_LOGGER_WARN(get_logger(), "Unsupported jacobian entity type '{}'",
                       type);
}

void TSCNRLDS_JACOBIAN::build_mk() {
    for (std::size_t parameter_index = 0;
         parameter_index < _parameter_names.size(); ++parameter_index) {
        const auto& d_kl_dd = _d_kl_dd_matrices[parameter_index];
        const auto& d_kl_db = _d_kl_db_matrices[parameter_index];
        const auto& d_kr_dd = _d_kr_dd_matrices[parameter_index];
        const auto& d_kr_db = _d_kr_db_matrices[parameter_index];

        VectorXd conductive_diagonal =
            d_kl_dd.selfadjointView<Eigen::Upper>() * (-_ones_domain);
        conductive_diagonal.noalias() -= d_kl_db * _ones_boundary;

        VectorXd radiative_diagonal =
            d_kr_dd.selfadjointView<Eigen::Upper>() * (-_ones_domain);
        radiative_diagonal.noalias() -= d_kr_db * _ones_boundary;

        _m_k.col(to_idx(parameter_index)) =
            d_kl_dd.selfadjointView<Eigen::Upper>() * Td +
            d_kr_dd.selfadjointView<Eigen::Upper>() * _t4_domain +
            conductive_diagonal.cwiseProduct(Td) +
            radiative_diagonal.cwiseProduct(_t4_domain);
    }
}

void TSCNRLDS_JACOBIAN::build_mq() {
    for (std::size_t parameter_index = 0;
         parameter_index < _parameter_names.size(); ++parameter_index) {
        const auto& d_kl_db = _d_kl_db_matrices[parameter_index];
        const auto& d_kr_db = _d_kr_db_matrices[parameter_index];

        _m_q.col(to_idx(parameter_index)) =
            _d_heat_flux_matrix.col(to_idx(parameter_index));
        _m_q.col(to_idx(parameter_index)).noalias() += d_kl_db * Tb;
        _m_q.col(to_idx(parameter_index)).noalias() += d_kr_db * _t4_boundary;
    }
}

void TSCNRLDS_JACOBIAN::build_mc() {
    const VectorXd initial_temperature_derivative =
        _capacities_inverse.asDiagonal() * (-_kt_q_n0);
    _m_c = initial_temperature_derivative.asDiagonal() * _d_capacity_matrix;
}

void TSCNRLDS_JACOBIAN::solve_jacobian_step() {
#if PYCANHA_USE_MKL
    _pardiso_phase = 33;
    _pardiso_nrhs = static_cast<MKL_INT>(_parameter_names.size());
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    pardiso(reinterpret_cast<void*>(_pardiso_pt.data()), &_pardiso_maxfct,
            &_pardiso_mnum, &_pardiso_mtype, &_pardiso_phase, &_pardiso_size,
            _k_matrix.valuePtr(), _k_matrix_outer_index.data(),
            _k_matrix_inner_index.data(), _pardiso_perm.data(), &_pardiso_nrhs,
            _pardiso_iparm.data(), &_pardiso_msglvl, _mb.data(), _mt.data(),
            &_pardiso_error);
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    _pardiso_phase = 23;
    _pardiso_nrhs = 1;

    if (_pardiso_error != 0) {
        throw std::runtime_error(
            "MKL PARDISO jacobian solve failed with error " +
            std::to_string(_pardiso_error));
    }
#else
    _mt = _eigen_solver.solve(_mb);
    if (_eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen SparseLU jacobian solve failed");
    }
#endif
}

void TSCNRLDS_JACOBIAN::solve() {
    SPDLOG_LOGGER_INFO(get_logger(), "TSCNRLDS_JACOBIAN solving...");

    if (_parameter_names.empty()) {
        TSCNRLDS::solve();
        return;
    }

    restart_solve();
    tmm.thermal_data.get_table(output_jacobian_table_name).setZero();
    _mt.setZero();
    _mb.setZero();

    callback_transient_time_change();
    callback_solver_loop();
    outputs_first_last();

    build_capacities();

    build_conductance_matrix();
    build_heat_flux();
    store_heat_flux_at_n0();

    build_mc();
    build_mk();
    build_mq();
    save_jacobian_data();

    for (time_iter = 0; time_iter < num_time_steps; ++time_iter) {
        callback_solver_loop();
        build_conductance_matrix();
        build_heat_flux();
        store_heat_flux_at_n0();

        _sp_nd_diag.diagonal() = _capacities.array() * (2.0 / dtime);
        _mb = (_sp_nd_diag + _k_matrix) * _mt;
        _mb += _m_c + _m_k + _m_q;

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
                get_logger(),
                "TSCNRLDS_JACOBIAN did not converge after {} iterations.",
                MAX_ITERS);
            SPDLOG_LOGGER_ERROR(get_logger(), "Time iter: {} Time: {} s",
                                time_iter, time);
            SPDLOG_LOGGER_ERROR(get_logger(), "Max. dT: {} K at index: {}",
                                max_dT, max_index);
        }

        _sp_nd_diag.diagonal() = _capacities.array() * (2.0 / dtime);
#if PYCANHA_USE_MKL
        _kt_q_n0 = (-_k_matrix + _sp_nd_diag) * Td + Qd;
#else
        _kt_q_n0 = (_k_matrix + _sp_nd_diag) * Td + Qd;
#endif

        build_mc();
        build_mk();
        build_mq();

        _mb += _m_c + _m_k + _m_q;
        solve_jacobian_step();

        callback_transient_after_timestep();
        outputs();
        if (((time_iter + 1) % wait_n_dtimes) == 0) {
            save_jacobian_data();
        }
    }

    outputs_first_last();
    save_jacobian_data();
}

void TSCNRLDS_JACOBIAN::save_jacobian_data() {
    const auto parameter_count = to_idx(_parameter_names.size());
    tmm.thermal_data.get_table(output_jacobian_table_name)(idata_out, 0) = time;
    std::size_t output_index = 1U;
    for (Index node_index = 0; node_index < nd; ++node_index) {
        for (Index parameter_index = 0; parameter_index < parameter_count;
             ++parameter_index) {
            tmm.thermal_data.get_table(output_jacobian_table_name)(
                idata_out, to_idx(output_index)) =
                _mt(node_index, parameter_index);
            ++output_index;
        }
    }
}

}  // namespace pycanha
