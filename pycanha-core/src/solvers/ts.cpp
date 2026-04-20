#include "pycanha-core/solvers/ts.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha {

namespace {

constexpr std::array<DataModelAttribute, 13> k_dense_attributes = {
    DataModelAttribute::T,   DataModelAttribute::C,  DataModelAttribute::QS,
    DataModelAttribute::QA,  DataModelAttribute::QE, DataModelAttribute::QI,
    DataModelAttribute::QR,  DataModelAttribute::A,  DataModelAttribute::APH,
    DataModelAttribute::EPS, DataModelAttribute::FX, DataModelAttribute::FY,
    DataModelAttribute::FZ,
};

std::vector<Index> build_node_numbers(const Nodes& nodes) {
    std::vector<Index> node_numbers;
    node_numbers.reserve(to_sizet(nodes.get_num_nodes()));

    for (Index index = 0; index < nodes.get_num_nodes(); ++index) {
        const auto node_num = nodes.get_node_num_from_idx(index);
        if (!node_num.has_value()) {
            throw std::runtime_error(
                "Could not resolve node number for solver output mapping");
        }

        node_numbers.push_back(to_idx(*node_num));
    }

    return node_numbers;
}

void set_dense_output(DataModel& model, DataModelAttribute attribute,
                      Index row_index, double time,
                      const Eigen::Ref<const Eigen::VectorXd>& values) {
    model.get_dense_attribute(attribute).set_row(row_index, time, values);
}

Eigen::VectorXd densify_sparse_vector(const Eigen::SparseVector<double>& sparse,
                                      Index size) {
    Eigen::VectorXd dense = Eigen::VectorXd::Zero(size);
    for (Eigen::SparseVector<double>::InnerIterator it(sparse); it; ++it) {
        dense(it.index()) = it.value();
    }

    return dense;
}

SparseTimeSeries::SparseMatrixType build_full_coupling_matrix(
    const CouplingMatrices& matrices) {
    using SparseMatrix = SparseTimeSeries::SparseMatrixType;
    using Triplet = Eigen::Triplet<double>;

    const Index nd = matrices.sparse_dd.rows();
    const Index total_nodes =
        matrices.sparse_db.rows() + matrices.sparse_db.cols();

    std::vector<Triplet> triplets;
    triplets.reserve(to_sizet((matrices.sparse_dd.nonZeros() * 2) +
                              (matrices.sparse_db.nonZeros() * 2) +
                              (matrices.sparse_bb.nonZeros() * 2)));

    for (Index row = 0; row < matrices.sparse_dd.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_dd, row); it;
             ++it) {
            triplets.emplace_back(it.row(), it.col(), it.value());
            if (it.row() != it.col()) {
                triplets.emplace_back(it.col(), it.row(), it.value());
            }
        }
    }

    for (Index row = 0; row < matrices.sparse_db.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_db, row); it;
             ++it) {
            const Index boundary_col = nd + it.col();
            triplets.emplace_back(it.row(), boundary_col, it.value());
            triplets.emplace_back(boundary_col, it.row(), it.value());
        }
    }

    for (Index row = 0; row < matrices.sparse_bb.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_bb, row); it;
             ++it) {
            const Index mapped_row = nd + it.row();
            const Index mapped_col = nd + it.col();
            triplets.emplace_back(mapped_row, mapped_col, it.value());
            if (mapped_row != mapped_col) {
                triplets.emplace_back(mapped_col, mapped_row, it.value());
            }
        }
    }

    SparseMatrix output(total_nodes, total_nodes);
    output.setFromTriplets(triplets.begin(), triplets.end(),
                           [](double lhs, double rhs) { return lhs + rhs; });
    output.makeCompressed();
    return output;
}

void append_or_replace_sparse_sample(SparseTimeSeries& series, double time,
                                     SparseTimeSeries::SparseMatrixType matrix,
                                     double eps_time) {
    if ((series.num_timesteps() > 0) &&
        (std::abs(series.time_at(series.num_timesteps() - 1) - time) <=
         eps_time)) {
        series.at(series.num_timesteps() - 1) = std::move(matrix);
        return;
    }

    series.push_back(time, std::move(matrix));
}

}  // namespace

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

// TODO: Refactor initialize_common naming throughout the solver hierarchy to
// avoid duplicate inherited member warnings once the API is stabilized.
// cppcheck-suppress duplInheritedMember
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

    if (output_model_name.empty()) {
        throw std::invalid_argument("Output model name must not be empty");
    }

    DataModel model(build_node_numbers(tmm.nodes()));
    for (const auto attribute : k_dense_attributes) {
        if (!output_config.has(attribute)) {
            continue;
        }

        model.get_dense_attribute(attribute).resize(num_outputs, N);
    }
    if (output_config.has(DataModelAttribute::KL)) {
        model.get_sparse_attribute(DataModelAttribute::KL)
            .reserve(to_sizet(num_outputs));
    }
    if (output_config.has(DataModelAttribute::KR)) {
        model.get_sparse_attribute(DataModelAttribute::KR)
            .reserve(to_sizet(num_outputs));
    }

    tmm.thermal_data.models().add_model(output_model_name, std::move(model));
}

void TransientSolver::save_output_data() {
    auto& model = tmm.thermal_data.models().get_model(output_model_name);

    if (output_config.has(DataModelAttribute::T)) {
        set_dense_output(model, DataModelAttribute::T, idata_out, time, T);
    }
    if (output_config.has(DataModelAttribute::C)) {
        set_dense_output(model, DataModelAttribute::C, idata_out, time, C);
    }
    if (output_config.has(DataModelAttribute::QS)) {
        const auto dense = densify_sparse_vector(QS_sp, N);
        set_dense_output(model, DataModelAttribute::QS, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::QA)) {
        const auto dense = densify_sparse_vector(QA_sp, N);
        set_dense_output(model, DataModelAttribute::QA, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::QE)) {
        const auto dense = densify_sparse_vector(QE_sp, N);
        set_dense_output(model, DataModelAttribute::QE, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::QI)) {
        const auto dense = densify_sparse_vector(QI_sp, N);
        set_dense_output(model, DataModelAttribute::QI, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::QR)) {
        const auto dense = densify_sparse_vector(QR_sp, N);
        set_dense_output(model, DataModelAttribute::QR, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::A)) {
        const auto dense = densify_sparse_vector(tmm.nodes().a_vector, N);
        set_dense_output(model, DataModelAttribute::A, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::APH)) {
        const auto dense = densify_sparse_vector(tmm.nodes().aph_vector, N);
        set_dense_output(model, DataModelAttribute::APH, idata_out, time,
                         dense);
    }
    if (output_config.has(DataModelAttribute::EPS)) {
        const auto dense = densify_sparse_vector(tmm.nodes().eps_vector, N);
        set_dense_output(model, DataModelAttribute::EPS, idata_out, time,
                         dense);
    }
    if (output_config.has(DataModelAttribute::FX)) {
        const auto dense = densify_sparse_vector(tmm.nodes().fx_vector, N);
        set_dense_output(model, DataModelAttribute::FX, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::FY)) {
        const auto dense = densify_sparse_vector(tmm.nodes().fy_vector, N);
        set_dense_output(model, DataModelAttribute::FY, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::FZ)) {
        const auto dense = densify_sparse_vector(tmm.nodes().fz_vector, N);
        set_dense_output(model, DataModelAttribute::FZ, idata_out, time, dense);
    }
    if (output_config.has(DataModelAttribute::KL)) {
        append_or_replace_sparse_sample(
            model.get_sparse_attribute(DataModelAttribute::KL), time,
            build_full_coupling_matrix(ltcs), eps_time);
    }
    if (output_config.has(DataModelAttribute::KR)) {
        append_or_replace_sparse_sample(
            model.get_sparse_attribute(DataModelAttribute::KR), time,
            build_full_coupling_matrix(rtcs), eps_time);
    }
}

void TransientSolver::outputs_first_last() {
    PYCANHA_PROFILE_SCOPE("Outputs");
    if (idata_out == 0) {
        save_output_data();
    } else if (idata_out < num_outputs - 1) {
        idata_out = num_outputs - 1;
        save_output_data();
    }
}

void TransientSolver::outputs() {
    PYCANHA_PROFILE_SCOPE("Outputs");
    if (((time_iter + 1) % wait_n_dtimes) == 0) {
        ++idata_out;
        save_output_data();
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

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "(Re)starting solve...");
}

}  // namespace pycanha
