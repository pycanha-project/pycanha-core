#include "pycanha-core/thermaldata/data_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"

namespace pycanha {
namespace {

using SparseMatrixType = SparseTimeSeries::SparseMatrixType;

constexpr std::array<DataModelAttribute, 16> k_all_attributes = {
    DataModelAttribute::T,   DataModelAttribute::C,  DataModelAttribute::QS,
    DataModelAttribute::QA,  DataModelAttribute::QE, DataModelAttribute::QI,
    DataModelAttribute::QR,  DataModelAttribute::A,  DataModelAttribute::APH,
    DataModelAttribute::EPS, DataModelAttribute::FX, DataModelAttribute::FY,
    DataModelAttribute::FZ,  DataModelAttribute::KL, DataModelAttribute::KR,
    DataModelAttribute::JAC,
};

Index find_node_column(const std::vector<Index>& node_numbers, Index node_num) {
    const auto iterator =
        std::find(node_numbers.begin(), node_numbers.end(), node_num);
    if (iterator == node_numbers.end()) {
        throw std::invalid_argument(
            "Requested node is not present in DataModel");
    }

    return static_cast<Index>(std::distance(node_numbers.begin(), iterator));
}

void validate_flow_inputs(const DataModel& model,
                          const SparseTimeSeries& coupling_series,
                          const char* coupling_name) {
    if (model.T().num_timesteps() == 0) {
        throw std::runtime_error("Temperature series has no data");
    }
    if (coupling_series.num_timesteps() == 0) {
        throw std::runtime_error(std::string(coupling_name) +
                                 " series has no data");
    }
    if (model.T().num_columns() != to_idx(model.node_numbers().size())) {
        throw std::runtime_error(
            "Temperature column count does not match DataModel node mapping");
    }
    if ((coupling_series.rows() != model.T().num_columns()) ||
        (coupling_series.cols() != model.T().num_columns())) {
        throw std::runtime_error(std::string(coupling_name) +
                                 " matrix dimensions do not match "
                                 "temperature series columns");
    }
}

Index find_floor_time_index(const Eigen::VectorXd& times, double time) {
    if (times.size() == 0) {
        throw std::runtime_error("Time series has no samples");
    }

    const double* begin = times.data();
    const double* end = std::next(begin, times.size());
    const double* upper_bound = std::upper_bound(begin, end, time + TOL);
    if (upper_bound == begin) {
        throw std::out_of_range("Requested time is before the first sample");
    }

    return static_cast<Index>(std::distance(begin, upper_bound) - 1);
}

Index find_ceil_time_index(const Eigen::VectorXd& times, double time) {
    if (times.size() == 0) {
        throw std::runtime_error("Time series has no samples");
    }

    const double* begin = times.data();
    const double* end = std::next(begin, times.size());
    const double* lower_bound = std::lower_bound(begin, end, time - TOL);
    if (lower_bound == end) {
        throw std::out_of_range("Requested time is after the last sample");
    }

    return static_cast<Index>(std::distance(begin, lower_bound));
}

std::vector<Index> build_time_row_selection(const DenseTimeSeries& temperature,
                                            double start_time,
                                            double end_time) {
    if (end_time < start_time) {
        throw std::invalid_argument("end_time must be greater than start_time");
    }

    const Index start_row =
        find_floor_time_index(temperature.times(), start_time);
    const Index end_row = find_ceil_time_index(temperature.times(), end_time);
    std::vector<Index> rows;
    rows.reserve(to_sizet(end_row - start_row + 1));

    for (Index row = start_row; row <= end_row; ++row) {
        rows.push_back(row);
    }

    return rows;
}

Index find_matching_sparse_time_index(const SparseTimeSeries& series,
                                      double time, const char* series_name) {
    const Eigen::VectorXd& times = series.times();
    if (times.size() == 0) {
        throw std::runtime_error(std::string(series_name) +
                                 " series has no samples");
    }

    const double* begin = times.data();
    const double* end = std::next(begin, times.size());
    const double* lower_bound = std::lower_bound(begin, end, time);

    if ((lower_bound != end) && (std::abs(*lower_bound - time) <= TOL)) {
        return static_cast<Index>(std::distance(begin, lower_bound));
    }
    if ((lower_bound != begin) &&
        (std::abs(*std::prev(lower_bound) - time) <= TOL)) {
        return static_cast<Index>(std::distance(begin, lower_bound) - 1);
    }

    throw std::runtime_error(std::string(series_name) +
                             " times do not match the resolved "
                             "temperature times");
}

std::vector<Index> build_sparse_row_selection(
    const DenseTimeSeries& temperature, const SparseTimeSeries& coupling,
    const std::vector<Index>& rows, const char* series_name) {
    std::vector<Index> sparse_rows;
    sparse_rows.reserve(rows.size());

    for (const Index row : rows) {
        sparse_rows.push_back(find_matching_sparse_time_index(
            coupling, temperature.times()(row), series_name));
    }

    return sparse_rows;
}

template <typename FlowFunction>
double sum_group_flow(const std::vector<Index>& node_columns_1,
                      const std::vector<Index>& node_columns_2,
                      const FlowFunction& flow_function) {
    double total_flow = 0.0;
    for (const Index node_column_1 : node_columns_1) {
        for (const Index node_column_2 : node_columns_2) {
            total_flow += flow_function(node_column_1, node_column_2);
        }
    }

    return total_flow;
}

double conductive_pair_flow(const DenseTimeSeries& temperature,
                            const SparseMatrixType& conductance,
                            Index temperature_row, Index node_column_1,
                            Index node_column_2) {
    if (node_column_1 == node_column_2) {
        return 0.0;
    }

    const double temperature_1 =
        temperature.values()(temperature_row, node_column_1);
    const double temperature_2 =
        temperature.values()(temperature_row, node_column_2);
    const double coupling_value =
        conductance.coeff(node_column_1, node_column_2);

    return coupling_value * (temperature_2 - temperature_1);
}

double radiative_pair_flow(const DenseTimeSeries& temperature,
                           const SparseMatrixType& conductance,
                           Index temperature_row, Index node_column_1,
                           Index node_column_2) {
    if (node_column_1 == node_column_2) {
        return 0.0;
    }

    const double temperature_1 =
        temperature.values()(temperature_row, node_column_1);
    const double temperature_2 =
        temperature.values()(temperature_row, node_column_2);
    const double coupling_value =
        conductance.coeff(node_column_1, node_column_2);

    return coupling_value * STF_BOLTZ *
           (std::pow(temperature_2, 4) - std::pow(temperature_1, 4));
}

template <typename FlowFunction>
Eigen::MatrixXd compute_flow_matrix(const DenseTimeSeries& temperature,
                                    const SparseTimeSeries& coupling,
                                    const std::vector<Index>& temperature_rows,
                                    const std::vector<Index>& sparse_rows,
                                    const std::vector<Index>& node_columns_1,
                                    const std::vector<Index>& node_columns_2,
                                    const FlowFunction& flow_function) {
    Eigen::MatrixXd output(static_cast<Index>(temperature_rows.size()), 2);

    for (Index output_row = 0;
         output_row < static_cast<Index>(temperature_rows.size());
         ++output_row) {
        const Index temperature_row = temperature_rows.at(to_sizet(output_row));
        const Index sparse_row = sparse_rows.at(to_sizet(output_row));
        const auto& matrix = coupling.at(sparse_row);
        output(output_row, 0) = temperature.times()(temperature_row);
        output(output_row, 1) = sum_group_flow(
            node_columns_1, node_columns_2,
            [&](Index node_column_1, Index node_column_2) {
                return flow_function(temperature, matrix, temperature_row,
                                     node_column_1, node_column_2);
            });
    }

    return output;
}

std::vector<Index> resolve_node_columns(
    const DataModel& model, const std::vector<Index>& node_numbers) {
    std::vector<Index> node_columns;
    node_columns.reserve(node_numbers.size());
    for (const Index node_num : node_numbers) {
        node_columns.push_back(
            find_node_column(model.node_numbers(), node_num));
    }

    return node_columns;
}

template <typename FlowFunction>
Eigen::MatrixXd compute_single_time_flow(const DataModel& model,
                                         const SparseTimeSeries& coupling,
                                         const std::vector<Index>& node_nums_1,
                                         const std::vector<Index>& node_nums_2,
                                         Index temperature_row,
                                         const char* series_name,
                                         const FlowFunction& flow_function) {
    validate_flow_inputs(model, coupling, series_name);
    const std::vector<Index> temperature_rows{temperature_row};
    const std::vector<Index> sparse_rows = build_sparse_row_selection(
        model.T(), coupling, temperature_rows, series_name);

    return compute_flow_matrix(
        model.T(), coupling, temperature_rows, sparse_rows,
        resolve_node_columns(model, node_nums_1),
        resolve_node_columns(model, node_nums_2), flow_function);
}

template <typename FlowFunction>
Eigen::MatrixXd compute_range_flow(const DataModel& model,
                                   const SparseTimeSeries& coupling,
                                   const std::vector<Index>& node_nums_1,
                                   const std::vector<Index>& node_nums_2,
                                   double start_time, double end_time,
                                   const char* series_name,
                                   const FlowFunction& flow_function) {
    validate_flow_inputs(model, coupling, series_name);
    const std::vector<Index> temperature_rows =
        build_time_row_selection(model.T(), start_time, end_time);
    const std::vector<Index> sparse_rows = build_sparse_row_selection(
        model.T(), coupling, temperature_rows, series_name);

    return compute_flow_matrix(
        model.T(), coupling, temperature_rows, sparse_rows,
        resolve_node_columns(model, node_nums_1),
        resolve_node_columns(model, node_nums_2), flow_function);
}

}  // namespace

DataModel::DataModel(std::vector<Index> node_numbers)
    : _node_numbers(std::move(node_numbers)) {}

DenseTimeSeries& DataModel::T() noexcept { return _T; }

const DenseTimeSeries& DataModel::T() const noexcept { return _T; }

DenseTimeSeries& DataModel::C() noexcept { return _C; }

const DenseTimeSeries& DataModel::C() const noexcept { return _C; }

DenseTimeSeries& DataModel::QS() noexcept { return _QS; }

const DenseTimeSeries& DataModel::QS() const noexcept { return _QS; }

DenseTimeSeries& DataModel::QA() noexcept { return _QA; }

const DenseTimeSeries& DataModel::QA() const noexcept { return _QA; }

DenseTimeSeries& DataModel::QE() noexcept { return _QE; }

const DenseTimeSeries& DataModel::QE() const noexcept { return _QE; }

DenseTimeSeries& DataModel::QI() noexcept { return _QI; }

const DenseTimeSeries& DataModel::QI() const noexcept { return _QI; }

DenseTimeSeries& DataModel::QR() noexcept { return _QR; }

const DenseTimeSeries& DataModel::QR() const noexcept { return _QR; }

DenseTimeSeries& DataModel::A() noexcept { return _A; }

const DenseTimeSeries& DataModel::A() const noexcept { return _A; }

DenseTimeSeries& DataModel::APH() noexcept { return _APH; }

const DenseTimeSeries& DataModel::APH() const noexcept { return _APH; }

DenseTimeSeries& DataModel::EPS() noexcept { return _EPS; }

const DenseTimeSeries& DataModel::EPS() const noexcept { return _EPS; }

DenseTimeSeries& DataModel::FX() noexcept { return _FX; }

const DenseTimeSeries& DataModel::FX() const noexcept { return _FX; }

DenseTimeSeries& DataModel::FY() noexcept { return _FY; }

const DenseTimeSeries& DataModel::FY() const noexcept { return _FY; }

DenseTimeSeries& DataModel::FZ() noexcept { return _FZ; }

const DenseTimeSeries& DataModel::FZ() const noexcept { return _FZ; }

SparseTimeSeries& DataModel::conductive_couplings() noexcept {
    return _conductive_couplings;
}

const SparseTimeSeries& DataModel::conductive_couplings() const noexcept {
    return _conductive_couplings;
}

SparseTimeSeries& DataModel::radiative_couplings() noexcept {
    return _radiative_couplings;
}

const SparseTimeSeries& DataModel::radiative_couplings() const noexcept {
    return _radiative_couplings;
}

DenseMatrixTimeSeries& DataModel::jacobian() noexcept { return _jacobian; }

const DenseMatrixTimeSeries& DataModel::jacobian() const noexcept {
    return _jacobian;
}

DenseTimeSeries& DataModel::get_dense_attribute(DataModelAttribute attr) {
    switch (attr) {
        case DataModelAttribute::T:
            return _T;
        case DataModelAttribute::C:
            return _C;
        case DataModelAttribute::QS:
            return _QS;
        case DataModelAttribute::QA:
            return _QA;
        case DataModelAttribute::QE:
            return _QE;
        case DataModelAttribute::QI:
            return _QI;
        case DataModelAttribute::QR:
            return _QR;
        case DataModelAttribute::A:
            return _A;
        case DataModelAttribute::APH:
            return _APH;
        case DataModelAttribute::EPS:
            return _EPS;
        case DataModelAttribute::FX:
            return _FX;
        case DataModelAttribute::FY:
            return _FY;
        case DataModelAttribute::FZ:
            return _FZ;
        case DataModelAttribute::KL:
        case DataModelAttribute::KR:
        case DataModelAttribute::JAC:
            break;
    }

    throw std::invalid_argument("Requested attribute is not dense");
}

const DenseTimeSeries& DataModel::get_dense_attribute(
    DataModelAttribute attr) const {
    switch (attr) {
        case DataModelAttribute::T:
            return _T;
        case DataModelAttribute::C:
            return _C;
        case DataModelAttribute::QS:
            return _QS;
        case DataModelAttribute::QA:
            return _QA;
        case DataModelAttribute::QE:
            return _QE;
        case DataModelAttribute::QI:
            return _QI;
        case DataModelAttribute::QR:
            return _QR;
        case DataModelAttribute::A:
            return _A;
        case DataModelAttribute::APH:
            return _APH;
        case DataModelAttribute::EPS:
            return _EPS;
        case DataModelAttribute::FX:
            return _FX;
        case DataModelAttribute::FY:
            return _FY;
        case DataModelAttribute::FZ:
            return _FZ;
        case DataModelAttribute::KL:
        case DataModelAttribute::KR:
        case DataModelAttribute::JAC:
            break;
    }

    throw std::invalid_argument("Requested attribute is not dense");
}

SparseTimeSeries& DataModel::get_sparse_attribute(DataModelAttribute attr) {
    switch (attr) {
        case DataModelAttribute::KL:
            return _conductive_couplings;
        case DataModelAttribute::KR:
            return _radiative_couplings;
        case DataModelAttribute::T:
        case DataModelAttribute::C:
        case DataModelAttribute::QS:
        case DataModelAttribute::QA:
        case DataModelAttribute::QE:
        case DataModelAttribute::QI:
        case DataModelAttribute::QR:
        case DataModelAttribute::A:
        case DataModelAttribute::APH:
        case DataModelAttribute::EPS:
        case DataModelAttribute::FX:
        case DataModelAttribute::FY:
        case DataModelAttribute::FZ:
        case DataModelAttribute::JAC:
            break;
    }

    throw std::invalid_argument("Requested attribute is not sparse");
}

const SparseTimeSeries& DataModel::get_sparse_attribute(
    DataModelAttribute attr) const {
    switch (attr) {
        case DataModelAttribute::KL:
            return _conductive_couplings;
        case DataModelAttribute::KR:
            return _radiative_couplings;
        case DataModelAttribute::T:
        case DataModelAttribute::C:
        case DataModelAttribute::QS:
        case DataModelAttribute::QA:
        case DataModelAttribute::QE:
        case DataModelAttribute::QI:
        case DataModelAttribute::QR:
        case DataModelAttribute::A:
        case DataModelAttribute::APH:
        case DataModelAttribute::EPS:
        case DataModelAttribute::FX:
        case DataModelAttribute::FY:
        case DataModelAttribute::FZ:
        case DataModelAttribute::JAC:
            break;
    }

    throw std::invalid_argument("Requested attribute is not sparse");
}

DenseMatrixTimeSeries& DataModel::get_matrix_attribute(
    DataModelAttribute attr) {
    if (attr != DataModelAttribute::JAC) {
        throw std::invalid_argument("Requested attribute is not a matrix");
    }

    return _jacobian;
}

const DenseMatrixTimeSeries& DataModel::get_matrix_attribute(
    DataModelAttribute attr) const {
    if (attr != DataModelAttribute::JAC) {
        throw std::invalid_argument("Requested attribute is not a matrix");
    }

    return _jacobian;
}

void DataModel::set_node_numbers(std::vector<Index> node_numbers) {
    _node_numbers = std::move(node_numbers);
}

std::vector<Index>& DataModel::node_numbers() noexcept { return _node_numbers; }

const std::vector<Index>& DataModel::node_numbers() const noexcept {
    return _node_numbers;
}

std::vector<DataModelAttribute> DataModel::populated_attributes() const {
    std::vector<DataModelAttribute> attributes;
    attributes.reserve(k_all_attributes.size());

    for (const auto attribute : k_all_attributes) {
        switch (attribute) {
            case DataModelAttribute::KL:
            case DataModelAttribute::KR:
                if (get_sparse_attribute(attribute).num_timesteps() > 0) {
                    attributes.push_back(attribute);
                }
                break;
            case DataModelAttribute::JAC:
                if (get_matrix_attribute(attribute).num_timesteps() > 0) {
                    attributes.push_back(attribute);
                }
                break;
            default:
                if (get_dense_attribute(attribute).num_timesteps() > 0) {
                    attributes.push_back(attribute);
                }
                break;
        }
    }

    return attributes;
}

Eigen::MatrixXd DataModel::flow_conductive(Index node_num_1,
                                           Index node_num_2) const {
    return flow_conductive(std::vector<Index>{node_num_1},
                           std::vector<Index>{node_num_2});
}

Eigen::MatrixXd DataModel::flow_conductive(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2) const {
    return compute_single_time_flow(*this, conductive_couplings(), node_nums_1,
                                    node_nums_2, 0, "Conductive coupling",
                                    conductive_pair_flow);
}

Eigen::MatrixXd DataModel::flow_conductive(Index node_num_1, Index node_num_2,
                                           double time) const {
    return flow_conductive(std::vector<Index>{node_num_1},
                           std::vector<Index>{node_num_2}, time);
}

Eigen::MatrixXd DataModel::flow_conductive(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2, double time) const {
    validate_flow_inputs(*this, conductive_couplings(), "Conductive coupling");
    return compute_single_time_flow(
        *this, conductive_couplings(), node_nums_1, node_nums_2,
        find_floor_time_index(T().times(), time), "Conductive coupling",
        conductive_pair_flow);
}

Eigen::MatrixXd DataModel::flow_conductive(Index node_num_1, Index node_num_2,
                                           double start_time,
                                           double end_time) const {
    return flow_conductive(std::vector<Index>{node_num_1},
                           std::vector<Index>{node_num_2}, start_time,
                           end_time);
}

Eigen::MatrixXd DataModel::flow_conductive(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2, double start_time,
    double end_time) const {
    return compute_range_flow(*this, conductive_couplings(), node_nums_1,
                              node_nums_2, start_time, end_time,
                              "Conductive coupling", conductive_pair_flow);
}

Eigen::MatrixXd DataModel::flow_radiative(Index node_num_1,
                                          Index node_num_2) const {
    return flow_radiative(std::vector<Index>{node_num_1},
                          std::vector<Index>{node_num_2});
}

Eigen::MatrixXd DataModel::flow_radiative(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2) const {
    return compute_single_time_flow(*this, radiative_couplings(), node_nums_1,
                                    node_nums_2, 0, "Radiative coupling",
                                    radiative_pair_flow);
}

Eigen::MatrixXd DataModel::flow_radiative(Index node_num_1, Index node_num_2,
                                          double time) const {
    return flow_radiative(std::vector<Index>{node_num_1},
                          std::vector<Index>{node_num_2}, time);
}

Eigen::MatrixXd DataModel::flow_radiative(const std::vector<Index>& node_nums_1,
                                          const std::vector<Index>& node_nums_2,
                                          double time) const {
    validate_flow_inputs(*this, radiative_couplings(), "Radiative coupling");
    return compute_single_time_flow(*this, radiative_couplings(), node_nums_1,
                                    node_nums_2,
                                    find_floor_time_index(T().times(), time),
                                    "Radiative coupling", radiative_pair_flow);
}

Eigen::MatrixXd DataModel::flow_radiative(Index node_num_1, Index node_num_2,
                                          double start_time,
                                          double end_time) const {
    return flow_radiative(std::vector<Index>{node_num_1},
                          std::vector<Index>{node_num_2}, start_time, end_time);
}

Eigen::MatrixXd DataModel::flow_radiative(const std::vector<Index>& node_nums_1,
                                          const std::vector<Index>& node_nums_2,
                                          double start_time,
                                          double end_time) const {
    return compute_range_flow(*this, radiative_couplings(), node_nums_1,
                              node_nums_2, start_time, end_time,
                              "Radiative coupling", radiative_pair_flow);
}

}  // namespace pycanha
