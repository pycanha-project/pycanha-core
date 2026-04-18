#pragma once

#include <cstdint>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"

namespace pycanha {

enum class DataModelAttribute : std::uint8_t {
    T = 0,
    C = 1,
    QS = 2,
    QA = 3,
    QE = 4,
    QI = 5,
    QR = 6,
    A = 7,
    APH = 8,
    EPS = 9,
    FX = 10,
    FY = 11,
    FZ = 12,
    KL = 13,
    KR = 14,
    JAC = 15,
};

class DataModel {
  public:
    DataModel() = default;
    explicit DataModel(std::vector<Index> node_numbers);

    [[nodiscard]] DenseTimeSeries& T() noexcept;
    [[nodiscard]] const DenseTimeSeries& T() const noexcept;
    [[nodiscard]] DenseTimeSeries& C() noexcept;
    [[nodiscard]] const DenseTimeSeries& C() const noexcept;
    [[nodiscard]] DenseTimeSeries& QS() noexcept;
    [[nodiscard]] const DenseTimeSeries& QS() const noexcept;
    [[nodiscard]] DenseTimeSeries& QA() noexcept;
    [[nodiscard]] const DenseTimeSeries& QA() const noexcept;
    [[nodiscard]] DenseTimeSeries& QE() noexcept;
    [[nodiscard]] const DenseTimeSeries& QE() const noexcept;
    [[nodiscard]] DenseTimeSeries& QI() noexcept;
    [[nodiscard]] const DenseTimeSeries& QI() const noexcept;
    [[nodiscard]] DenseTimeSeries& QR() noexcept;
    [[nodiscard]] const DenseTimeSeries& QR() const noexcept;
    [[nodiscard]] DenseTimeSeries& A() noexcept;
    [[nodiscard]] const DenseTimeSeries& A() const noexcept;
    [[nodiscard]] DenseTimeSeries& APH() noexcept;
    [[nodiscard]] const DenseTimeSeries& APH() const noexcept;
    [[nodiscard]] DenseTimeSeries& EPS() noexcept;
    [[nodiscard]] const DenseTimeSeries& EPS() const noexcept;
    [[nodiscard]] DenseTimeSeries& FX() noexcept;
    [[nodiscard]] const DenseTimeSeries& FX() const noexcept;
    [[nodiscard]] DenseTimeSeries& FY() noexcept;
    [[nodiscard]] const DenseTimeSeries& FY() const noexcept;
    [[nodiscard]] DenseTimeSeries& FZ() noexcept;
    [[nodiscard]] const DenseTimeSeries& FZ() const noexcept;

    [[nodiscard]] SparseTimeSeries& conductive_couplings() noexcept;
    [[nodiscard]] const SparseTimeSeries& conductive_couplings() const noexcept;
    [[nodiscard]] SparseTimeSeries& radiative_couplings() noexcept;
    [[nodiscard]] const SparseTimeSeries& radiative_couplings() const noexcept;
    [[nodiscard]] DenseMatrixTimeSeries& jacobian() noexcept;
    [[nodiscard]] const DenseMatrixTimeSeries& jacobian() const noexcept;

    [[nodiscard]] DenseTimeSeries& get_dense_attribute(DataModelAttribute attr);
    [[nodiscard]] const DenseTimeSeries& get_dense_attribute(
        DataModelAttribute attr) const;

    [[nodiscard]] SparseTimeSeries& get_sparse_attribute(
        DataModelAttribute attr);
    [[nodiscard]] const SparseTimeSeries& get_sparse_attribute(
        DataModelAttribute attr) const;

    [[nodiscard]] DenseMatrixTimeSeries& get_matrix_attribute(
        DataModelAttribute attr);
    [[nodiscard]] const DenseMatrixTimeSeries& get_matrix_attribute(
        DataModelAttribute attr) const;

    void set_node_numbers(std::vector<Index> node_numbers);
    [[nodiscard]] std::vector<Index>& node_numbers() noexcept;
    [[nodiscard]] const std::vector<Index>& node_numbers() const noexcept;

    [[nodiscard]] std::vector<DataModelAttribute> populated_attributes() const;

    [[nodiscard]] Eigen::MatrixXd flow_conductive(Index node_num_1,
                                                  Index node_num_2) const;
    [[nodiscard]] Eigen::MatrixXd flow_conductive(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2) const;
    [[nodiscard]] Eigen::MatrixXd flow_conductive(Index node_num_1,
                                                  Index node_num_2,
                                                  double time) const;
    [[nodiscard]] Eigen::MatrixXd flow_conductive(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2, double time) const;
    [[nodiscard]] Eigen::MatrixXd flow_conductive(Index node_num_1,
                                                  Index node_num_2,
                                                  double start_time,
                                                  double end_time) const;
    [[nodiscard]] Eigen::MatrixXd flow_conductive(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2, double start_time,
        double end_time) const;

    [[nodiscard]] Eigen::MatrixXd flow_radiative(Index node_num_1,
                                                 Index node_num_2) const;
    [[nodiscard]] Eigen::MatrixXd flow_radiative(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2) const;
    [[nodiscard]] Eigen::MatrixXd flow_radiative(Index node_num_1,
                                                 Index node_num_2,
                                                 double time) const;
    [[nodiscard]] Eigen::MatrixXd flow_radiative(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2, double time) const;
    [[nodiscard]] Eigen::MatrixXd flow_radiative(Index node_num_1,
                                                 Index node_num_2,
                                                 double start_time,
                                                 double end_time) const;
    [[nodiscard]] Eigen::MatrixXd flow_radiative(
        const std::vector<Index>& node_nums_1,
        const std::vector<Index>& node_nums_2, double start_time,
        double end_time) const;

  private:
    DenseTimeSeries _T;
    DenseTimeSeries _C;
    DenseTimeSeries _QS;
    DenseTimeSeries _QA;
    DenseTimeSeries _QE;
    DenseTimeSeries _QI;
    DenseTimeSeries _QR;
    DenseTimeSeries _A;
    DenseTimeSeries _APH;
    DenseTimeSeries _EPS;
    DenseTimeSeries _FX;
    DenseTimeSeries _FY;
    DenseTimeSeries _FZ;

    SparseTimeSeries _conductive_couplings;
    SparseTimeSeries _radiative_couplings;
    DenseMatrixTimeSeries _jacobian;

    std::vector<Index> _node_numbers;
};

}  // namespace pycanha
