#include "pycanha-core/thermaldata/data_model.hpp"

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"

namespace pycanha {
namespace {

constexpr std::array<DataModelAttribute, 16> k_all_attributes = {
    DataModelAttribute::T,   DataModelAttribute::C,  DataModelAttribute::QS,
    DataModelAttribute::QA,  DataModelAttribute::QE, DataModelAttribute::QI,
    DataModelAttribute::QR,  DataModelAttribute::A,  DataModelAttribute::APH,
    DataModelAttribute::EPS, DataModelAttribute::FX, DataModelAttribute::FY,
    DataModelAttribute::FZ,  DataModelAttribute::KL, DataModelAttribute::KR,
    DataModelAttribute::JAC,
};

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

}  // namespace pycanha
