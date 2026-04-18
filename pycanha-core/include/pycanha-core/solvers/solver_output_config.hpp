#pragma once

#include <array>
#include <unordered_set>

#include "pycanha-core/thermaldata/data_model.hpp"

namespace pycanha {
namespace detail {

constexpr std::array<DataModelAttribute, 13> k_dense_output_attributes = {
    DataModelAttribute::T,   DataModelAttribute::C,  DataModelAttribute::QS,
    DataModelAttribute::QA,  DataModelAttribute::QE, DataModelAttribute::QI,
    DataModelAttribute::QR,  DataModelAttribute::A,  DataModelAttribute::APH,
    DataModelAttribute::EPS, DataModelAttribute::FX, DataModelAttribute::FY,
    DataModelAttribute::FZ,
};

}  // namespace detail

struct SolverOutputConfig {
    std::unordered_set<DataModelAttribute> attributes = {DataModelAttribute::T};

    void output_all_dense() {
        attributes.clear();
        attributes.insert(detail::k_dense_output_attributes.begin(),
                          detail::k_dense_output_attributes.end());
    }

    void output_all() {
        output_all_dense();
        attributes.insert(DataModelAttribute::KL);
        attributes.insert(DataModelAttribute::KR);
        attributes.insert(DataModelAttribute::JAC);
    }

    void add(DataModelAttribute attr) { attributes.insert(attr); }

    void remove(DataModelAttribute attr) { attributes.erase(attr); }

    [[nodiscard]] bool has(DataModelAttribute attr) const noexcept {
        return attributes.contains(attr);
    }
};

}  // namespace pycanha
