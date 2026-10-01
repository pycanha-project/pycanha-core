#include "pycanha-core/tmm/radiativecouplings.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

RadiativeCouplings::RadiativeCouplings(std::shared_ptr<Nodes> nodes) noexcept
    : _couplings(std::move(nodes)) {}

void RadiativeCouplings::add_coupling(Index node_num_1, Index node_num_2,
                                      double value) {
    _couplings.add_ovw_coupling(node_num_1, node_num_2, value);
}

void RadiativeCouplings::add_coupling(const Coupling& coupling) {
    _couplings.add_ovw_coupling(coupling);
}

void RadiativeCouplings::set_coupling_value(Index node_num_1, Index node_num_2,
                                            double value) {
    _couplings.set_coupling_value(node_num_1, node_num_2, value);
}

double RadiativeCouplings::get_coupling_value(Index node_num_1,
                                              Index node_num_2) {
    return _couplings.get_coupling_value(node_num_1, node_num_2);
}

BulkReport RadiativeCouplings::add_couplings(
    std::span<const NodeNum> node_nums_1, std::span<const NodeNum> node_nums_2,
    std::span<const double> values, CouplingMerge merge) {
    return _couplings.add_couplings(node_nums_1, node_nums_2, values, merge);
}

BulkReport RadiativeCouplings::add_couplings(
    std::span<const std::int64_t> node_nums_1,
    std::span<const std::int64_t> node_nums_2, std::span<const double> values,
    CouplingMerge merge) {
    return _couplings.add_couplings(node_nums_1, node_nums_2, values, merge);
}

BulkReport RadiativeCouplings::append_couplings(
    std::span<const CouplingChunk> chunks) {
    return _couplings.append_couplings(chunks);
}

BulkReport RadiativeCouplings::get_values(std::span<const NodeNum> node_nums_1,
                                          std::span<const NodeNum> node_nums_2,
                                          std::span<double> values) {
    return _couplings.get_values(node_nums_1, node_nums_2, values);
}

BulkReport RadiativeCouplings::set_values(std::span<const NodeNum> node_nums_1,
                                          std::span<const NodeNum> node_nums_2,
                                          std::span<const double> values) {
    return _couplings.set_values(node_nums_1, node_nums_2, values);
}

Couplings::CouplingArrays RadiativeCouplings::to_arrays() {
    return _couplings.to_arrays();
}

double* RadiativeCouplings::get_coupling_value_ref(Index node_num_1,
                                                   Index node_num_2) {
    return _couplings.get_coupling_value_ref(node_num_1, node_num_2);
}

}  // namespace pycanha
