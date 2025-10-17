#include "pycanha-core/tmm/radiativecouplings.hpp"

#include <memory>
#include <utility>

#include "pycanha-core/parameters.hpp"
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

}  // namespace pycanha
