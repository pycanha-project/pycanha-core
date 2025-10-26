#include "pycanha-core/tmm/conductivecouplings.hpp"

#include <memory>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

ConductiveCouplings::ConductiveCouplings(std::shared_ptr<Nodes> nodes) noexcept
    : _couplings(std::move(nodes)) {}

void ConductiveCouplings::add_coupling(Index node_num_1, Index node_num_2,
                                       double value) {
    _couplings.add_ovw_coupling(node_num_1, node_num_2, value);
}

void ConductiveCouplings::add_coupling(const Coupling& coupling) {
    _couplings.add_ovw_coupling(coupling);
}

void ConductiveCouplings::set_coupling_value(Index node_num_1, Index node_num_2,
                                             double value) {
    _couplings.set_coupling_value(node_num_1, node_num_2, value);
}

double ConductiveCouplings::get_coupling_value(Index node_num_1,
                                               Index node_num_2) {
    return _couplings.get_coupling_value(node_num_1, node_num_2);
}

}  // namespace pycanha
