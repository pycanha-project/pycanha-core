#include "pycanha-core/tmm/thermalnetwork.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

[[nodiscard]] NodeNum to_node_num(Index node_num) {
    if (node_num > static_cast<Index>(std::numeric_limits<NodeNum>::max()) ||
        node_num < static_cast<Index>(std::numeric_limits<NodeNum>::min())) {
        throw std::out_of_range(
            "ThermalNetwork node number exceeds NodeNum range");
    }
    return static_cast<NodeNum>(node_num);
}

template <typename FlowFunction>
[[nodiscard]] double sum_flow_over_node_groups(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2, const FlowFunction& flow_function) {
    double total_flow = 0.0;

    for (const Index node_num_1 : node_nums_1) {
        for (const Index node_num_2 : node_nums_2) {
            const double pair_flow = flow_function(node_num_1, node_num_2);
            if (std::isnan(pair_flow)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            total_flow += pair_flow;
        }
    }

    return total_flow;
}

}  // namespace

ThermalNetwork::ThermalNetwork()
    : _nodes(std::make_shared<Nodes>()),
      _conductive_couplings(std::make_shared<ConductiveCouplings>(_nodes)),
      _radiative_couplings(std::make_shared<RadiativeCouplings>(_nodes)) {
    SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                        "ThermalNetwork: default constructor");
}

ThermalNetwork::ThermalNetwork(std::shared_ptr<Nodes> nodes,
                               std::shared_ptr<ConductiveCouplings> conductive,
                               std::shared_ptr<RadiativeCouplings> radiative)
    : _nodes(std::move(nodes)),
      _conductive_couplings(std::move(conductive)),
      _radiative_couplings(std::move(radiative)) {
    if (_nodes == nullptr) {
        _nodes = std::make_shared<Nodes>();
    }

    if (_conductive_couplings == nullptr) {
        _conductive_couplings = std::make_shared<ConductiveCouplings>(_nodes);
    }

    if (_radiative_couplings == nullptr) {
        _radiative_couplings = std::make_shared<RadiativeCouplings>(_nodes);
    }

    SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                        "ThermalNetwork: constructor with shared resources");
}

void ThermalNetwork::add_node(Node& node) {
    if (_nodes == nullptr || _conductive_couplings == nullptr ||
        _radiative_couplings == nullptr) {
        return;
    }

    const char type = node.get_type();
    const int user_node_num = node.get_node_num();

    if (_nodes->is_node(user_node_num)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "ThermalNetwork: node {} already exists.",
                           user_node_num);
        return;
    }

    Index insert_idx = 0;
    Index total_insert_idx = 0;

    auto& conductive_storage =
        _conductive_couplings->_couplings.get_coupling_matrices();
    auto& radiative_storage =
        _radiative_couplings->_couplings.get_coupling_matrices();

    if (type == 'D') {
        auto& diff_nodes = _nodes->_diff_node_num_vector;
        const auto it = std::upper_bound(diff_nodes.begin(), diff_nodes.end(),
                                         user_node_num);
        insert_idx = to_idx(std::distance(diff_nodes.begin(), it));

        conductive_storage._add_node_diff(insert_idx);
        radiative_storage._add_node_diff(insert_idx);

        total_insert_idx = insert_idx;
    } else if (type == 'B') {
        auto& bound_nodes = _nodes->_bound_node_num_vector;
        const auto diff_count = to_idx(_nodes->_diff_node_num_vector.size());
        const auto it = std::upper_bound(bound_nodes.begin(), bound_nodes.end(),
                                         user_node_num);
        insert_idx = to_idx(std::distance(bound_nodes.begin(), it));

        conductive_storage._add_node_bound(insert_idx);
        radiative_storage._add_node_bound(insert_idx);

        total_insert_idx = diff_count + insert_idx;
    } else {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "ThermalNetwork: wrong node type for {}",
                           user_node_num);
        return;
    }

    _nodes->add_node_insert_idx(node, total_insert_idx);
}

void ThermalNetwork::remove_node(Index node_num) {
    if (_nodes == nullptr || _conductive_couplings == nullptr ||
        _radiative_couplings == nullptr) {
        return;
    }

    const NodeNum inttype_node_num = to_node_num(node_num);
    const auto idx = _nodes->get_idx_from_node_num(inttype_node_num);

    if (!idx.has_value()) {
        return;
    }

    const auto diff_count = to_idx(_nodes->_diff_node_num_vector.size());

    auto& conductive_storage =
        _conductive_couplings->_couplings.get_coupling_matrices();
    auto& radiative_storage =
        _radiative_couplings->_couplings.get_coupling_matrices();

    if (*idx < diff_count) {
        conductive_storage._remove_node_diff(*idx);
        radiative_storage._remove_node_diff(*idx);
    } else {
        const Index boundary_idx = *idx - diff_count;
        conductive_storage._remove_node_bound(boundary_idx);
        radiative_storage._remove_node_bound(boundary_idx);
    }

    _nodes->remove_node(inttype_node_num);
}

Nodes& ThermalNetwork::nodes() noexcept { return *_nodes; }

const Nodes& ThermalNetwork::nodes() const noexcept { return *_nodes; }

CouplingMatrices& ThermalNetwork::conductive_matrices() noexcept {
    return _conductive_couplings->matrices();
}

const CouplingMatrices& ThermalNetwork::conductive_matrices() const noexcept {
    return _conductive_couplings->matrices();
}

CouplingMatrices& ThermalNetwork::radiative_matrices() noexcept {
    return _radiative_couplings->matrices();
}

const CouplingMatrices& ThermalNetwork::radiative_matrices() const noexcept {
    return _radiative_couplings->matrices();
}

ConductiveCouplings& ThermalNetwork::conductive_couplings() noexcept {
    return *_conductive_couplings;
}

const ConductiveCouplings& ThermalNetwork::conductive_couplings()
    const noexcept {
    return *_conductive_couplings;
}

RadiativeCouplings& ThermalNetwork::radiative_couplings() noexcept {
    return *_radiative_couplings;
}

const RadiativeCouplings& ThermalNetwork::radiative_couplings() const noexcept {
    return *_radiative_couplings;
}

double ThermalNetwork::flow_conductive(Index node_num_1, Index node_num_2) {
    if (node_num_1 == node_num_2) {
        return 0.0;
    }

    const NodeNum inttype_node_num_1 = to_node_num(node_num_1);
    const NodeNum inttype_node_num_2 = to_node_num(node_num_2);

    if (!_nodes->is_node(inttype_node_num_1) ||
        !_nodes->is_node(inttype_node_num_2)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "ThermalNetwork: invalid node numbers {}, {}",
                           node_num_1, node_num_2);
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double temperature_1 = _nodes->get_T(inttype_node_num_1);
    const double temperature_2 = _nodes->get_T(inttype_node_num_2);
    const double conductance =
        _conductive_couplings->get_coupling_value(node_num_1, node_num_2);

    return conductance * (temperature_2 - temperature_1);
}

double ThermalNetwork::flow_conductive(const std::vector<Index>& node_nums_1,
                                       const std::vector<Index>& node_nums_2) {
    return sum_flow_over_node_groups(
        node_nums_1, node_nums_2, [this](Index node_num_1, Index node_num_2) {
            return flow_conductive(node_num_1, node_num_2);
        });
}

double ThermalNetwork::flow_radiative(Index node_num_1, Index node_num_2) {
    if (node_num_1 == node_num_2) {
        return 0.0;
    }

    const NodeNum inttype_node_num_1 = to_node_num(node_num_1);
    const NodeNum inttype_node_num_2 = to_node_num(node_num_2);

    if (!_nodes->is_node(inttype_node_num_1) ||
        !_nodes->is_node(inttype_node_num_2)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "ThermalNetwork: invalid node numbers {}, {}",
                           node_num_1, node_num_2);
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double temperature_1 = _nodes->get_T(inttype_node_num_1);
    const double temperature_2 = _nodes->get_T(inttype_node_num_2);
    const double radiative_conductance =
        _radiative_couplings->get_coupling_value(node_num_1, node_num_2);

    return radiative_conductance * STF_BOLTZ *
           (std::pow(temperature_2, 4) - std::pow(temperature_1, 4));
}

double ThermalNetwork::flow_radiative(const std::vector<Index>& node_nums_1,
                                      const std::vector<Index>& node_nums_2) {
    return sum_flow_over_node_groups(
        node_nums_1, node_nums_2, [this](Index node_num_1, Index node_num_2) {
            return flow_radiative(node_num_1, node_num_2);
        });
}

std::shared_ptr<Nodes> ThermalNetwork::nodes_ptr() noexcept { return _nodes; }

std::shared_ptr<const Nodes> ThermalNetwork::nodes_ptr() const noexcept {
    return _nodes;
}

}  // namespace pycanha
