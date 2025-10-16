#include "pycanha-core/tmm/thermalnetwork.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"

namespace pycanha {

namespace {

[[nodiscard]] bool is_node_number_in_range(Index node_num) {
    return node_num <= static_cast<Index>(std::numeric_limits<int>::max()) &&
           node_num >= static_cast<Index>(std::numeric_limits<int>::min());
}

}  // namespace

ThermalNetwork::ThermalNetwork()
    : _nodes(std::make_shared<Nodes>()),
      _conductive_couplings(std::make_shared<ConductiveCouplings>(_nodes)),
      _radiative_couplings(std::make_shared<RadiativeCouplings>(_nodes)) {
    if (DEBUG) {
        std::cout << "ThermalNetwork: default constructor" << '\n';
    }
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

    if (DEBUG) {
        std::cout << "ThermalNetwork: constructor with shared resources"
                  << '\n';
    }
}

void ThermalNetwork::add_node(Node& node) {
    if (_nodes == nullptr || _conductive_couplings == nullptr ||
        _radiative_couplings == nullptr) {
        return;
    }

    const char type = node.get_type();
    const int user_node_num = node.get_node_num();

    if (_nodes->is_node(user_node_num)) {
        if (VERBOSE) {
            std::cout << "ThermalNetwork: node " << user_node_num
                      << " already exists." << '\n';
        }
        return;
    }

    Index insert_idx = 0;
    Index total_insert_idx = 0;

    auto& conductive_matrices =
        _conductive_couplings->_couplings.get_coupling_matrices();
    auto& radiative_matrices =
        _radiative_couplings->_couplings.get_coupling_matrices();

    if (type == 'D') {
        auto& diff_nodes = _nodes->_diff_node_num_vector;
        const auto it = std::upper_bound(diff_nodes.begin(), diff_nodes.end(),
                                         user_node_num);
        insert_idx = static_cast<Index>(std::distance(diff_nodes.begin(), it));

        conductive_matrices._add_node_diff(insert_idx);
        radiative_matrices._add_node_diff(insert_idx);

        total_insert_idx = insert_idx;
    } else if (type == 'B') {
        auto& bound_nodes = _nodes->_bound_node_num_vector;
        const auto diff_count =
            static_cast<Index>(_nodes->_diff_node_num_vector.size());
        const auto it = std::upper_bound(bound_nodes.begin(), bound_nodes.end(),
                                         user_node_num);
        insert_idx = static_cast<Index>(std::distance(bound_nodes.begin(), it));

        conductive_matrices._add_node_bound(insert_idx);
        radiative_matrices._add_node_bound(insert_idx);

        total_insert_idx = diff_count + insert_idx;
    } else {
        if (VERBOSE) {
            std::cout << "ThermalNetwork: wrong node type for " << user_node_num
                      << '\n';
        }
        return;
    }

    _nodes->add_node_insert_idx(node, total_insert_idx);
}

void ThermalNetwork::remove_node(Index node_num) {
    if (_nodes == nullptr || _conductive_couplings == nullptr ||
        _radiative_couplings == nullptr) {
        return;
    }

    if (!is_node_number_in_range(node_num)) {
        if (VERBOSE) {
            std::cout << "ThermalNetwork: node number out of range " << node_num
                      << '\n';
        }
        return;
    }

    const int user_node_num = static_cast<int>(node_num);
    const Index idx = _nodes->get_idx_from_node_num(user_node_num);

    if (idx < 0) {
        return;
    }

    const auto diff_count =
        static_cast<Index>(_nodes->_diff_node_num_vector.size());

    auto& conductive_matrices =
        _conductive_couplings->_couplings.get_coupling_matrices();
    auto& radiative_matrices =
        _radiative_couplings->_couplings.get_coupling_matrices();

    if (idx < diff_count) {
        conductive_matrices._remove_node_diff(idx);
        radiative_matrices._remove_node_diff(idx);
    } else {
        const Index boundary_idx = idx - diff_count;
        conductive_matrices._remove_node_bound(boundary_idx);
        radiative_matrices._remove_node_bound(boundary_idx);
    }

    _nodes->remove_node(user_node_num);
}

Nodes& ThermalNetwork::nodes() noexcept { return *_nodes; }

const Nodes& ThermalNetwork::nodes() const noexcept { return *_nodes; }

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

std::shared_ptr<Nodes> ThermalNetwork::nodes_ptr() noexcept { return _nodes; }

std::shared_ptr<const Nodes> ThermalNetwork::nodes_ptr() const noexcept {
    return _nodes;
}

}  // namespace pycanha
