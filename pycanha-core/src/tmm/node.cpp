
#include "pycanha-core/tmm/node.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/nodes.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

Node::Node(int node_num)
    : _node_num(node_num),
      _local_storage_ptr(std::make_unique<LocalStorage>()) {}

Node::Node(int node_num, const std::weak_ptr<Nodes>& parent_pointer)
    : _node_num(node_num), _parent_pointer(parent_pointer) {}

double Node::resolve_get_double(double (Nodes::*nodes_getter)(int),
                                double LocalStorage::*local_member) {
    if (auto parent_nodes = _parent_pointer.lock()) {
        Nodes& nodes_ref = *parent_nodes;
        const double temp = (nodes_ref.*nodes_getter)(_node_num);
        if (std::isnan(temp)) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Attribute unavailable. "
                          << "Probably the node was deleted. "
                          << "The node is now an unvalid container.\n";
            }
        }
        return temp;
    }

    if (_local_storage_ptr != nullptr) {
        return (*_local_storage_ptr).*local_member;
    }

    std::cout << "WARNING: The node is an unvalid container. "
              << "Create a new one to have a valid node again.\n";
    return std::nan("");
}

void Node::resolve_set_double(bool (Nodes::*nodes_setter)(int, double),
                              double LocalStorage::*local_member,
                              double value) {
    if (auto parent_nodes = _parent_pointer.lock()) {
        Nodes& nodes_ref = *parent_nodes;
        if (!(nodes_ref.*nodes_setter)(_node_num, value)) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Cannot set attribute. "
                          << "Probably the node was deleted from TNs. "
                          << "The node is now an unvalid container.\n";
            }
        }
        return;
    }

    if (_local_storage_ptr) {
        (*_local_storage_ptr).*local_member = value;
        return;
    }

    if (VERBOSE) {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
    }
}

// Move constructor
Node::Node(Node&& other_node) noexcept
    : _node_num(other_node._node_num),
      _parent_pointer(std::move(other_node._parent_pointer)),
      _local_storage_ptr(std::move(other_node._local_storage_ptr)) {
    if (DEBUG) {
        std::cout << "Move constructor called \n";
    }
}

// Copy constructor
Node::Node(const Node& other_node)
    : _node_num(other_node._node_num),
      _parent_pointer(other_node._parent_pointer) {
    if (other_node._local_storage_ptr) {
        _local_storage_ptr =
            std::make_unique<LocalStorage>(*other_node._local_storage_ptr);
    }

    if (DEBUG) {
        std::cout << "Copy constructor called \n";
    }
}

// Assignment operator
Node& Node::operator=(const Node& other_node) {
    // Check for self-assignment
    if (this == &other_node) {
        return *this;
    }

    _node_num = other_node._node_num;
    _parent_pointer = other_node._parent_pointer;

    if (other_node._local_storage_ptr) {
        _local_storage_ptr =
            std::make_unique<LocalStorage>(*other_node._local_storage_ptr);
    } else {
        _local_storage_ptr.reset();
    }

    return *this;
}

// Move assignment operator
Node& Node::operator=(Node&& other_node) noexcept {
    if (this != &other_node) {
        // Transfer ownership of resources from other_node to this object
        _node_num = other_node._node_num;
        _parent_pointer = std::move(other_node._parent_pointer);
        _local_storage_ptr = std::move(other_node._local_storage_ptr);

        if (DEBUG) {
            std::cout << "Move assignment operator called \n";
        }
    }
    return *this;
}

Node::~Node() = default;

double Node::get_T() {
    return resolve_get_double(&Nodes::get_T, &LocalStorage::T);
}

double Node::get_C() {
    return resolve_get_double(&Nodes::get_C, &LocalStorage::C);
}

double Node::get_qs() {
    return resolve_get_double(&Nodes::get_qs, &LocalStorage::qs);
}

double Node::get_qa() {
    return resolve_get_double(&Nodes::get_qa, &LocalStorage::qa);
}

double Node::get_qe() {
    return resolve_get_double(&Nodes::get_qe, &LocalStorage::qe);
}

double Node::get_qi() {
    return resolve_get_double(&Nodes::get_qi, &LocalStorage::qi);
}

double Node::get_qr() {
    return resolve_get_double(&Nodes::get_qr, &LocalStorage::qr);
}

double Node::get_a() {
    return resolve_get_double(&Nodes::get_a, &LocalStorage::a);
}

double Node::get_fx() {
    return resolve_get_double(&Nodes::get_fx, &LocalStorage::fx);
}

double Node::get_fy() {
    return resolve_get_double(&Nodes::get_fy, &LocalStorage::fy);
}

double Node::get_fz() {
    return resolve_get_double(&Nodes::get_fz, &LocalStorage::fz);
}

double Node::get_eps() {
    return resolve_get_double(&Nodes::get_eps, &LocalStorage::eps);
}

double Node::get_aph() {
    return resolve_get_double(&Nodes::get_aph, &LocalStorage::aph);
}

void Node::set_T(double value) {
    resolve_set_double(&Nodes::set_T, &LocalStorage::T, value);
}

void Node::set_C(double value) {
    resolve_set_double(&Nodes::set_C, &LocalStorage::C, value);
}

void Node::set_qs(double value) {
    resolve_set_double(&Nodes::set_qs, &LocalStorage::qs, value);
}

void Node::set_qa(double value) {
    resolve_set_double(&Nodes::set_qa, &LocalStorage::qa, value);
}

void Node::set_qe(double value) {
    resolve_set_double(&Nodes::set_qe, &LocalStorage::qe, value);
}

void Node::set_qi(double value) {
    resolve_set_double(&Nodes::set_qi, &LocalStorage::qi, value);
}

void Node::set_qr(double value) {
    resolve_set_double(&Nodes::set_qr, &LocalStorage::qr, value);
}

void Node::set_a(double value) {
    resolve_set_double(&Nodes::set_a, &LocalStorage::a, value);
}

void Node::set_fx(double value) {
    resolve_set_double(&Nodes::set_fx, &LocalStorage::fx, value);
}

void Node::set_fy(double value) {
    resolve_set_double(&Nodes::set_fy, &LocalStorage::fy, value);
}

void Node::set_fz(double value) {
    resolve_set_double(&Nodes::set_fz, &LocalStorage::fz, value);
}

void Node::set_eps(double value) {
    resolve_set_double(&Nodes::set_eps, &LocalStorage::eps, value);
}

void Node::set_aph(double value) {
    resolve_set_double(&Nodes::set_aph, &LocalStorage::aph, value);
}

char Node::get_type() {
    if (auto parent_nodes = _parent_pointer.lock()) {
        const char temp = parent_nodes->get_type(_node_num);
        if (temp == static_cast<char>(0)) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Attribute unavailable. "
                          << "Probably the node was deleted. "
                          << "The node is now an unvalid container.\n";
            }
        }
        return temp;
    } else if (_local_storage_ptr != nullptr) {
        return _local_storage_ptr->type;
    } else {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
        return static_cast<char>(0);
    }
}

void Node::set_type(char type) {
    if (auto parent_nodes = _parent_pointer.lock()) {
        if (!(parent_nodes->set_type(_node_num, type))) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Cannot set attribute. "
                          << "Probably the node was deleted from TNs. "
                          << "The node is now an unvalid container.\n";
            }
        }
    } else if (_local_storage_ptr != nullptr) {
        _local_storage_ptr->type = type;
    } else {
        if (VERBOSE) {
            std::cout << "WARNING: The node is an unvalid container. "
                      << "Create a new one to have a valid node again.\n";
        }
    }
}

std::string Node::get_literal_C() const {
    if (auto parent_nodes = _parent_pointer.lock()) {
        return parent_nodes->get_literal_C(_node_num);
    } else if (_local_storage_ptr != nullptr) {
        return _local_storage_ptr->literal_C;
    } else {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
        return {};
    }
}

void Node::set_literal_C(const std::string& str) {
    if (auto parent_nodes = _parent_pointer.lock()) {
        if (!(parent_nodes->set_literal_C(_node_num, str))) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Cannot set attribute. "
                          << "Probably the node was deleted from TNs. "
                          << "The node is now an unvalid container.\n";
            }
        }
    } else if (_local_storage_ptr != nullptr) {
        _local_storage_ptr->literal_C = str;
    } else {
        if (VERBOSE) {
            std::cout << "WARNING: The node is an unvalid container. "
                      << "Create a new one to have a valid node again.\n";
        }
    }
}
// TODO: Put in the macro?

void Node::set_node_num(int node_num) { this->_node_num = node_num; }

int Node::get_node_num() const { return _node_num; }

int Node::get_int_node_num() {
    if (auto parent_nodes = _parent_pointer.lock()) {
        const Index temp = parent_nodes->get_idx_from_node_num(_node_num);
        if (temp < 0) {
            if (DEBUG) {
                std::cout << "WARNING: Attribute unavailable. "
                          << "Probably the node was deleted. "
                          << "The node is now unassociated from TNs.\n";
            }
            _parent_pointer.reset();
        }
        return static_cast<int>(temp);
    } else {
        if (DEBUG) {
            std::cout << "WARNING: Node is not associated to any TNs. "
                      << "IntNodeNum is undefined. Returning -1.\n";
        }
        return -1;
    }
}

std::weak_ptr<Nodes> Node::get_parent_pointer() { return _parent_pointer; }

uint64_t Node::get_int_parent_pointer() {
    auto parent_nodes = _parent_pointer.lock();
    if (!parent_nodes) {
        return 0U;
    }

    auto* const raw_pointer = parent_nodes.get();
    const auto address = std::bit_cast<std::uintptr_t>(raw_pointer);
    return static_cast<uint64_t>(address);
}

void Node::set_thermal_nodes_parent(
    std::weak_ptr<Nodes>&& thermal_nodes_parent_ptr) {
    _parent_pointer = std::move(thermal_nodes_parent_ptr);
    local_storage_destructor();
}

void Node::local_storage_destructor() { _local_storage_ptr.reset(); }
