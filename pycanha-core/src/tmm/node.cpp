
#include "pycanha-core/tmm/node.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/config.hpp"
#include "pycanha-core/tmm/nodes.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

Node::Node(int node_num)
    : _node_num(node_num),
      _local_storage_ptr(std::make_unique<LocalStorage>()) {}

Node::Node(int node_num, const std::weak_ptr<Nodes>& parent_pointer)
    : _node_num(node_num), _parent_pointer(parent_pointer) {}

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

/*
Getters and setters are always the same except which atributte is needed.
A macro is used to get/set most of them.
*/

#define GET_SET_DOUBLE_ATTR(attr)                                              \
    double Node::get_##attr() {                                                \
        if (auto ptr_TNs = _parent_pointer.lock()) {                           \
            const double temp = ptr_TNs->get_##attr(_node_num);                \
            if (std::isnan(temp)) {                                            \
                _parent_pointer.reset();                                       \
                if (VERBOSE) {                                                 \
                    std::cout << "WARNING: Attribute unavailable. "            \
                              << "Probably the node was deleted. "             \
                              << "The node is now an unvalid container.\n";    \
                }                                                              \
            }                                                                  \
            return temp;                                                       \
        } else if (_local_storage_ptr != nullptr) {                            \
            return _local_storage_ptr->attr;                                   \
        } else {                                                               \
            std::cout << "WARNING: The node is an unvalid container. "         \
                      << "Create a new one to have a valid node again.\n";     \
            return std::nan("");                                               \
        }                                                                      \
    }                                                                          \
    void Node::set_##attr(double value) {                                      \
        if (auto ptr_TNs = _parent_pointer.lock()) {                           \
            if (!(ptr_TNs->set_##attr(_node_num, value))) {                    \
                _parent_pointer.reset();                                       \
                if (VERBOSE) {                                                 \
                    std::cout << "WARNING: Cannot set attribute. "             \
                              << "Probably the node was deleted from TNs. "    \
                              << "The node is now an unvalid container.\n";    \
                }                                                              \
            }                                                                  \
        } else if (_local_storage_ptr) {                                       \
            _local_storage_ptr->attr = value;                                  \
        } else {                                                               \
            if (VERBOSE) {                                                     \
                std::cout << "WARNING: The node is an unvalid container. "     \
                          << "Create a new one to have a valid node again.\n"; \
            }                                                                  \
        }                                                                      \
    }

// GETSET(int, node_num)
GET_SET_DOUBLE_ATTR(T)
GET_SET_DOUBLE_ATTR(C)
GET_SET_DOUBLE_ATTR(qs)
GET_SET_DOUBLE_ATTR(qa)
GET_SET_DOUBLE_ATTR(qe)
GET_SET_DOUBLE_ATTR(qi)
GET_SET_DOUBLE_ATTR(qr)
GET_SET_DOUBLE_ATTR(a)
GET_SET_DOUBLE_ATTR(fx)
GET_SET_DOUBLE_ATTR(fy)
GET_SET_DOUBLE_ATTR(fz)
GET_SET_DOUBLE_ATTR(eps)
GET_SET_DOUBLE_ATTR(aph)

char Node::get_type() {
    if (auto ptr_TNs = _parent_pointer.lock()) {
        const char temp = ptr_TNs->get_type(_node_num);
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
    if (auto ptr_TNs = _parent_pointer.lock()) {
        if (!(ptr_TNs->set_type(_node_num, type))) {
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
    if (auto ptr_TNs = _parent_pointer.lock()) {
        return ptr_TNs->get_literal_C(_node_num);
    } else if (_local_storage_ptr != nullptr) {
        return _local_storage_ptr->literal_C;
    } else {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
        return std::string();
    }
}

void Node::set_literal_C(const std::string& str) {
    if (auto ptr_TNs = _parent_pointer.lock()) {
        if (!(ptr_TNs->set_literal_C(_node_num, str))) {
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
    if (auto PtrTNs = _parent_pointer.lock()) {
        const Index temp = PtrTNs->get_idx_from_node_num(_node_num);
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
    auto PtrTNs = _parent_pointer.lock();

    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(PtrTNs.get()));
}

void Node::set_thermal_nodes_parent(
    std::weak_ptr<Nodes>&& thermal_nodes_parent_ptr) {
    _parent_pointer = std::move(thermal_nodes_parent_ptr);
    local_storage_destructor();
}

void Node::local_storage_destructor() { _local_storage_ptr.reset(); }
