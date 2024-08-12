
#include "pycanha-core/tmm/node.hpp"

#include <iostream>

#include "pycanha-core/config.hpp"

Node::Node(int UsrNodeNum) : UsrNodeNum(UsrNodeNum) {
    m_local_storage_ptr = new local_storage{'D', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

Node::Node(int UsrNodeNum, std::weak_ptr<Nodes> parent_pointer)
    : UsrNodeNum(UsrNodeNum),
      _parent_pointer(parent_pointer),
      m_local_storage_ptr(nullptr) {}

// Move constructor
Node::Node(Node&& otherNode)
    : UsrNodeNum(otherNode.UsrNodeNum),
      _parent_pointer(otherNode._parent_pointer),
      m_local_storage_ptr(otherNode.m_local_storage_ptr) {
    if (DEBUG) {
        std::cout << "Move constructor called \n";
    }

    // When otherNode call its destructor, the storage (if exists) its not
    // deleted.
    otherNode.m_local_storage_ptr = nullptr;
}

// Copy constructor
Node::Node(const Node& otherNode)
    : UsrNodeNum(otherNode.UsrNodeNum),
      _parent_pointer(otherNode._parent_pointer),
      m_local_storage_ptr(otherNode.m_local_storage_ptr) {
    // Instead of copyng attributes individually, call memcpy
    // memcpy(this, &otherNode, sizeof(Node)); //THIS IS EVIL DON'T DO IT.
    // PROGRAM WILL CRASH
    // TODO: I need to manually write a copy constructor because I need to
    // handle the local_storage. If I change the local storage to be an RII
    // class that manage the storage itself (it would be like std::string or
    // std::vector), then can I use the default constructors?

    if (m_local_storage_ptr) {
        // Pointer is not null and the node is local, not associated with TNs

        // Copy memory buffer containing the node info to other place
        m_local_storage_ptr = new local_storage;
        *m_local_storage_ptr = *otherNode.m_local_storage_ptr;
    }
    // else: m_local_storage_ptr is nullptr, and ParentPointer should be valid
    // (is not checked)

    if (DEBUG) {
        std::cout << "Copy constructor called \n";
    }
}

// Assignment operator
Node& Node::operator=(const Node& otherNode) {
    // First, delete the old buffer if exists
    if (m_local_storage_ptr) {
        delete m_local_storage_ptr;
    }

    // Shallow copy. Instead of copyng attributes individually, call memcpy
    // memcpy(this, &otherNode, sizeof(Node)); //DON'T!!!!

    this->UsrNodeNum = otherNode.UsrNodeNum;
    this->_parent_pointer = otherNode._parent_pointer;
    this->m_local_storage_ptr = otherNode.m_local_storage_ptr;

    if (m_local_storage_ptr) {
        // Pointer is not null and the node is local, not associated with TNs

        // Copy memory buffer containing the node info to other place
        m_local_storage_ptr = new local_storage;
        *m_local_storage_ptr = *otherNode.m_local_storage_ptr;
    }
    // else: m_local_storage_ptr is nullptr, and ParentPointer should be valid
    // (is not checked)

    return *this;
}

// Move assignment operator
Node& Node::operator=(Node&& otherNode) noexcept {
    if (this != &otherNode) {
        // Release any resources currently held by this object
        delete m_local_storage_ptr;

        // Transfer ownership of resources from otherNode to this object
        UsrNodeNum = otherNode.UsrNodeNum;
        _parent_pointer = std::move(otherNode._parent_pointer);
        m_local_storage_ptr = otherNode.m_local_storage_ptr;

        // Invalidate the resources of the otherNode
        otherNode.m_local_storage_ptr = nullptr;

        if (DEBUG) {
            std::cout << "Move assignment operator called \n";
        }
    }
    return *this;
}

Node::~Node() { _local_storage_destructor(); }

/*
Getters and setters are always the same except which atributte is needed.
A macro is used to get/set most of them.
*/

#define GET_SET_DOUBLE_ATTR(attr)                                              \
    double Node::get_##attr() {                                                \
        if (auto ptr_TNs = _parent_pointer.lock()) {                           \
            double temp = ptr_TNs->get_##attr(UsrNodeNum);                     \
            if (std::isnan(temp)) {                                            \
                _parent_pointer.reset();                                       \
                if (VERBOSE) {                                                 \
                    std::cout << "WARNING: Attribute unavailable. "            \
                              << "Probably the node was deleted. "             \
                              << "The node is now an unvalid container.\n";    \
                }                                                              \
            }                                                                  \
            return temp;                                                       \
        } else if (m_local_storage_ptr) {                                      \
            return m_local_storage_ptr->m_##attr;                              \
        } else {                                                               \
            std::cout << "WARNING: The node is an unvalid container. "         \
                      << "Create a new one to have a valid node again.\n";     \
            return std::nan("");                                               \
        }                                                                      \
    }                                                                          \
    void Node::set_##attr(double value) {                                      \
        if (auto ptr_TNs = _parent_pointer.lock()) {                           \
            if (!(ptr_TNs->set_##attr(UsrNodeNum, value))) {                   \
                _parent_pointer.reset();                                       \
                if (VERBOSE) {                                                 \
                    std::cout << "WARNING: Cannot set attribute. "             \
                              << "Probably the node was deleted from TNs. "    \
                              << "The node is now an unvalid container.\n";    \
                }                                                              \
            }                                                                  \
        } else if (m_local_storage_ptr) {                                      \
            m_local_storage_ptr->m_##attr = value;                             \
        } else {                                                               \
            if (VERBOSE) {                                                     \
                std::cout << "WARNING: The node is an unvalid container. "     \
                          << "Create a new one to have a valid node again.\n"; \
            }                                                                  \
        }                                                                      \
    }

// GETSET(int, UsrNodeNum)
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
        char temp = ptr_TNs->get_type(UsrNodeNum);
        if (temp == static_cast<char>(0)) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Attribute unavailable. "
                          << "Probably the node was deleted. "
                          << "The node is now an unvalid container.\n";
            }
        }
        return temp;
    } else if (m_local_storage_ptr) {
        return m_local_storage_ptr->m_type;
    } else {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
        return static_cast<char>(0);
    }
}

void Node::set_type(char type) {
    if (auto ptr_TNs = _parent_pointer.lock()) {
        if (!(ptr_TNs->set_type(UsrNodeNum, type))) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Cannot set attribute. "
                          << "Probably the node was deleted from TNs. "
                          << "The node is now an unvalid container.\n";
            }
        }
    } else if (m_local_storage_ptr) {
        m_local_storage_ptr->m_type = type;
    } else {
        if (VERBOSE) {
            std::cout << "WARNING: The node is an unvalid container. "
                      << "Create a new one to have a valid node again.\n";
        }
    }
}

std::string Node::get_literal_C() const {
    if (auto ptr_TNs = _parent_pointer.lock()) {
        return ptr_TNs->get_literal_C(UsrNodeNum);
    } else if (m_local_storage_ptr) {
        return m_local_storage_ptr->m_literal_C;
    } else {
        std::cout << "WARNING: The node is an unvalid container. "
                  << "Create a new one to have a valid node again.\n";
        return std::string();
    }
}

void Node::set_literal_C(std::string str) {
    if (auto ptr_TNs = _parent_pointer.lock()) {
        if (!(ptr_TNs->set_literal_C(UsrNodeNum, str))) {
            _parent_pointer.reset();
            if (VERBOSE) {
                std::cout << "WARNING: Cannot set attribute. "
                          << "Probably the node was deleted from TNs. "
                          << "The node is now an unvalid container.\n";
            }
        }
    } else if (m_local_storage_ptr) {
        m_local_storage_ptr->m_literal_C = str;
    } else {
        if (VERBOSE) {
            std::cout << "WARNING: The node is an unvalid container. "
                      << "Create a new one to have a valid node again.\n";
        }
    }
}
// TODO: Put in the macro?

void Node::setUsrNodeNum(int UsrNodeNum) { this->UsrNodeNum = UsrNodeNum; }

int Node::getUsrNodeNum() { return UsrNodeNum; }

int Node::getIntNodeNum() {
    if (auto PtrTNs = _parent_pointer.lock()) {
        int temp = PtrTNs->get_idx_from_node_num(UsrNodeNum);
        if (temp < 0) {
            if (DEBUG) {
                std::cout << "WARNING: Attribute unavailable. "
                          << "Probably the node was deleted. "
                          << "The node is now unassociated from TNs.\n";
            }
            _parent_pointer.reset();
        }
        return temp;
    } else {
        if (DEBUG) {
            std::cout << "WARNING: Node is not associated to any TNs. "
                      << "IntNodeNum is undefined. Returning -1.\n";
        }
        return -1;
    }
}

std::weak_ptr<Nodes> Node::getParentPointer() { return _parent_pointer; }

uint64_t Node::getintParentPointer() {
    auto PtrTNs = _parent_pointer.lock();

    return (uint64_t)PtrTNs.get();
}

void Node::set_thermal_nodes_parent(
    std::weak_ptr<Nodes> thermal_nodes_parent_ptr) {
    _parent_pointer = thermal_nodes_parent_ptr;
    _local_storage_destructor();
}

void Node::_local_storage_destructor() {
    delete m_local_storage_ptr;
    m_local_storage_ptr = nullptr;
}
