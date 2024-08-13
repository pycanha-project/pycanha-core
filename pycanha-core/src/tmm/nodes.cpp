#include "pycanha-core/tmm/nodes.hpp"

#include <iostream>
#include <string>

#include "pycanha-core/config.hpp"
#include "pycanha-core/parameters.hpp"

// Default constructor
Nodes::Nodes() : _node_num_mapped(false) {
    if (DEBUG) {
        std::cout << "Default constructor of TNs called " << self_pointer
                  << "\n";
    }

    // Create the shared pointer with a dummy destructor, otherwise TNs
    // destructor would be called twice
    self_pointer = std::shared_ptr<Nodes>(
        this, [](Nodes* p) { std::cout << "Self deleting \n"; });
}

// Nodes::Nodes (const Nodes&) = delete;
// Nodes& Nodes::operator= (const Nodes&) = delete;

// Destructor
Nodes::~Nodes() {
    if (DEBUG) {
        std::cout << "Destructor of TNs called " << this << "\n";
    }
}

void Nodes::add_node(Node& node) {
    // Info obtained from "node"
    char type = node.get_type();
    int node_num = node.get_node_num();

    // Update the node number mapping
    Index insert_idx;

    if (_usr_to_int_node_num.find(node_num) != _usr_to_int_node_num.end()) {
        // TODO: ERROR. Duplicated node
        std::cout << "ERROR. Node " << node_num << " already inserted.\n";
        return;
    }

    if (type == 'D') {
        auto it = std::upper_bound(_diff_node_num_vector.begin(),
                                   _diff_node_num_vector.end(), node_num);
        insert_idx = std::distance(_diff_node_num_vector.begin(), it);
    } else if (type == 'B') {
        auto it = std::upper_bound(_bound_node_num_vector.begin(),
                                   _bound_node_num_vector.end(), node_num);
        insert_idx = std::distance(_bound_node_num_vector.begin(), it);
        insert_idx += _diff_node_num_vector.size();
    } else {
        // TODO: ERROR. WRONG NODE TYPE
        std::cout << "ERROR. Wrong node type?\n";
        return;
    }

    _add_node_insert_idx(node, insert_idx);
}

void Nodes::add_nodes(std::vector<Node>& node_vector) {
    for (int i = 0; i < node_vector.size(); i++) {
        add_node(node_vector[i]);
    }
}

int Nodes::num_nodes() const { return T_vector.size(); }

int Nodes::get_num_nodes() const { return T_vector.size(); }

int Nodes::get_num_diff_nodes() const { return _diff_node_num_vector.size(); }

int Nodes::get_num_bound_nodes() const { return _bound_node_num_vector.size(); }

/*
Getters and setters are always the same except which atributte is needed.
*/
#define GET_SET_DOUBLE_ATTR(attr)                             \
    bool Nodes::set_##attr(int node_num, double attr) {       \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            attr##_vector[it->second] = attr;                 \
            return true;                                      \
        } else {                                              \
            std::cout << "Set Error: Node does not exist.\n"; \
            return false;                                     \
        }                                                     \
    }                                                         \
    double Nodes::get_##attr(int node_num) {                  \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            return attr##_vector[it->second];                 \
        } else {                                              \
            std::cout << "Get Error: Node does not exist.\n"; \
            return std::nan("");                              \
        }                                                     \
    }                                                         \
    double* Nodes::get_##attr##_value_ref(int node_num) {     \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            return &(attr##_vector[it->second]);              \
        } else {                                              \
            std::cout << "Get Error: Node does not exist.\n"; \
            return nullptr;                                   \
        }                                                     \
    }

#define GET_SET_DOUBLE_SPARSE(attr)                           \
    double Nodes::get_##attr(int node_num) {                  \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            return attr##_vector.coeff(it->second);           \
        } else {                                              \
            std::cout << "Get Error: Node does not exist.\n"; \
            return std::nan("");                              \
        }                                                     \
    }                                                         \
    bool Nodes::set_##attr(int node_num, double value) {      \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            attr##_vector.insert(it->second) = value;         \
            return true;                                      \
        } else {                                              \
            std::cout << "Set Error: Node does not exist.\n"; \
            return false;                                     \
        }                                                     \
    }                                                         \
    double* Nodes::get_##attr##_value_ref(int node_num) {     \
        if (!_node_num_mapped) {                              \
            create_node_num_map();                            \
        }                                                     \
        auto it = _usr_to_int_node_num.find(node_num);        \
        if (it != _usr_to_int_node_num.end()) {               \
            return &(attr##_vector.coeffRef(it->second));     \
        } else {                                              \
            std::cout << "Get Error: Node does not exist.\n"; \
            return nullptr;                                   \
        }                                                     \
    }

#define GET_SET_DOUBLE_ATTR_AND_LITERAL(attr) GET_SET_DOUBLE_ATTR(attr)

GET_SET_DOUBLE_ATTR(T)
GET_SET_DOUBLE_ATTR(C)

GET_SET_DOUBLE_SPARSE(qs)
GET_SET_DOUBLE_SPARSE(qa)
GET_SET_DOUBLE_SPARSE(qe)
GET_SET_DOUBLE_SPARSE(qi)
GET_SET_DOUBLE_SPARSE(qr)
GET_SET_DOUBLE_SPARSE(a)
GET_SET_DOUBLE_SPARSE(fx)
GET_SET_DOUBLE_SPARSE(fy)
GET_SET_DOUBLE_SPARSE(fz)
GET_SET_DOUBLE_SPARSE(eps)
GET_SET_DOUBLE_SPARSE(aph)

/////////////////////////////////////////////////////////////////////////

bool Nodes::set_type(int node_num, char Type) {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    if (!(Type == 'D' || Type == 'B')) {
        if (VERBOSE) {
            std::cout << "Error: Invalid node type. It should be 'D' or 'B'.\n";
        }
        return false;
    }

    auto current_type = get_type(node_num);

    if (current_type == Type) {
        return false;
    }

    if (current_type == 'D') {
        if (Type == 'B') {
            diffusive_to_boundary(node_num);
        }
        return true;
    } else if (current_type == 'B') {
        if (Type == 'D') {
            boundary_to_diffusive(node_num);
        }
        return true;
    } else {
        // Node does not exists
        return false;
    }
}
char Nodes::get_type(int node_num) {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    auto it = _usr_to_int_node_num.find(node_num);
    if (it != _usr_to_int_node_num.end()) {
        if (it->second < _diff_node_num_vector.size()) {
            return 'D';
        } else {
            return 'B';
        }
    } else {
        std::cout << "Error: Node does not exists.\n";
        return static_cast<char>(0);
    }
}

void Nodes::create_node_num_map() const {
    int int_num;
    for (int_num = 0; int_num < _diff_node_num_vector.size(); int_num++) {
        _usr_to_int_node_num[_diff_node_num_vector[int_num]] = int_num;
    }
    for (int bound_num = 0; bound_num < _bound_node_num_vector.size();
         bound_num++) {
        _usr_to_int_node_num[_bound_node_num_vector[bound_num]] = int_num;
        int_num++;
    }

    _node_num_mapped = true;
}

void Nodes::diffusive_to_boundary(int usr_node_num) {
    std::cout << "TODO: Not implemented yet\n";
    // 1. Copy all the info of usr_node_num to a new node not associated with
    // any TNs
    // 2. Change 'D' to 'B'
    // 3. Delete node from the model
    // 4. Insert the copy of the node in the model
}

void Nodes::boundary_to_diffusive(int usr_node_num) {
    std::cout << "TODO: Not implemented yet\n";
    // 1. Copy all the info of usr_node_num to a new node not associated with
    // any TNs
    // 2. Change 'B' to 'D'
    // 3. Delete node from the model
    // 4. Insert the copy of the node in the model
}

Index Nodes::get_idx_from_node_num(int node_num) const {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    auto it = _usr_to_int_node_num.find(node_num);
    if (it != _usr_to_int_node_num.end()) {
        return it->second;
    } else {
        std::cout << "Error: Node does not exists" << std::endl;
        return -1;
    }
}

int Nodes::get_node_num_from_idx(Index idx) const {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    if (0 <= idx && idx < _diff_node_num_vector.size()) {
        return _diff_node_num_vector[idx];
    } else if (idx <
               _diff_node_num_vector.size() + _bound_node_num_vector.size()) {
        idx -= _diff_node_num_vector.size();
        return _bound_node_num_vector[idx];
    } else {
        std::cout << "Error: Node does not exists\n";
        return -1;
    }
}

bool Nodes::is_node(int node_num) {
    if (get_idx_from_node_num(node_num) != -1) {
        return true;
    } else {
        return false;
    }
}

Node Nodes::get_node_from_node_num(int node_num) {
    if (is_node(node_num)) {
        return Node(node_num, self_pointer);
    } else {
        return Node(-1);
    }
}

Node Nodes::get_node_from_idx(Index idx) {
    return Node(get_node_num_from_idx(idx), self_pointer);
}

void Nodes::insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Index index, const LiteralString& string) {
    sparse.conservativeResize(sparse.size() + 1);
    for (int i = sparse.nonZeros() - 1;
         i >= 0 && sparse.innerIndexPtr()[i] >= index; i--) {
        sparse.innerIndexPtr()[i] += 1;
    }
    if (!string.is_empty()) {
        sparse.insert(index) = string;
    }
}

void Nodes::insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Eigen::Index index, const std::string& string) {
    insert_displace(sparse, index, LiteralString(string));
}

void Nodes::insert_displace(Eigen::SparseVector<double>& sparse, Index index,
                            double value) {
    sparse.conservativeResize(sparse.size() + 1);
    for (int i = sparse.nonZeros() - 1;
         i >= 0 && sparse.innerIndexPtr()[i] >= index; i--) {
        sparse.innerIndexPtr()[i] += 1;
    }
    if (std::abs(value) > ZERO_THR_ATTR) {
        sparse.insert(index) = value;
    }
}

void Nodes::delete_displace(Eigen::SparseVector<LiteralString>& sparse,
                            int index) {
    int to_remove_index = -1;
    for (int i = 0; i < sparse.nonZeros(); i++) {
        if (sparse.innerIndexPtr()[i] == index) {
            to_remove_index = i;
            break;
        }
    }
    if (to_remove_index >= 0) {
        for (int i = to_remove_index + 1; i < sparse.nonZeros(); i++) {
            sparse.innerIndexPtr()[i - 1] = sparse.innerIndexPtr()[i] - 1;
            sparse.valuePtr()[i - 1] = sparse.valuePtr()[i];
        }
        sparse.conservativeResize(sparse.size() - 1);
    }
}

void Nodes::delete_displace(Eigen::SparseVector<double>& sparse, int index) {
    sparse.coeffRef(index) = 0.0;
    sparse.prune(ZERO_THR_ATTR, 1.0);  // sparse.prune(ref,prec); num <=
                                       // ref*prec
}

std::string Nodes::get_literal_C(int node_num) const {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    auto it = _usr_to_int_node_num.find(node_num);
    if (it != _usr_to_int_node_num.end()) {
        return literals_C.coeff(it->second).get_literal();
    } else {
        std::cout << "Get Error: Node does not exist.\n";
        return std::string();
    }
}

bool Nodes::set_literal_C(int node_num, std::string str) {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
    auto it = _usr_to_int_node_num.find(node_num);
    if (it != _usr_to_int_node_num.end()) {
        literals_C.insert(it->second) = str;
        return true;
    } else {
        std::cout << "Set Error: Node does not exist.\n";
        return false;
    }
}

void Nodes::remove_node(int node_num) {
    auto idx = get_idx_from_node_num(node_num);

    if (idx < 0) {
        // Node does not exist
        if (VERBOSE) {
            std::cout << "Node " << node_num << " does not exists."
                      << std::endl;
        }
        return;
    }

    _usr_to_int_node_num.extract(node_num);

    T_vector.erase(T_vector.begin() + idx);
    C_vector.erase(C_vector.begin() + idx);

    delete_displace(qs_vector, idx);
    delete_displace(qa_vector, idx);
    delete_displace(qe_vector, idx);
    delete_displace(qi_vector, idx);
    delete_displace(qr_vector, idx);
    delete_displace(a_vector, idx);
    delete_displace(fx_vector, idx);
    delete_displace(fy_vector, idx);
    delete_displace(fz_vector, idx);
    delete_displace(eps_vector, idx);
    delete_displace(aph_vector, idx);

    delete_displace(literals_C, idx);

    // Reshape node vector and conductors
    if (idx < _diff_node_num_vector.size()) {
        _diff_node_num_vector.erase(_diff_node_num_vector.begin() + idx);
    } else {
        _bound_node_num_vector.erase(_bound_node_num_vector.begin() +
                                     (idx - _diff_node_num_vector.size()));
    }
    _node_num_mapped = false;
}

void Nodes::_add_node_insert_idx(Node& node, Index insert_idx) {
    // Info obtained from "node"
    char type = node.get_type();
    int node_num = node.get_node_num();

    if (type == 'D') {
        _diff_node_num_vector.insert(_diff_node_num_vector.begin() + insert_idx,
                                     node_num);
    } else if (type == 'B') {
        _bound_node_num_vector.insert(
            _bound_node_num_vector.begin() +
                (insert_idx - static_cast<Index>(_diff_node_num_vector.size())),
            node_num);
    } else {
        // TODO: ERROR. WRONG NODE TYPE
        std::cout << "ERROR. Wrong node type?\n";
        return;
    }

    _node_num_mapped = false;
    int nn = num_nodes() + 1;

// Fill the containers with the node properties
#define FILL_VECTOR_ATTR(attr)                           \
    auto it_##attr = attr##_vector.begin() + insert_idx; \
    attr##_vector.insert(it_##attr, node.get_##attr());

    FILL_VECTOR_ATTR(T)
    FILL_VECTOR_ATTR(C)

    // FILL_VECTOR_ATTR(aph)
    insert_displace(qs_vector, insert_idx, node.get_qs());
    insert_displace(qa_vector, insert_idx, node.get_qa());
    insert_displace(qe_vector, insert_idx, node.get_qe());
    insert_displace(qi_vector, insert_idx, node.get_qi());
    insert_displace(qr_vector, insert_idx, node.get_qr());
    insert_displace(a_vector, insert_idx, node.get_a());
    insert_displace(fx_vector, insert_idx, node.get_fx());
    insert_displace(fy_vector, insert_idx, node.get_fy());
    insert_displace(fz_vector, insert_idx, node.get_fz());
    insert_displace(eps_vector, insert_idx, node.get_eps());
    insert_displace(aph_vector, insert_idx, node.get_aph());

    insert_displace(literals_C, insert_idx, node.get_literal_C());

    // The node instance now points to this TNs instance
    node.set_thermal_nodes_parent(self_pointer);

    return;
}

bool Nodes::is_mapped() const { return _node_num_mapped; }
