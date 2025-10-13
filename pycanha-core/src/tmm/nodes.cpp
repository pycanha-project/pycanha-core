#include "pycanha-core/tmm/nodes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/parameters.hpp"
#include "pycanha-core/tmm/node.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// Default constructor
Nodes::Nodes() : estimated_number_of_nodes(100), _node_num_mapped(false) {
    if (DEBUG) {
        std::cout << "Default constructor of TNs called " << _self_pointer
                  << "\n";
    }

    // Create the shared pointer with a dummy destructor, otherwise TNs
    // destructor would be called twice
    _self_pointer = std::shared_ptr<Nodes>(
        this, [](Nodes* /*p*/) { std::cout << "Self deleting \n"; });
}

// Copy Constructor
Nodes::Nodes(const Nodes& other)
    : estimated_number_of_nodes(other.estimated_number_of_nodes),
      _diff_node_num_vector(other._diff_node_num_vector),
      _bound_node_num_vector(other._bound_node_num_vector),
      T_vector(other.T_vector),
      C_vector(other.C_vector),
      qs_vector(other.qs_vector),
      qa_vector(other.qa_vector),
      qe_vector(other.qe_vector),
      qi_vector(other.qi_vector),
      qr_vector(other.qr_vector),
      a_vector(other.a_vector),
      fx_vector(other.fx_vector),
      fy_vector(other.fy_vector),
      fz_vector(other.fz_vector),
      eps_vector(other.eps_vector),
      aph_vector(other.aph_vector),
      literals_C(other.literals_C),
      literals_qs(other.literals_qs),
      literals_qa(other.literals_qa),
      literals_qe(other.literals_qe),
      literals_qi(other.literals_qi),
      literals_qr(other.literals_qr),
      literals_a(other.literals_a),
      literals_fx(other.literals_fx),
      literals_fy(other.literals_fy),
      literals_fz(other.literals_fz),
      literals_eps(other.literals_eps),
      literals_aph(other.literals_aph),
      _usr_to_int_node_num(other._usr_to_int_node_num),
      _node_num_mapped(other._node_num_mapped) {
    if (DEBUG) {
        std::cout << "Copy constructor of TNs called " << &other << " -> "
                  << this << "\n";
    }

    // Create the shared pointer with a dummy destructor, otherwise TNs
    // destructor would be called twice
    _self_pointer = std::shared_ptr<Nodes>(
        this, [](Nodes* /*p*/) { std::cout << "Self deleting \n"; });
}

// Copy Assignment Operator
Nodes& Nodes::operator=(const Nodes& other) {
    if (this != &other) {
        if (DEBUG) {
            std::cout << "Copy assignment operator of TNs called " << &other
                      << " -> " << this << "\n";
        }

        // Copy data members
        estimated_number_of_nodes = other.estimated_number_of_nodes;
        _diff_node_num_vector = other._diff_node_num_vector;
        _bound_node_num_vector = other._bound_node_num_vector;
        T_vector = other.T_vector;
        C_vector = other.C_vector;
        qs_vector = other.qs_vector;
        qa_vector = other.qa_vector;
        qe_vector = other.qe_vector;
        qi_vector = other.qi_vector;
        qr_vector = other.qr_vector;
        a_vector = other.a_vector;
        fx_vector = other.fx_vector;
        fy_vector = other.fy_vector;
        fz_vector = other.fz_vector;
        eps_vector = other.eps_vector;
        aph_vector = other.aph_vector;
        literals_C = other.literals_C;
        literals_qs = other.literals_qs;
        literals_qa = other.literals_qa;
        literals_qe = other.literals_qe;
        literals_qi = other.literals_qi;
        literals_qr = other.literals_qr;
        literals_a = other.literals_a;
        literals_fx = other.literals_fx;
        literals_fy = other.literals_fy;
        literals_fz = other.literals_fz;
        literals_eps = other.literals_eps;
        literals_aph = other.literals_aph;
        _usr_to_int_node_num = other._usr_to_int_node_num;
        _node_num_mapped = other._node_num_mapped;

        // Recreate self_pointer
        _self_pointer = std::shared_ptr<Nodes>(
            this, [](Nodes* /*p*/) { std::cout << "Self deleting \n"; });
    }
    return *this;
}

// Move Constructor
Nodes::Nodes(Nodes&& other) noexcept
    : estimated_number_of_nodes(other.estimated_number_of_nodes),
      _diff_node_num_vector(std::move(other._diff_node_num_vector)),
      _bound_node_num_vector(std::move(other._bound_node_num_vector)),
      T_vector(std::move(other.T_vector)),
      C_vector(std::move(other.C_vector)),
      qs_vector(std::move(other.qs_vector)),
      qa_vector(std::move(other.qa_vector)),
      qe_vector(std::move(other.qe_vector)),
      qi_vector(std::move(other.qi_vector)),
      qr_vector(std::move(other.qr_vector)),
      a_vector(std::move(other.a_vector)),
      fx_vector(std::move(other.fx_vector)),
      fy_vector(std::move(other.fy_vector)),
      fz_vector(std::move(other.fz_vector)),
      eps_vector(std::move(other.eps_vector)),
      aph_vector(std::move(other.aph_vector)),
      literals_C(std::move(other.literals_C)),
      literals_qs(std::move(other.literals_qs)),
      literals_qa(std::move(other.literals_qa)),
      literals_qe(std::move(other.literals_qe)),
      literals_qi(std::move(other.literals_qi)),
      literals_qr(std::move(other.literals_qr)),
      literals_a(std::move(other.literals_a)),
      literals_fx(std::move(other.literals_fx)),
      literals_fy(std::move(other.literals_fy)),
      literals_fz(std::move(other.literals_fz)),
      literals_eps(std::move(other.literals_eps)),
      literals_aph(std::move(other.literals_aph)),
      _usr_to_int_node_num(std::move(other._usr_to_int_node_num)),
      _node_num_mapped(other._node_num_mapped) {
    if (DEBUG) {
        std::cout << "Move constructor of TNs called " << &other << " -> "
                  << this << "\n";
    }

    // Transfer the self_pointer
    _self_pointer = std::move(other._self_pointer);

    // Reset the other object's self_pointer
    other._self_pointer.reset();
}

// Move Assignment Operator
Nodes& Nodes::operator=(Nodes&& other) noexcept {
    if (this != &other) {
        if (DEBUG) {
            std::cout << "Move assignment operator of TNs called " << &other
                      << " -> " << this << "\n";
        }

        // Move data members
        estimated_number_of_nodes = other.estimated_number_of_nodes;
        _diff_node_num_vector = std::move(other._diff_node_num_vector);
        _bound_node_num_vector = std::move(other._bound_node_num_vector);
        T_vector = std::move(other.T_vector);
        C_vector = std::move(other.C_vector);
        qs_vector = std::move(other.qs_vector);
        qa_vector = std::move(other.qa_vector);
        qe_vector = std::move(other.qe_vector);
        qi_vector = std::move(other.qi_vector);
        qr_vector = std::move(other.qr_vector);
        a_vector = std::move(other.a_vector);
        fx_vector = std::move(other.fx_vector);
        fy_vector = std::move(other.fy_vector);
        fz_vector = std::move(other.fz_vector);
        eps_vector = std::move(other.eps_vector);
        aph_vector = std::move(other.aph_vector);
        literals_C = std::move(other.literals_C);
        literals_qs = std::move(other.literals_qs);
        literals_qa = std::move(other.literals_qa);
        literals_qe = std::move(other.literals_qe);
        literals_qi = std::move(other.literals_qi);
        literals_qr = std::move(other.literals_qr);
        literals_a = std::move(other.literals_a);
        literals_fx = std::move(other.literals_fx);
        literals_fy = std::move(other.literals_fy);
        literals_fz = std::move(other.literals_fz);
        literals_eps = std::move(other.literals_eps);
        literals_aph = std::move(other.literals_aph);
        _usr_to_int_node_num = std::move(other._usr_to_int_node_num);
        _node_num_mapped = other._node_num_mapped;

        // Transfer the self_pointer
        _self_pointer = std::move(other._self_pointer);

        // Reset the other object's self_pointer
        other._self_pointer.reset();
    }
    return *this;
}

// Destructor
Nodes::~Nodes() {
    if (DEBUG) {
        std::cout << "Destructor of TNs called " << this << "\n";
    }
}

void Nodes::add_node(Node& node) {
    // Info obtained from "node"
    const char type = node.get_type();
    const int node_num = node.get_node_num();

    // Update the node number mapping
    Index insert_idx = 0;

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

    add_node_insert_idx(node, insert_idx);
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
    if (Type != 'D' && Type != 'B') {
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
    int int_num = 0;
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
        std::cout << "Error: Node does not exists" << '\n';
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

bool Nodes::is_node(int node_num) const {
    return get_idx_from_node_num(node_num) != -1;
}

Node Nodes::get_node_from_node_num(int node_num) {
    if (is_node(node_num)) {
        return Node(node_num, _self_pointer);
    } else {
        return Node(-1);
    }
}

Node Nodes::get_node_from_idx(Index idx) {
    return {get_node_num_from_idx(idx), _self_pointer};
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
                            Index index) {
    std::vector<int> indices;
    std::vector<LiteralString> values;
    for (typename Eigen::SparseVector<LiteralString>::InnerIterator it(sparse);
         it; ++it) {
        if (it.index() != index) {
            indices.push_back(it.index() > index ? it.index() - 1 : it.index());
            values.push_back(it.value());
        }
    }
    sparse.setZero();
    for (size_t i = 0; i < indices.size(); ++i) {
        sparse.coeffRef(indices[i]) = values[i];
    }
}

void Nodes::delete_displace(Eigen::SparseVector<double>& sparse, Index index) {
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
        return {};
    }
}

bool Nodes::set_literal_C(int node_num, const std::string& str) {
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
            std::cout << "Node " << node_num << " does not exists." << '\n';
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
        _bound_node_num_vector.erase(
            _bound_node_num_vector.begin() +
            (idx - static_cast<Index>(_diff_node_num_vector.size())));
    }
    _node_num_mapped = false;
}

void Nodes::add_node_insert_idx(Node& node, Index insert_idx) {
    // Info obtained from "node"
    const char type = node.get_type();
    const int node_num = node.get_node_num();

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
    // int nn = num_nodes() + 1;

    // Fill the containers with the node properties
    auto it_t = T_vector.begin() + insert_idx;
    T_vector.insert(it_t, node.get_T());
    auto it_c = C_vector.begin() + insert_idx;
    C_vector.insert(it_c, node.get_C());

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

    // TODO: The Sparse Vector for LiteralString is not working properly. FIX
    // _insert_displace(literals_C, insert_idx, node.get_literal_C());
    // _insert_displace(literals_C, insert_idx, "NOT IMPLEMENTED");

    // The node instance now points to this TNs instance
    node.set_thermal_nodes_parent(_self_pointer);
}

bool Nodes::is_mapped() const { return _node_num_mapped; }
