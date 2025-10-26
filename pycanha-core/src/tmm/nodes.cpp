#include "pycanha-core/tmm/nodes.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/literalstring.hpp"
#include "pycanha-core/tmm/node.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// Default constructor
Nodes::Nodes() {
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

void Nodes::ensure_node_map() const {
    if (!_node_num_mapped) {
        create_node_num_map();
    }
}

bool Nodes::find_node_index(int node_num, Index& index,
                            const char* error_prefix) const {
    ensure_node_map();
    auto it = _usr_to_int_node_num.find(node_num);
    if (it == _usr_to_int_node_num.end()) {
        if (error_prefix != nullptr) {
            std::cout << error_prefix << " Error: Node does not exist.\n";
        }
        return false;
    }
    index = it->second;
    return true;
}

double Nodes::resolve_get_dense_attr(int node_num,
                                     const std::vector<double>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return std::nan("");
    }
    return storage[static_cast<std::size_t>(index)];
}

bool Nodes::resolve_set_dense_attr(int node_num, std::vector<double>& storage,
                                   double value) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Set")) {
        return false;
    }
    storage[static_cast<std::size_t>(index)] = value;
    return true;
}

double* Nodes::resolve_get_dense_attr_ref(int node_num,
                                          std::vector<double>& storage) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return nullptr;
    }
    return &storage[static_cast<std::size_t>(index)];
}

double Nodes::resolve_get_sparse_attr(
    int node_num, const Eigen::SparseVector<double>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return std::nan("");
    }
    return storage.coeff(index);
}

bool Nodes::resolve_set_sparse_attr(int node_num,
                                    Eigen::SparseVector<double>& storage,
                                    double value) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Set")) {
        return false;
    }
    storage.coeffRef(index) = value;
    return true;
}

double* Nodes::resolve_get_sparse_attr_ref(
    int node_num, Eigen::SparseVector<double>& storage) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return nullptr;
    }
    return &storage.coeffRef(index);
}

std::string Nodes::resolve_get_literal_attr(
    int node_num, const Eigen::SparseVector<LiteralString>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return {};
    }
    return storage.coeff(index).get_literal();
}

bool Nodes::resolve_set_literal_attr(
    int node_num, Eigen::SparseVector<LiteralString>& storage,
    const std::string& value) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Set")) {
        return false;
    }
    storage.coeffRef(index) = value;
    return true;
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
        insert_idx = static_cast<Index>(
            std::distance(_diff_node_num_vector.begin(), it));
    } else if (type == 'B') {
        auto it = std::upper_bound(_bound_node_num_vector.begin(),
                                   _bound_node_num_vector.end(), node_num);
        insert_idx = static_cast<Index>(
            std::distance(_bound_node_num_vector.begin(), it));
        insert_idx += static_cast<Index>(_diff_node_num_vector.size());
    } else {
        // TODO: ERROR. WRONG NODE TYPE
        std::cout << "ERROR. Wrong node type?\n";
        return;
    }

    add_node_insert_idx(node, insert_idx);
}

void Nodes::add_nodes(std::vector<Node>& node_vector) {
    for (auto& node : node_vector) {
        add_node(node);
    }
}

int Nodes::num_nodes() const { return static_cast<int>(T_vector.size()); }

int Nodes::get_num_nodes() const { return static_cast<int>(T_vector.size()); }

int Nodes::get_num_diff_nodes() const {
    return static_cast<int>(_diff_node_num_vector.size());
}

int Nodes::get_num_bound_nodes() const {
    return static_cast<int>(_bound_node_num_vector.size());
}

double Nodes::get_T(int node_num) {
    return resolve_get_dense_attr(node_num, T_vector);
}

double Nodes::get_C(int node_num) {
    return resolve_get_dense_attr(node_num, C_vector);
}

bool Nodes::set_T(int node_num, double T) {
    return resolve_set_dense_attr(node_num, T_vector, T);
}

bool Nodes::set_C(int node_num, double C) {
    return resolve_set_dense_attr(node_num, C_vector, C);
}

double* Nodes::get_T_value_ref(int node_num) {
    return resolve_get_dense_attr_ref(node_num, T_vector);
}

double* Nodes::get_C_value_ref(int node_num) {
    return resolve_get_dense_attr_ref(node_num, C_vector);
}

double Nodes::get_qs(int node_num) {
    return resolve_get_sparse_attr(node_num, qs_vector);
}

double Nodes::get_qa(int node_num) {
    return resolve_get_sparse_attr(node_num, qa_vector);
}

double Nodes::get_qe(int node_num) {
    return resolve_get_sparse_attr(node_num, qe_vector);
}

double Nodes::get_qi(int node_num) {
    return resolve_get_sparse_attr(node_num, qi_vector);
}

double Nodes::get_qr(int node_num) {
    return resolve_get_sparse_attr(node_num, qr_vector);
}

double Nodes::get_a(int node_num) {
    return resolve_get_sparse_attr(node_num, a_vector);
}

double Nodes::get_fx(int node_num) {
    return resolve_get_sparse_attr(node_num, fx_vector);
}

double Nodes::get_fy(int node_num) {
    return resolve_get_sparse_attr(node_num, fy_vector);
}

double Nodes::get_fz(int node_num) {
    return resolve_get_sparse_attr(node_num, fz_vector);
}

double Nodes::get_eps(int node_num) {
    return resolve_get_sparse_attr(node_num, eps_vector);
}

double Nodes::get_aph(int node_num) {
    return resolve_get_sparse_attr(node_num, aph_vector);
}

bool Nodes::set_qs(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, qs_vector, value);
}

bool Nodes::set_qa(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, qa_vector, value);
}

bool Nodes::set_qe(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, qe_vector, value);
}

bool Nodes::set_qi(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, qi_vector, value);
}

bool Nodes::set_qr(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, qr_vector, value);
}

bool Nodes::set_a(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, a_vector, value);
}

bool Nodes::set_fx(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, fx_vector, value);
}

bool Nodes::set_fy(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, fy_vector, value);
}

bool Nodes::set_fz(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, fz_vector, value);
}

bool Nodes::set_eps(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, eps_vector, value);
}

bool Nodes::set_aph(int node_num, double value) {
    return resolve_set_sparse_attr(node_num, aph_vector, value);
}

double* Nodes::get_qs_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, qs_vector);
}

double* Nodes::get_qa_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, qa_vector);
}

double* Nodes::get_qe_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, qe_vector);
}

double* Nodes::get_qi_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, qi_vector);
}

double* Nodes::get_qr_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, qr_vector);
}

double* Nodes::get_a_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, a_vector);
}

double* Nodes::get_fx_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, fx_vector);
}

double* Nodes::get_fy_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, fy_vector);
}

double* Nodes::get_fz_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, fz_vector);
}

double* Nodes::get_eps_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, eps_vector);
}

double* Nodes::get_aph_value_ref(int node_num) {
    return resolve_get_sparse_attr_ref(node_num, aph_vector);
}

/////////////////////////////////////////////////////////////////////////

bool Nodes::set_type(int node_num, char type) {
    ensure_node_map();
    if (type != 'D' && type != 'B') {
        if (VERBOSE) {
            std::cout << "Error: Invalid node type. It should be 'D' or 'B'.\n";
        }
        return false;
    }

    const auto current_type = get_type(node_num);

    if (current_type == type) {
        return false;
    }

    if (current_type == 'D') {
        if (type == 'B') {
            diffusive_to_boundary(node_num);
        }
        return true;
    } else if (current_type == 'B') {
        if (type == 'D') {
            boundary_to_diffusive(node_num);
        }
        return true;
    } else {
        // Node does not exists
        return false;
    }
}
char Nodes::get_type(int node_num) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return static_cast<char>(0);
    }

    const auto diff_size = static_cast<Index>(_diff_node_num_vector.size());
    if (index < diff_size) {
        return 'D';
    }
    return 'B';
}

void Nodes::create_node_num_map() const {
    _usr_to_int_node_num.clear();

    Index internal_index = 0;
    for (const int diff_node_num : _diff_node_num_vector) {
        _usr_to_int_node_num[diff_node_num] = static_cast<int>(internal_index);
        ++internal_index;
    }
    for (const int bound_node_num : _bound_node_num_vector) {
        _usr_to_int_node_num[bound_node_num] = static_cast<int>(internal_index);
        ++internal_index;
    }

    _node_num_mapped = true;
}

void Nodes::diffusive_to_boundary(int usr_node_num) {
    std::cout << "TODO: Not implemented yet\n";
    static_cast<void>(_node_num_mapped);
    static_cast<void>(usr_node_num);
    // 1. Copy all the info of usr_node_num to a new node not associated with
    // any TNs
    // 2. Change 'D' to 'B'
    // 3. Delete node from the model
    // 4. Insert the copy of the node in the model
}

void Nodes::boundary_to_diffusive(int usr_node_num) {
    std::cout << "TODO: Not implemented yet\n";
    static_cast<void>(_node_num_mapped);
    static_cast<void>(usr_node_num);
    // 1. Copy all the info of usr_node_num to a new node not associated with
    // any TNs
    // 2. Change 'B' to 'D'
    // 3. Delete node from the model
    // 4. Insert the copy of the node in the model
}

Index Nodes::get_idx_from_node_num(int node_num) const {
    Index index = 0;
    if (!find_node_index(node_num, index, nullptr)) {
        std::cout << "Error: Node does not exists" << '\n';
        return -1;
    }
    return index;
}

int Nodes::get_node_num_from_idx(Index idx) const {
    ensure_node_map();
    const auto diff_size = static_cast<Index>(_diff_node_num_vector.size());
    const auto bound_size = static_cast<Index>(_bound_node_num_vector.size());

    if (idx < 0) {
        std::cout << "Error: Node does not exists\n";
        return -1;
    }

    if (idx < diff_size) {
        return _diff_node_num_vector[static_cast<std::size_t>(idx)];
    }

    const auto adjusted = idx - diff_size;
    if (adjusted >= 0 && adjusted < bound_size) {
        return _bound_node_num_vector[static_cast<std::size_t>(adjusted)];
    }

    std::cout << "Error: Node does not exists\n";
    return -1;
}

bool Nodes::is_node(int node_num) const {
    Index index = 0;
    return find_node_index(node_num, index, nullptr);
}

Node Nodes::get_node_from_node_num(int node_num) {
    if (is_node(node_num)) {
        return {node_num, _self_pointer};
    } else {
        return Node(-1);
    }
}

Node Nodes::get_node_from_idx(Index idx) {
    return {get_node_num_from_idx(idx), _self_pointer};
}

void Nodes::insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Index index, const LiteralString& string) {
    Eigen::SparseVector<LiteralString> result(sparse.size() + 1);
    result.reserve(sparse.nonZeros() + (string.is_empty() ? 0 : 1));

    for (typename Eigen::SparseVector<LiteralString>::InnerIterator it(sparse);
         it; ++it) {
        const Index target_index =
            it.index() >= index ? it.index() + 1 : it.index();
        result.coeffRef(target_index) = it.value();
    }

    if (!string.is_empty()) {
        result.coeffRef(index) = string;
    }

    sparse = result;
}

void Nodes::insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Eigen::Index index, const std::string& string) {
    insert_displace(sparse, index, LiteralString(string));
}

void Nodes::insert_displace(Eigen::SparseVector<double>& sparse, Index index,
                            double value) {
    Eigen::SparseVector<double> result(sparse.size() + 1);
    const bool store_value = std::abs(value) > ZERO_THR_ATTR;
    result.reserve(sparse.nonZeros() + (store_value ? 1 : 0));

    for (typename Eigen::SparseVector<double>::InnerIterator it(sparse); it;
         ++it) {
        const Index target_index =
            it.index() >= index ? it.index() + 1 : it.index();
        result.coeffRef(target_index) = it.value();
    }

    if (store_value) {
        result.coeffRef(index) = value;
    }

    sparse = result;
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
    return resolve_get_literal_attr(node_num, literals_C);
}

bool Nodes::set_literal_C(int node_num, const std::string& str) {
    return resolve_set_literal_attr(node_num, literals_C, str);
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
