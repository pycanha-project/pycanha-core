#include "pycanha-core/tmm/nodes.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/literalstring.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/utils/logger.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// Default constructor
Nodes::Nodes() {
    SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                        "Default constructor of TNs called");

    // Create the shared pointer with a dummy destructor, otherwise TNs
    // destructor would be called twice
    _self_pointer = std::shared_ptr<Nodes>(this, [](Nodes* /*p*/) {});
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
      _dense_node_index(other._dense_node_index),
      _dense_base(other._dense_base),
      _dense_mapped(other._dense_mapped),
      _structure_version(other._structure_version),
      _mapped_version(other._mapped_version) {
    SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                        "Copy constructor of TNs called");

    // Create the shared pointer with a dummy destructor, otherwise TNs
    // destructor would be called twice
    _self_pointer = std::shared_ptr<Nodes>(this, [](Nodes* /*p*/) {});
}

// Copy Assignment Operator
Nodes& Nodes::operator=(const Nodes& other) {
    if (this != &other) {
        SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                            "Copy assignment operator of TNs called");

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
        _dense_node_index = other._dense_node_index;
        _dense_base = other._dense_base;
        _dense_mapped = other._dense_mapped;
        // A whole new structure under whatever observes this instance: a
        // version neither side has used yet.
        _structure_version =
            std::max(_structure_version, other._structure_version) + 1U;
        _mapped_version = other.is_mapped() ? _structure_version : 0U;

        // Recreate self_pointer
        _self_pointer = std::shared_ptr<Nodes>(this, [](Nodes* /*p*/) {});
    }
    return *this;
}

// Move Constructor
Nodes::Nodes(Nodes&& other) noexcept
    : estimated_number_of_nodes(other.estimated_number_of_nodes),
      _self_pointer(std::move(other._self_pointer)),
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
      _dense_node_index(std::move(other._dense_node_index)),
      _dense_base(other._dense_base),
      _dense_mapped(other._dense_mapped),
      _structure_version(other._structure_version),
      _mapped_version(other._mapped_version) {
    SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                        "Move constructor of TNs called");

    // Reset the other object's self_pointer
    other._self_pointer.reset();
}

// Move Assignment Operator
Nodes& Nodes::operator=(Nodes&& other) noexcept {
    if (this != &other) {
        SPDLOG_LOGGER_TRACE(pycanha::get_logger(),
                            "Move assignment operator of TNs called");

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
        const bool other_mapped = other.is_mapped();
        _usr_to_int_node_num = std::move(other._usr_to_int_node_num);
        _dense_node_index = std::move(other._dense_node_index);
        _dense_base = other._dense_base;
        _dense_mapped = other._dense_mapped;
        _structure_version =
            std::max(_structure_version, other._structure_version) + 1U;
        _mapped_version = other_mapped ? _structure_version : 0U;

        // Transfer the self_pointer
        _self_pointer = std::move(other._self_pointer);

        // Reset the other object's self_pointer
        other._self_pointer.reset();
    }
    return *this;
}

// Destructor
Nodes::~Nodes() {
    SPDLOG_LOGGER_TRACE(pycanha::get_logger(), "Destructor of TNs called");
    // A coupling container still registered here cannot own this instance
    // (it would have unregistered before releasing it), so it holds a
    // non-owning handle that is about to dangle: cut it.
    while (_first_observer != nullptr) {
        Couplings* observer = _first_observer;
        remove_observer(observer);
        observer->_nodes.reset();
    }
}

void Nodes::ensure_node_map() const {
    if (!is_mapped()) {
        create_node_num_map();
    }
}

std::optional<Index> Nodes::lookup_node_index(NodeNum node_num) const {
    if (_dense_mapped) {
        const Index offset =
            static_cast<Index>(node_num) - static_cast<Index>(_dense_base);
        if (offset < 0 || offset >= std::ssize(_dense_node_index)) {
            return std::nullopt;
        }
        const std::int32_t index = _dense_node_index[to_sizet(offset)];
        if (index < 0) {
            return std::nullopt;
        }
        return Index{index};
    }
    const auto it = _usr_to_int_node_num.find(node_num);
    if (it == _usr_to_int_node_num.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool Nodes::find_node_index(NodeNum node_num, Index& index,
                            const char* error_prefix) const {
    ensure_node_map();
    const std::optional<Index> found = lookup_node_index(node_num);
    if (!found.has_value()) {
        if (error_prefix != nullptr) {
            SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                               "{} Error: Node does not exist.", error_prefix);
        }
        return false;
    }
    index = *found;
    return true;
}

double Nodes::resolve_get_dense_attr(NodeNum node_num,
                                     const std::vector<double>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return std::nan("");
    }
    return storage[to_sizet(index)];
}

bool Nodes::resolve_set_dense_attr(NodeNum node_num,
                                   std::vector<double>& storage, double value) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Set")) {
        return false;
    }
    storage[to_sizet(index)] = value;
    return true;
}

double* Nodes::resolve_get_dense_attr_ref(NodeNum node_num,
                                          std::vector<double>& storage) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return nullptr;
    }
    return &storage[to_sizet(index)];
}

double Nodes::resolve_get_sparse_attr(
    NodeNum node_num, const Eigen::SparseVector<double>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return std::nan("");
    }
    return storage.coeff(index);
}

bool Nodes::resolve_set_sparse_attr(NodeNum node_num,
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
    NodeNum node_num, Eigen::SparseVector<double>& storage) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return nullptr;
    }
    return &storage.coeffRef(index);
}

std::string Nodes::resolve_get_literal_attr(
    NodeNum node_num, const Eigen::SparseVector<LiteralString>& storage) const {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return {};
    }
    return storage.coeff(index).get_literal();
}

bool Nodes::resolve_set_literal_attr(
    NodeNum node_num, Eigen::SparseVector<LiteralString>& storage,
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
    const NodeNum node_num = node.get_node_num();

    Index insert_idx = 0;

    // Searched in the sorted number vectors rather than in the map, so an
    // insertion never forces a rebuild of the map.
    if (has_node_number(node_num)) {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Node {} already inserted.",
                            node_num);
        return;
    }

    if (type == 'D') {
        auto it = std::ranges::upper_bound(_diff_node_num_vector, node_num);
        insert_idx = to_idx(std::distance(_diff_node_num_vector.begin(), it));
    } else if (type == 'B') {
        auto it = std::ranges::upper_bound(_bound_node_num_vector, node_num);
        insert_idx = to_idx(std::distance(_bound_node_num_vector.begin(), it));
        insert_idx += to_idx(_diff_node_num_vector.size());
    } else {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Wrong node type.");
        return;
    }

    add_node_insert_idx(node, insert_idx);
}

void Nodes::add_nodes(std::vector<Node>& node_vector) {
    for (auto& node : node_vector) {
        add_node(node);
    }
}

Index Nodes::num_nodes() const { return to_idx(T_vector.size()); }

Index Nodes::get_num_nodes() const { return to_idx(T_vector.size()); }

Index Nodes::get_num_diff_nodes() const {
    return to_idx(_diff_node_num_vector.size());
}

Index Nodes::get_num_bound_nodes() const {
    return to_idx(_bound_node_num_vector.size());
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

bool Nodes::set_type(NodeNum node_num, char type) {
    if (type != 'D' && type != 'B') {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Invalid node type. It should be 'D' or 'B'.");
        return false;
    }
    // Like the other setters, true whenever the node exists: a node that
    // already has the type is left as it is.
    const char current_type = get_type(node_num);
    if (current_type == 0) {
        return false;
    }
    if (current_type != type) {
        static_cast<void>(
            set_types(std::span<const NodeNum>(&node_num, 1), type));
    }
    return true;
}

char Nodes::get_type(NodeNum node_num) {
    Index index = 0;
    if (!find_node_index(node_num, index, "Get")) {
        return static_cast<char>(0);
    }

    const auto diff_size = to_idx(_diff_node_num_vector.size());
    if (index < diff_size) {
        return 'D';
    }
    return 'B';
}

namespace {

// Largest node-number span the flat table is used for, as a function of the
// node count: beyond it the table would be mostly holes and the unordered_map
// is the smaller of the two.
[[nodiscard]] Index dense_map_limit(Index num_nodes) noexcept {
    return (4 * num_nodes) + 1024;
}

}  // namespace

void Nodes::create_node_num_map() const {
    _usr_to_int_node_num.clear();
    _dense_node_index.clear();

    const Index node_count = to_idx(_diff_node_num_vector.size()) +
                             to_idx(_bound_node_num_vector.size());
    _dense_mapped = true;
    _dense_base = 0;
    if (node_count > 0) {
        // Both vectors are sorted, so their ends bound every number.
        NodeNum lowest = std::numeric_limits<NodeNum>::max();
        NodeNum highest = std::numeric_limits<NodeNum>::min();
        for (const auto* numbers :
             {&_diff_node_num_vector, &_bound_node_num_vector}) {
            if (!numbers->empty()) {
                lowest = std::min(lowest, numbers->front());
                highest = std::max(highest, numbers->back());
            }
        }
        const Index span =
            static_cast<Index>(highest) - static_cast<Index>(lowest) + 1;
        _dense_mapped = span <= dense_map_limit(node_count);
        if (_dense_mapped) {
            _dense_base = lowest;
            _dense_node_index.assign(to_sizet(span), -1);
        } else {
            _usr_to_int_node_num.reserve(to_sizet(node_count));
        }

        Index internal_index = 0;
        for (const auto* numbers :
             {&_diff_node_num_vector, &_bound_node_num_vector}) {
            for (const NodeNum node_num : *numbers) {
                static_cast<void>(store_node_index(node_num, internal_index));
                ++internal_index;
            }
        }
    }

    _mapped_version = _structure_version;
}

bool Nodes::store_node_index(NodeNum node_num, Index index) const {
    if (!_dense_mapped) {
        _usr_to_int_node_num[node_num] = index;
        return true;
    }
    const Index offset =
        static_cast<Index>(node_num) - static_cast<Index>(_dense_base);
    if (offset < 0) {
        return false;
    }
    if (offset >= std::ssize(_dense_node_index)) {
        const Index node_count = to_idx(_diff_node_num_vector.size()) +
                                 to_idx(_bound_node_num_vector.size());
        if (offset >= dense_map_limit(node_count)) {
            return false;
        }
        _dense_node_index.resize(to_sizet(offset + 1), -1);
    }
    _dense_node_index[to_sizet(offset)] = static_cast<std::int32_t>(index);
    return true;
}

bool Nodes::shift_boundary_node_indices(Index shift) const {
    return std::ranges::all_of(_bound_node_num_vector, [this, shift](
                                                           NodeNum node_num) {
        const std::optional<Index> index = lookup_node_index(node_num);
        return index.has_value() && store_node_index(node_num, *index + shift);
    });
}

void Nodes::update_map_after_append(char type, std::span<const NodeNum> numbers,
                                    bool map_was_valid) {
    // The structure changed either way; the map follows in place only when
    // it was valid and the new numbers fit it, and is rebuilt on the next
    // lookup otherwise.
    ++_structure_version;
    if (!map_was_valid) {
        return;
    }

    const auto count = to_idx(numbers.size());
    const Index diff_count = to_idx(_diff_node_num_vector.size());
    const Index first_index =
        type == 'D'
            ? diff_count - count
            : diff_count + to_idx(_bound_node_num_vector.size()) - count;

    // A diffusive append moves the whole boundary block down by `count`.
    const bool updated =
        (type != 'D' || shift_boundary_node_indices(count)) &&
        std::ranges::all_of(
            std::views::iota(Index{0}, count), [&](Index position) {
                return store_node_index(numbers[to_sizet(position)],
                                        first_index + position);
            });
    if (updated) {
        _mapped_version = _structure_version;
    }
}

bool Nodes::has_node_number(NodeNum node_num) const {
    return std::ranges::binary_search(_diff_node_num_vector, node_num) ||
           std::ranges::binary_search(_bound_node_num_vector, node_num);
}

std::uint64_t Nodes::structure_version() const noexcept {
    return _structure_version;
}

std::optional<Index> Nodes::get_idx_from_node_num(NodeNum node_num) const {
    Index index = 0;
    if (!find_node_index(node_num, index, nullptr)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Node does not exist");
        return std::nullopt;
    }
    return index;
}

std::optional<NodeNum> Nodes::get_node_num_from_idx(Index idx) const {
    ensure_node_map();
    const auto diff_size = to_idx(_diff_node_num_vector.size());
    const auto bound_size = to_idx(_bound_node_num_vector.size());

    if (idx < 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Node does not exist");
        return std::nullopt;
    }

    if (idx < diff_size) {
        return _diff_node_num_vector[to_sizet(idx)];
    }

    const auto adjusted = idx - diff_size;
    if (adjusted >= 0 && adjusted < bound_size) {
        return _bound_node_num_vector[to_sizet(adjusted)];
    }

    SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Node does not exist");
    return std::nullopt;
}

bool Nodes::is_node(NodeNum node_num) const {
    Index index = 0;
    return find_node_index(node_num, index, nullptr);
}

Node Nodes::get_node_from_node_num(NodeNum node_num) {
    if (is_node(node_num)) {
        return {node_num, _self_pointer};
    } else {
        return Node(NodeNum{-1});
    }
}

Node Nodes::get_node_from_idx(Index idx) {
    const auto node_num = get_node_num_from_idx(idx);
    if (!node_num.has_value()) {
        return Node(NodeNum{-1});
    }

    return {*node_num, _self_pointer};
}

void Nodes::insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Index index, const LiteralString& string) {
    Eigen::SparseVector<LiteralString> result(sparse.size() + 1);
    result.reserve(sparse.nonZeros() + (string.is_empty() ? 0 : 1));

    for (Eigen::SparseVector<LiteralString>::InnerIterator it(sparse); it;
         ++it) {
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
    // In place: only the entries at or after `index` move, so appending at
    // the end costs O(1) amortised (the storage grows geometrically).
    const Index stored = sparse.nonZeros();
    const bool store_value = std::abs(value) > ZERO_THR_ATTR;
    const Index gap = store_value ? 1 : 0;
    sparse.conservativeResize(sparse.size() + 1);
    sparse.data().resize(stored + gap, 1.0);

    const std::span<int> indices(sparse.innerIndexPtr(),
                                 to_sizet(stored + gap));
    const std::span<double> values(sparse.valuePtr(), to_sizet(stored + gap));
    Index position = stored;
    while (position > 0 && indices[to_sizet(position - 1)] >= index) {
        indices[to_sizet(position - 1 + gap)] =
            indices[to_sizet(position - 1)] + 1;
        values[to_sizet(position - 1 + gap)] = values[to_sizet(position - 1)];
        --position;
    }
    if (store_value) {
        indices[to_sizet(position)] = static_cast<int>(index);
        values[to_sizet(position)] = value;
    }
}

void Nodes::delete_displace(Eigen::SparseVector<LiteralString>& sparse,
                            Index index) {
    std::vector<int> indices;
    std::vector<LiteralString> values;
    for (Eigen::SparseVector<LiteralString>::InnerIterator it(sparse); it;
         ++it) {
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
    // Drops the entry at `index` and moves every later one up by one, so the
    // attributes stay aligned with the nodes that remain.
    const auto stored = to_sizet(sparse.nonZeros());
    const std::span<int> indices(sparse.innerIndexPtr(), stored);
    const std::span<double> values(sparse.valuePtr(), stored);
    std::size_t kept = 0;
    for (std::size_t position = 0; position < stored; ++position) {
        if (indices[position] == index) {
            continue;
        }
        indices[kept] = indices[position] > index ? indices[position] - 1
                                                  : indices[position];
        values[kept] = values[position];
        ++kept;
    }
    sparse.data().resize(to_idx(kept));
    sparse.conservativeResize(sparse.size() - 1);
}

std::string Nodes::get_literal_C(int node_num) const {
    return resolve_get_literal_attr(node_num, literals_C);
}

bool Nodes::set_literal_C(NodeNum node_num, const std::string& str) {
    return resolve_set_literal_attr(node_num, literals_C, str);
}

void Nodes::remove_node(NodeNum node_num) {
    auto idx = get_idx_from_node_num(node_num);

    if (!idx.has_value()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "Node {} does not exist.",
                           node_num);
        return;
    }

    const Index diff_count = to_idx(_diff_node_num_vector.size());
    const char type = *idx < diff_count ? 'D' : 'B';
    const Index local_index = type == 'D' ? *idx : *idx - diff_count;
    const Index old_block_size =
        type == 'D' ? diff_count : to_idx(_bound_node_num_vector.size());

    T_vector.erase(T_vector.begin() + *idx);
    C_vector.erase(C_vector.begin() + *idx);

    delete_displace(qs_vector, *idx);
    delete_displace(qa_vector, *idx);
    delete_displace(qe_vector, *idx);
    delete_displace(qi_vector, *idx);
    delete_displace(qr_vector, *idx);
    delete_displace(a_vector, *idx);
    delete_displace(fx_vector, *idx);
    delete_displace(fy_vector, *idx);
    delete_displace(fz_vector, *idx);
    delete_displace(eps_vector, *idx);
    delete_displace(aph_vector, *idx);

    delete_displace(literals_C, *idx);

    // Reshape node vector and conductors
    if (*idx < to_idx(_diff_node_num_vector.size())) {
        _diff_node_num_vector.erase(_diff_node_num_vector.begin() + *idx);
    } else {
        _bound_node_num_vector.erase(
            _bound_node_num_vector.begin() +
            (*idx - to_idx(_diff_node_num_vector.size())));
    }
    ++_structure_version;

    // Every coupling container drops the node's row and column and moves the
    // later nodes up by one.
    if (_first_observer != nullptr) {
        std::vector<Index> old_to_new(to_sizet(old_block_size));
        for (Index old_index = 0; old_index < old_block_size; ++old_index) {
            Index new_index = old_index;
            if (old_index == local_index) {
                new_index = -1;
            } else if (old_index > local_index) {
                new_index = old_index - 1;
            }
            old_to_new[to_sizet(old_index)] = new_index;
        }
        notify_block_remap(type, old_to_new);
    }
}

void Nodes::add_node_insert_idx(Node& node, Index insert_idx) {
    // Info obtained from "node"
    const char type = node.get_type();
    const NodeNum node_num = node.get_node_num();

    const bool map_was_valid = is_mapped();
    const Index diff_count = to_idx(_diff_node_num_vector.size());
    const Index local_index =
        type == 'D' ? insert_idx : insert_idx - diff_count;
    const Index old_block_size =
        type == 'D' ? diff_count : to_idx(_bound_node_num_vector.size());

    if (type == 'D') {
        _diff_node_num_vector.insert(_diff_node_num_vector.begin() + insert_idx,
                                     node_num);
    } else if (type == 'B') {
        _bound_node_num_vector.insert(
            _bound_node_num_vector.begin() +
                (insert_idx - to_idx(_diff_node_num_vector.size())),
            node_num);
    } else {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Wrong node type.");
        return;
    }

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

    if (local_index == old_block_size) {
        // Appended at the end of its block: no stored index moves, so the
        // coupling containers only grow, lazily, and the map follows in place.
        update_map_after_append(type, std::span<const NodeNum>(&node_num, 1U),
                                map_was_valid);
    } else {
        ++_structure_version;
        notify_single_insertion(type, local_index, old_block_size);
    }
}

bool Nodes::is_mapped() const { return _mapped_version == _structure_version; }

std::vector<NodeNum> Nodes::node_numbers() const {
    std::vector<NodeNum> numbers;
    numbers.reserve(_diff_node_num_vector.size() +
                    _bound_node_num_vector.size());
    numbers.insert(numbers.end(), _diff_node_num_vector.begin(),
                   _diff_node_num_vector.end());
    numbers.insert(numbers.end(), _bound_node_num_vector.begin(),
                   _bound_node_num_vector.end());
    return numbers;
}

void Nodes::reserve(Index num_nodes) {
    const auto size = to_sizet(std::max(num_nodes, Index{0}));
    T_vector.reserve(size);
    C_vector.reserve(size);
    _diff_node_num_vector.reserve(size);
}

// ---------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------

void Nodes::add_observer(Couplings* couplings) noexcept {
    couplings->_previous_observer = nullptr;
    couplings->_next_observer = _first_observer;
    if (_first_observer != nullptr) {
        _first_observer->_previous_observer = couplings;
    }
    _first_observer = couplings;
}

void Nodes::remove_observer(Couplings* couplings) noexcept {
    if (couplings->_previous_observer != nullptr) {
        couplings->_previous_observer->_next_observer =
            couplings->_next_observer;
    } else if (_first_observer == couplings) {
        _first_observer = couplings->_next_observer;
    }
    if (couplings->_next_observer != nullptr) {
        couplings->_next_observer->_previous_observer =
            couplings->_previous_observer;
    }
    couplings->_previous_observer = nullptr;
    couplings->_next_observer = nullptr;
}

void Nodes::notify_block_remap(char type, std::span<const Index> old_to_new) {
    for (Couplings* observer = _first_observer; observer != nullptr;
         observer = observer->_next_observer) {
        observer->remap_nodes(type, old_to_new);
    }
}

void Nodes::notify_single_insertion(char type, Index local_index,
                                    Index old_block_size) {
    if (_first_observer == nullptr) {
        return;
    }
    std::vector<Index> old_to_new(to_sizet(old_block_size));
    for (Index old_index = 0; old_index < old_block_size; ++old_index) {
        old_to_new[to_sizet(old_index)] =
            old_index < local_index ? old_index : old_index + 1;
    }
    notify_block_remap(type, old_to_new);
}

// ---------------------------------------------------------------------------
// Type changes
// ---------------------------------------------------------------------------

namespace {

// The internal order after some nodes change type: each block is the nodes
// that stay in it merged, by increasing number, with the ones moving into it;
// the diffusive block first. `numbers` is the old internal order and
// `moving` marks the nodes that change block. Returns the old internal index
// of every new position.
[[nodiscard]] std::vector<Index> retyped_order(
    std::span<const NodeNum> numbers, Index diff_count,
    std::span<const std::uint8_t> moving) {
    const auto num_nodes = to_idx(numbers.size());
    std::vector<Index> new_to_old;
    new_to_old.reserve(numbers.size());
    // Moves a cursor to the next node of [cursor, end) whose mark is `wanted`.
    const auto advance = [&moving](Index& cursor, Index end,
                                   std::uint8_t wanted) {
        while (cursor < end && moving[to_sizet(cursor)] != wanted) {
            ++cursor;
        }
    };
    for (const bool diffusive : {true, false}) {
        Index stay = diffusive ? 0 : diff_count;
        const Index stay_end = diffusive ? diff_count : num_nodes;
        Index come = diffusive ? diff_count : 0;
        const Index come_end = diffusive ? num_nodes : diff_count;
        advance(stay, stay_end, 0U);
        advance(come, come_end, 1U);
        while (stay < stay_end || come < come_end) {
            const bool take_stay =
                come >= come_end ||
                (stay < stay_end &&
                 numbers[to_sizet(stay)] < numbers[to_sizet(come)]);
            if (take_stay) {
                new_to_old.push_back(stay++);
                advance(stay, stay_end, 0U);
            } else {
                new_to_old.push_back(come++);
                advance(come, come_end, 1U);
            }
        }
    }
    return new_to_old;
}

void reorder_dense(std::vector<double>& values,
                   std::span<const Index> new_to_old) {
    std::vector<double> reordered(new_to_old.size());
    std::ranges::transform(
        new_to_old, reordered.begin(),
        [&values](Index old_index) { return values[to_sizet(old_index)]; });
    values.swap(reordered);
}

// A sparse attribute in the new order: where every old index keeps its value,
// then one pass over the new positions, so the cost is linear.
template <typename Scalar>
void reorder_sparse(Eigen::SparseVector<Scalar>& sparse,
                    std::span<const Index> new_to_old) {
    if (sparse.nonZeros() == 0) {
        return;
    }
    std::vector<Index> stored_at(new_to_old.size(), -1);
    for (Index entry = 0; entry < sparse.nonZeros(); ++entry) {
        stored_at[to_sizet(sparse.data().index(entry))] = entry;
    }
    Eigen::SparseVector<Scalar> reordered(sparse.size());
    reordered.reserve(sparse.nonZeros());
    for (Index new_index = 0; new_index < to_idx(new_to_old.size());
         ++new_index) {
        const Index entry =
            stored_at[to_sizet(new_to_old[to_sizet(new_index)])];
        if (entry >= 0) {
            reordered.insertBack(new_index) = sparse.data().value(entry);
        }
    }
    sparse.swap(reordered);
}

}  // namespace

std::vector<std::uint8_t> Nodes::retype_marks(
    std::span<const NodeNum> node_nums, char type, BulkReport& report) const {
    const Index diff_count = to_idx(_diff_node_num_vector.size());
    std::vector<std::uint8_t> moving(
        _diff_node_num_vector.size() + _bound_node_num_vector.size(), 0U);
    for (const NodeNum node_num : node_nums) {
        Index index = 0;
        if (!find_node_index(node_num, index, nullptr)) {
            report.reject("node " + std::to_string(node_num) +
                          " does not exist");
            continue;
        }
        ++report.accepted;
        const bool is_diffusive = index < diff_count;
        if (is_diffusive == (type == 'B')) {
            moving[to_sizet(index)] = 1U;
        }
    }
    return moving;
}

void Nodes::reorder_storage(std::span<const NodeNum> numbers,
                            std::span<const Index> new_to_old,
                            Index diff_count) {
    _diff_node_num_vector.resize(to_sizet(diff_count));
    _bound_node_num_vector.resize(new_to_old.size() - to_sizet(diff_count));
    for (std::size_t new_index = 0; new_index < new_to_old.size();
         ++new_index) {
        const NodeNum number = numbers[to_sizet(new_to_old[new_index])];
        if (std::cmp_less(new_index, diff_count)) {
            _diff_node_num_vector[new_index] = number;
        } else {
            _bound_node_num_vector[new_index - to_sizet(diff_count)] = number;
        }
    }

    reorder_dense(T_vector, new_to_old);
    reorder_dense(C_vector, new_to_old);
    for (auto* sparse :
         {&qs_vector, &qa_vector, &qe_vector, &qi_vector, &qr_vector, &a_vector,
          &fx_vector, &fy_vector, &fz_vector, &eps_vector, &aph_vector}) {
        reorder_sparse(*sparse, new_to_old);
    }
    for (auto* literals :
         {&literals_C, &literals_qs, &literals_qa, &literals_qe, &literals_qi,
          &literals_qr, &literals_a, &literals_fx, &literals_fy, &literals_fz,
          &literals_eps, &literals_aph}) {
        reorder_sparse(*literals, new_to_old);
    }
}

BulkReport Nodes::set_types(std::span<const NodeNum> node_nums, char type) {
    BulkReport report;
    if (type != 'D' && type != 'B') {
        report.rejected = node_nums.size();
        report.first_rejections.emplace_back(std::string("wrong node type '") +
                                             type + "'");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "set_types: {} nodes rejected: {}", report.rejected,
                           report.first_rejections.front());
        return report;
    }
    const std::vector<std::uint8_t> moving =
        retype_marks(node_nums, type, report);
    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "set_types: {} of {} nodes rejected, e.g. {}",
                           report.rejected, node_nums.size(),
                           report.first_rejections.front());
    }
    const auto moved = std::ranges::count(moving, std::uint8_t{1U});
    if (moved == 0) {
        return report;
    }

    // Every coupling container is sized to the current nodes before they
    // move, so its matrices match the old order exactly.
    for (Couplings* observer = _first_observer; observer != nullptr;
         observer = observer->_next_observer) {
        observer->synchronize_structure();
    }
    const Index old_diff_count = to_idx(_diff_node_num_vector.size());
    const Index new_diff_count =
        type == 'D' ? old_diff_count + moved : old_diff_count - moved;
    std::vector<NodeNum> numbers(_diff_node_num_vector);
    numbers.insert(numbers.end(), _bound_node_num_vector.begin(),
                   _bound_node_num_vector.end());
    const std::vector<Index> new_to_old =
        retyped_order(numbers, old_diff_count, moving);
    reorder_storage(numbers, new_to_old, new_diff_count);
    ++_structure_version;

    if (_first_observer != nullptr) {
        std::vector<Index> old_to_new(new_to_old.size());
        for (std::size_t new_index = 0; new_index < new_to_old.size();
             ++new_index) {
            old_to_new[to_sizet(new_to_old[new_index])] = to_idx(new_index);
        }
        for (Couplings* observer = _first_observer; observer != nullptr;
             observer = observer->_next_observer) {
            observer->reorder_nodes(old_to_new, old_diff_count);
        }
    }
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(),
                        "set_types: {} nodes moved to type '{}'", moved, type);
    return report;
}

// ---------------------------------------------------------------------------
// Bulk insertion
// ---------------------------------------------------------------------------

namespace {

// Where the accepted entries of a batch go, in increasing number order.
// `batch_position(j)` is the batch index of the j-th accepted entry and
// `new_local(j)` its position in the node's block after the insertion;
// `old_to_new` maps every node already in the block to its new position, and
// is empty on an append, where nothing moves.
struct BatchPlacement {
    bool identity = true;  // accepted j is batch entry j
    std::vector<std::uint32_t> accepted;
    std::vector<Index> new_local_positions;  // empty on an append
    std::vector<Index> old_to_new;           // empty on an append
    Index count = 0;
    Index old_block_size = 0;
    Index block_start = 0;  // internal index of the block's first node
    Index num_nodes_before = 0;
    bool append = true;

    [[nodiscard]] std::size_t batch_position(Index j) const {
        return identity ? to_sizet(j) : accepted[to_sizet(j)];
    }
    [[nodiscard]] Index new_local(Index j) const {
        return append ? old_block_size + j : new_local_positions[to_sizet(j)];
    }
    [[nodiscard]] Index old_local_to_new(Index old_local) const {
        return append ? old_local : old_to_new[to_sizet(old_local)];
    }
    // Internal index, after the insertion, of a node at `old_index` before.
    [[nodiscard]] Index old_internal_to_new(Index old_index) const {
        if (old_index < block_start) {
            return old_index;
        }
        if (old_index < block_start + old_block_size) {
            return block_start + old_local_to_new(old_index - block_start);
        }
        return old_index + count;
    }
};

// Inserts the batch's values into a dense per-node vector.
void place_dense(std::vector<double>& storage, const BatchPlacement& placement,
                 std::span<const double> column) {
    const Index block_end = placement.block_start + placement.old_block_size;
    const auto value_of = [&](Index j) {
        return column.empty() ? 0.0 : column[placement.batch_position(j)];
    };
    if (placement.append) {
        storage.insert(storage.begin() + block_end, to_sizet(placement.count),
                       0.0);
        for (Index j = 0; j < placement.count; ++j) {
            storage[to_sizet(block_end + j)] = value_of(j);
        }
        return;
    }
    const Index before = placement.num_nodes_before;
    storage.resize(to_sizet(before + placement.count));
    std::move_backward(storage.begin() + block_end, storage.begin() + before,
                       storage.end());
    // Old entries only ever move towards the end, so walking them from the
    // last one down never overwrites one that has not moved yet.
    for (Index old_local = placement.old_block_size - 1; old_local >= 0;
         --old_local) {
        storage[to_sizet(placement.block_start +
                         placement.old_local_to_new(old_local))] =
            storage[to_sizet(placement.block_start + old_local)];
    }
    for (Index j = 0; j < placement.count; ++j) {
        storage[to_sizet(placement.block_start + placement.new_local(j))] =
            value_of(j);
    }
}

// Same for the numbers of the block itself.
void place_numbers(std::vector<NodeNum>& block, const BatchPlacement& placement,
                   std::span<const NodeNum> numbers) {
    if (placement.append) {
        block.reserve(block.size() + to_sizet(placement.count));
        for (Index j = 0; j < placement.count; ++j) {
            block.push_back(numbers[placement.batch_position(j)]);
        }
        return;
    }
    block.resize(to_sizet(placement.old_block_size + placement.count));
    for (Index old_local = placement.old_block_size - 1; old_local >= 0;
         --old_local) {
        block[to_sizet(placement.old_local_to_new(old_local))] =
            block[to_sizet(old_local)];
    }
    for (Index j = 0; j < placement.count; ++j) {
        block[to_sizet(placement.new_local(j))] =
            numbers[placement.batch_position(j)];
    }
}

// Grows a compressed storage to `size` entries, geometrically when it has to
// reallocate, so that many small batches cost amortised O(1) per entry while
// one big batch allocates exactly once.
template <typename Storage>
void grow_storage(Storage& storage, Index size) {
    if (size > storage.allocatedSize()) {
        const Index grown = std::max(
            size, storage.allocatedSize() + (storage.allocatedSize() / 2));
        storage.reserve(grown - storage.size());
    }
    storage.resize(size);
}

// Inserts the batch's values into a sparse per-node vector: one backward
// merge of the stored entries (moved to their new indices) with the batch's
// non-zero values, in place.
void place_sparse(Eigen::SparseVector<double>& storage,
                  const BatchPlacement& placement,
                  std::span<const double> column) {
    const auto stores = [&](Index j) {
        return !column.empty() &&
               std::abs(column[placement.batch_position(j)]) > ZERO_THR_ATTR;
    };
    Index new_entries = 0;
    for (Index j = 0; j < placement.count; ++j) {
        new_entries += stores(j) ? 1 : 0;
    }

    const Index stored = storage.nonZeros();
    storage.conservativeResize(placement.num_nodes_before + placement.count);
    grow_storage(storage.data(), stored + new_entries);
    const std::span<int> indices(storage.innerIndexPtr(),
                                 to_sizet(stored + new_entries));
    const std::span<double> values(storage.valuePtr(),
                                   to_sizet(stored + new_entries));

    Index write = stored + new_entries;
    Index old_read = stored;
    Index next_new = placement.count;
    while (write > 0) {
        while (next_new > 0 && !stores(next_new - 1)) {
            --next_new;
        }
        const Index old_index =
            old_read > 0
                ? placement.old_internal_to_new(indices[to_sizet(old_read - 1)])
                : Index{-1};
        if (next_new == 0 && old_read > 0 &&
            old_index == indices[to_sizet(old_read - 1)] && write == old_read) {
            break;  // everything below is already where it belongs
        }
        const Index new_index =
            next_new > 0
                ? placement.block_start + placement.new_local(next_new - 1)
                : Index{-1};
        --write;
        if (old_index > new_index) {
            indices[to_sizet(write)] = static_cast<int>(old_index);
            values[to_sizet(write)] = values[to_sizet(old_read - 1)];
            --old_read;
        } else {
            indices[to_sizet(write)] = static_cast<int>(new_index);
            values[to_sizet(write)] =
                column[placement.batch_position(next_new - 1)];
            --next_new;
        }
    }
}

}  // namespace

namespace {

// Why a batch cannot be taken at all, if it cannot.
[[nodiscard]] std::optional<std::string> batch_problem(const NodeBatch& batch) {
    if (batch.type != 'D' && batch.type != 'B') {
        return std::string("wrong node type '") + batch.type + "'";
    }
    const std::array<std::span<const double>, 13> columns{
        batch.temperature, batch.capacity, batch.qs, batch.qa, batch.qe,
        batch.qi,          batch.qr,       batch.a,  batch.fx, batch.fy,
        batch.fz,          batch.eps,      batch.aph};
    // The iterator type is a pointer in some standard libraries and a class
    // in others, so the mismatch is tested first and only then read.
    const auto mismatched = [&batch](std::span<const double> column) {
        return !column.empty() && column.size() != batch.numbers.size();
    };
    if (std::ranges::any_of(columns, mismatched)) {
        return "an attribute has " +
               std::to_string(
                   std::ranges::find_if(columns, mismatched)->size()) +
               " values for " + std::to_string(batch.numbers.size()) +
               " node numbers";
    }
    return std::nullopt;
}

// The batch entries to insert, by increasing number: every copy of a number
// repeated in the batch and every number already stored is rejected. The
// batch itself is never copied, only a permutation of it when it is not
// sorted.
[[nodiscard]] BatchPlacement select_entries(
    std::span<const NodeNum> numbers, const std::vector<NodeNum>& block,
    const std::vector<NodeNum>& other_block, BulkReport& report) {
    const auto count = to_idx(numbers.size());
    const bool sorted = std::ranges::adjacent_find(
                            numbers, std::greater_equal<>{}) == numbers.end();
    std::vector<std::uint32_t> order;
    if (!sorted) {
        order.resize(to_sizet(count));
        std::ranges::iota(order, 0U);
        std::ranges::stable_sort(order, {}, [&numbers](std::uint32_t entry) {
            return numbers[entry];
        });
    }
    const auto ordered = [&](Index j) {
        return sorted ? to_sizet(j) : std::size_t{order[to_sizet(j)]};
    };

    BatchPlacement placement;
    placement.identity = sorted;
    std::size_t cursor = 0;
    for (Index j = 0; j < count; ++j) {
        const std::size_t position = ordered(j);
        const NodeNum node_num = numbers[position];
        const bool repeated =
            (j > 0 && numbers[ordered(j - 1)] == node_num) ||
            (j + 1 < count && numbers[ordered(j + 1)] == node_num);
        while (cursor < block.size() && block[cursor] < node_num) {
            ++cursor;
        }
        const bool exists =
            (cursor < block.size() && block[cursor] == node_num) ||
            std::ranges::binary_search(other_block, node_num);
        if (repeated || exists) {
            if (placement.identity) {
                // Everything before j was accepted: materialise it.
                placement.identity = false;
                placement.accepted.resize(to_sizet(j));
                std::ranges::iota(placement.accepted, 0U);
            }
            report.reject("node " + std::to_string(node_num) +
                          (repeated ? " appears more than once in the batch"
                                    : " already exists"));
            continue;
        }
        if (!placement.identity) {
            placement.accepted.push_back(static_cast<std::uint32_t>(position));
        }
    }
    placement.count =
        placement.identity ? count : to_idx(placement.accepted.size());
    return placement;
}

// Where every node of the block and every accepted entry lands: nothing moves
// on an append; otherwise one merge walk of the two sorted sequences.
void plan_positions(BatchPlacement& placement,
                    const std::vector<NodeNum>& block,
                    std::span<const NodeNum> numbers) {
    placement.old_block_size = to_idx(block.size());
    placement.append =
        block.empty() || numbers[placement.batch_position(0)] > block.back();
    if (placement.append) {
        return;
    }
    placement.old_to_new.resize(to_sizet(placement.old_block_size));
    placement.new_local_positions.resize(to_sizet(placement.count));
    Index old_local = 0;
    Index j = 0;
    for (Index merged = 0; merged < placement.old_block_size + placement.count;
         ++merged) {
        const bool take_old =
            j == placement.count ||
            (old_local < placement.old_block_size &&
             block[to_sizet(old_local)] < numbers[placement.batch_position(j)]);
        if (take_old) {
            placement.old_to_new[to_sizet(old_local)] = merged;
            ++old_local;
        } else {
            placement.new_local_positions[to_sizet(j)] = merged;
            ++j;
        }
    }
}

}  // namespace

BulkReport Nodes::add_nodes(const NodeBatch& batch) {
    BulkReport report;
    const char type = batch.type;
    if (const auto problem = batch_problem(batch); problem.has_value()) {
        report.rejected = batch.numbers.size();
        report.first_rejections.push_back(*problem);
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "add_nodes: batch of {} nodes rejected: {}",
                           batch.numbers.size(), *problem);
        return report;
    }

    auto& block = type == 'D' ? _diff_node_num_vector : _bound_node_num_vector;
    BatchPlacement placement = select_entries(
        batch.numbers, block,
        type == 'D' ? _bound_node_num_vector : _diff_node_num_vector, report);
    report.accepted = to_sizet(placement.count);

    if (placement.count > 0) {
        plan_positions(placement, block, batch.numbers);
        placement.block_start =
            type == 'D' ? 0 : to_idx(_diff_node_num_vector.size());
        placement.num_nodes_before = to_idx(T_vector.size());
        const bool map_was_valid = is_mapped();
        place_numbers(block, placement, batch.numbers);
        place_dense(T_vector, placement, batch.temperature);
        place_dense(C_vector, placement, batch.capacity);
        const std::array<
            std::pair<std::span<const double>, Eigen::SparseVector<double>*>,
            11>
            sparse_columns{{{batch.qs, &qs_vector},
                            {batch.qa, &qa_vector},
                            {batch.qe, &qe_vector},
                            {batch.qi, &qi_vector},
                            {batch.qr, &qr_vector},
                            {batch.a, &a_vector},
                            {batch.fx, &fx_vector},
                            {batch.fy, &fy_vector},
                            {batch.fz, &fz_vector},
                            {batch.eps, &eps_vector},
                            {batch.aph, &aph_vector}}};
        for (const auto& [column, storage] : sparse_columns) {
            place_sparse(*storage, placement, column);
        }

        if (placement.append) {
            update_map_after_append(
                type,
                std::span<const NodeNum>(block).last(to_sizet(placement.count)),
                map_was_valid);
        } else {
            ++_structure_version;
            notify_block_remap(type, placement.old_to_new);
        }
    }

    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(),
                        "add_nodes: {} '{}' nodes added, {} rejected",
                        report.accepted, type, report.rejected);
    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "add_nodes: {} of {} nodes rejected, e.g. {}",
                           report.rejected, batch.numbers.size(),
                           report.first_rejections.front());
    }
    return report;
}

// ---------------------------------------------------------------------------
// Bulk getters and setters
// ---------------------------------------------------------------------------

namespace {

// The storage behind one attribute, for a const or non-const Nodes alike.
template <typename NodesType>
[[nodiscard]] auto* dense_storage_of(NodesType& nodes,
                                     NodeAttribute attribute) noexcept {
    decltype(&nodes.T_vector) storage = nullptr;
    if (attribute == NodeAttribute::T) {
        storage = &nodes.T_vector;
    } else if (attribute == NodeAttribute::C) {
        storage = &nodes.C_vector;
    }
    return storage;
}

template <typename NodesType>
[[nodiscard]] auto* sparse_storage_of(NodesType& nodes,
                                      NodeAttribute attribute) noexcept {
    decltype(&nodes.qs_vector) storage = nullptr;
    switch (attribute) {
        case NodeAttribute::QS:
            storage = &nodes.qs_vector;
            break;
        case NodeAttribute::QA:
            storage = &nodes.qa_vector;
            break;
        case NodeAttribute::QE:
            storage = &nodes.qe_vector;
            break;
        case NodeAttribute::QI:
            storage = &nodes.qi_vector;
            break;
        case NodeAttribute::QR:
            storage = &nodes.qr_vector;
            break;
        case NodeAttribute::A:
            storage = &nodes.a_vector;
            break;
        case NodeAttribute::FX:
            storage = &nodes.fx_vector;
            break;
        case NodeAttribute::FY:
            storage = &nodes.fy_vector;
            break;
        case NodeAttribute::FZ:
            storage = &nodes.fz_vector;
            break;
        case NodeAttribute::EPS:
            storage = &nodes.eps_vector;
            break;
        case NodeAttribute::APH:
            storage = &nodes.aph_vector;
            break;
        case NodeAttribute::T:
        case NodeAttribute::C:
            break;
    }
    return storage;
}

}  // namespace

std::vector<double>* Nodes::dense_storage(NodeAttribute attribute) noexcept {
    return dense_storage_of(*this, attribute);
}

const std::vector<double>* Nodes::dense_storage(
    NodeAttribute attribute) const noexcept {
    return dense_storage_of(*this, attribute);
}

Eigen::SparseVector<double>* Nodes::sparse_storage(
    NodeAttribute attribute) noexcept {
    return sparse_storage_of(*this, attribute);
}

const Eigen::SparseVector<double>* Nodes::sparse_storage(
    NodeAttribute attribute) const noexcept {
    return sparse_storage_of(*this, attribute);
}

namespace {

// One merge of a sparse vector's stored entries with new values given as
// (index, value position) sorted by index: a node given twice keeps its last
// value, and a value at or below ZERO_THR_ATTR removes the entry.
void merge_sparse_values(Eigen::SparseVector<double>& sparse,
                         std::span<const std::pair<Index, std::size_t>> targets,
                         std::span<const double> values) {
    const auto stored = to_sizet(sparse.nonZeros());
    const std::span<const int> old_indices(sparse.innerIndexPtr(), stored);
    const std::span<const double> old_values(sparse.valuePtr(), stored);
    std::vector<int> merged_indices;
    std::vector<double> merged_values;
    merged_indices.reserve(stored + targets.size());
    merged_values.reserve(stored + targets.size());
    std::size_t old_read = 0;
    for (std::size_t next = 0; next < targets.size(); ++next) {
        const Index index = targets[next].first;
        if (next + 1 < targets.size() && targets[next + 1].first == index) {
            continue;  // a later value for the same node wins
        }
        while (old_read < stored && old_indices[old_read] < index) {
            merged_indices.push_back(old_indices[old_read]);
            merged_values.push_back(old_values[old_read]);
            ++old_read;
        }
        if (old_read < stored && old_indices[old_read] == index) {
            ++old_read;  // replaced
        }
        const double value = values[targets[next].second];
        if (std::abs(value) > ZERO_THR_ATTR) {
            merged_indices.push_back(static_cast<int>(index));
            merged_values.push_back(value);
        }
    }
    merged_indices.insert(merged_indices.end(),
                          old_indices.begin() + to_idx(old_read),
                          old_indices.end());
    merged_values.insert(merged_values.end(),
                         old_values.begin() + to_idx(old_read),
                         old_values.end());
    sparse.data().resize(to_idx(merged_indices.size()));
    std::ranges::copy(merged_indices, sparse.innerIndexPtr());
    std::ranges::copy(merged_values, sparse.valuePtr());
}

}  // namespace

BulkReport Nodes::set_values(NodeAttribute attribute,
                             std::span<const NodeNum> node_nums,
                             std::span<const double> values) {
    BulkReport report;
    if (node_nums.size() != values.size()) {
        report.rejected = node_nums.size();
        report.first_rejections.push_back(
            std::to_string(values.size()) + " values for " +
            std::to_string(node_nums.size()) + " node numbers");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "set_values: {}",
                           report.first_rejections.front());
        return report;
    }

    ensure_node_map();
    std::vector<std::pair<Index, std::size_t>> targets;  // (index, entry)
    targets.reserve(node_nums.size());
    for (std::size_t entry = 0; entry < node_nums.size(); ++entry) {
        const std::optional<Index> index = lookup_node_index(node_nums[entry]);
        if (!index.has_value()) {
            report.reject("node " + std::to_string(node_nums[entry]) +
                          " does not exist");
            continue;
        }
        targets.emplace_back(*index, entry);
    }
    report.accepted = targets.size();

    if (auto* dense = dense_storage(attribute); dense != nullptr) {
        for (const auto& [index, entry] : targets) {
            (*dense)[to_sizet(index)] = values[entry];
        }
    } else if (auto* sparse = sparse_storage(attribute); sparse != nullptr) {
        std::ranges::stable_sort(
            targets, {}, [](const auto& target) { return target.first; });
        merge_sparse_values(*sparse, targets, values);
    }

    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "set_values: {} of {} nodes skipped, e.g. {}",
                           report.rejected, node_nums.size(),
                           report.first_rejections.front());
    }
    return report;
}

BulkReport Nodes::get_values(NodeAttribute attribute,
                             std::span<const NodeNum> node_nums,
                             std::span<double> values) const {
    BulkReport report;
    if (node_nums.size() != values.size()) {
        report.rejected = node_nums.size();
        report.first_rejections.push_back(
            std::to_string(values.size()) + " outputs for " +
            std::to_string(node_nums.size()) + " node numbers");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "get_values: {}",
                           report.first_rejections.front());
        return report;
    }

    ensure_node_map();
    const auto* dense = dense_storage(attribute);
    const auto* sparse = sparse_storage(attribute);
    for (std::size_t entry = 0; entry < node_nums.size(); ++entry) {
        const std::optional<Index> index = lookup_node_index(node_nums[entry]);
        if (!index.has_value()) {
            values[entry] = std::numeric_limits<double>::quiet_NaN();
            report.reject("node " + std::to_string(node_nums[entry]) +
                          " does not exist");
            continue;
        }
        values[entry] = dense != nullptr ? (*dense)[to_sizet(*index)]
                                         : sparse->coeff(*index);
        ++report.accepted;
    }

    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "get_values: {} of {} nodes do not exist, e.g. {}",
                           report.rejected, node_nums.size(),
                           report.first_rejections.front());
    }
    return report;
}

std::vector<double> Nodes::get_values(NodeAttribute attribute) const {
    if (const auto* dense = dense_storage(attribute); dense != nullptr) {
        return *dense;
    }
    std::vector<double> values(T_vector.size(), 0.0);
    const auto* sparse = sparse_storage(attribute);
    for (Eigen::SparseVector<double>::InnerIterator it(*sparse); it; ++it) {
        values[to_sizet(it.index())] = it.value();
    }
    return values;
}
