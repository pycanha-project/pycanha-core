
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

enum class EntityType : std::uint8_t {
    t,
    c,
    qs,
    qa,
    qe,
    qi,
    qr,
    gl,
    gr,
};

namespace detail {

using GetValueFn = double (*)(ThermalNetwork&, NodeNum, NodeNum);
using GetValueRefFn = double* (*)(ThermalNetwork&, NodeNum, NodeNum);
using SetValueFn = bool (*)(ThermalNetwork&, NodeNum, NodeNum, double);
using ExistsFn = bool (*)(ThermalNetwork&, NodeNum, NodeNum);

struct EntityOps {
    std::string_view token;
    std::uint8_t node_count;
    bool writable;
    GetValueFn get_value;
    GetValueRefFn get_value_ref;
    SetValueFn set_value;
    ExistsFn exists;
};

[[nodiscard]] inline bool node_exists(ThermalNetwork& network, NodeNum node_1,
                                      NodeNum /*unused*/) {
    return network.nodes().get_idx_from_node_num(node_1).has_value();
}

[[nodiscard]] inline bool conductive_coupling_exists(ThermalNetwork& network,
                                                     NodeNum node_1,
                                                     NodeNum node_2) {
    return network.conductive_couplings().get_coupling_value_ref(
               node_1, node_2) != nullptr;
}

[[nodiscard]] inline bool radiative_coupling_exists(ThermalNetwork& network,
                                                    NodeNum node_1,
                                                    NodeNum node_2) {
    return network.radiative_couplings().get_coupling_value_ref(
               node_1, node_2) != nullptr;
}

using std::literals::string_view_literals::operator""sv;

constexpr std::array<EntityOps, 9> entity_ops_table{{
    {"T"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_T(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_T_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_T(node_1, value); },
     node_exists},
    {"C"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_C(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_C_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_C(node_1, value); },
     node_exists},
    {"QS"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qs(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qs_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_qs(node_1, value); },
     node_exists},
    {"QA"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qa(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qa_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_qa(node_1, value); },
     node_exists},
    {"QE"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qe(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qe_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_qe(node_1, value); },
     node_exists},
    {"QI"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qi(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qi_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_qi(node_1, value); },
     node_exists},
    {"QR"sv, 1U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qr(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/) {
         return network.nodes().get_qr_value_ref(node_1);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum /*unused*/,
        double value) { return network.nodes().set_qr(node_1, value); },
     node_exists},
    {"GL"sv, 2U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2) {
         return network.conductive_couplings().get_coupling_value(node_1,
                                                                  node_2);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2) {
         return network.conductive_couplings().get_coupling_value_ref(node_1,
                                                                      node_2);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2, double value) {
         if (network.conductive_couplings().get_coupling_value_ref(
                 node_1, node_2) == nullptr) {
             return false;
         }
         network.conductive_couplings().set_coupling_value(node_1, node_2,
                                                           value);
         return true;
     },
     conductive_coupling_exists},
    {"GR"sv, 2U, true,
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2) {
         return network.radiative_couplings().get_coupling_value(node_1,
                                                                 node_2);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2) {
         return network.radiative_couplings().get_coupling_value_ref(node_1,
                                                                     node_2);
     },
     [](ThermalNetwork& network, NodeNum node_1, NodeNum node_2, double value) {
         if (network.radiative_couplings().get_coupling_value_ref(
                 node_1, node_2) == nullptr) {
             return false;
         }
         network.radiative_couplings().set_coupling_value(node_1, node_2,
                                                          value);
         return true;
     },
     radiative_coupling_exists},
}};

[[nodiscard]] constexpr const EntityOps& entity_ops(EntityType type) {
    return entity_ops_table[std::to_underlying(type)];
}

[[nodiscard]] inline std::optional<EntityType> lookup_entity_type(
    std::string_view token) {
    for (std::size_t index = 0; index < entity_ops_table.size(); ++index) {
        if (entity_ops_table[index].token == token) {
            return static_cast<EntityType>(index);
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline std::string normalize_entity_text(std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());

    for (const char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            continue;
        }

        normalized.push_back(
            static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }

    return normalized;
}

[[nodiscard]] inline std::optional<NodeNum> parse_node_num(
    std::string_view text) {
    NodeNum value = 0;
    const auto* begin = text.data();
    const auto* end = begin + text.size();
    const auto [ptr, err] = std::from_chars(begin, end, value);
    if (err != std::errc{} || ptr != end) {
        return std::nullopt;
    }
    return value;
}

}  // namespace detail

class Entity {
  public:
    static constexpr NodeNum invalid_node = NodeNum{-1};

    Entity() = default;
    Entity(const Entity&) = default;
    Entity& operator=(const Entity&) = default;
    Entity(Entity&&) noexcept = default;
    Entity& operator=(Entity&&) noexcept = default;
    ~Entity() = default;

    [[nodiscard]] static Entity make(ThermalNetwork& network, EntityType type,
                                     NodeNum node_1 = invalid_node,
                                     NodeNum node_2 = invalid_node) {
        Entity entity;
        entity._network = &network;
        entity._type = type;
        entity._ops = &detail::entity_ops(type);

        if (entity._ops->node_count == 2U && node_1 > node_2) {
            std::swap(node_1, node_2);
        }

        if (entity._ops->node_count == 0U) {
            node_1 = invalid_node;
            node_2 = invalid_node;
        } else if (entity._ops->node_count == 1U) {
            node_2 = invalid_node;
        }

        entity._node_1 = node_1;
        entity._node_2 = node_2;
        return entity;
    }

    [[nodiscard]] static std::optional<Entity> from_string(
        ThermalNetwork& network, std::string_view text) {
        const std::string normalized = detail::normalize_entity_text(text);
        if (normalized.empty()) {
            return std::nullopt;
        }

        if (const auto open = normalized.find('('); open != std::string::npos) {
            const auto comma = normalized.find(',', open + 1U);
            const auto close = normalized.find(
                ')', comma == std::string::npos ? open + 1U : comma + 1U);
            if (comma == std::string::npos || close == std::string::npos ||
                close != normalized.size() - 1U) {
                return std::nullopt;
            }

            const auto parsed_type = detail::lookup_entity_type(
                std::string_view(normalized).substr(0U, open));
            if (!parsed_type.has_value() ||
                detail::entity_ops(*parsed_type).node_count != 2U) {
                return std::nullopt;
            }

            const auto parsed_node_1 = detail::parse_node_num(
                std::string_view(normalized)
                    .substr(open + 1U, comma - open - 1U));
            const auto parsed_node_2 = detail::parse_node_num(
                std::string_view(normalized)
                    .substr(comma + 1U, close - comma - 1U));
            if (!parsed_node_1.has_value() || !parsed_node_2.has_value()) {
                return std::nullopt;
            }

            return make(network, *parsed_type, *parsed_node_1, *parsed_node_2);
        }

        std::size_t token_length = 0U;
        while (token_length < normalized.size() &&
               std::isalpha(
                   static_cast<unsigned char>(normalized[token_length])) != 0) {
            ++token_length;
        }

        const auto parsed_type = detail::lookup_entity_type(
            std::string_view(normalized).substr(0U, token_length));
        if (!parsed_type.has_value()) {
            return std::nullopt;
        }

        const auto expected_node_count =
            detail::entity_ops(*parsed_type).node_count;
        if (token_length == normalized.size()) {
            return expected_node_count == 0U
                       ? std::optional<Entity>(make(network, *parsed_type))
                       : std::nullopt;
        }

        if (expected_node_count != 1U) {
            return std::nullopt;
        }

        const auto parsed_node = detail::parse_node_num(
            std::string_view(normalized).substr(token_length));
        if (!parsed_node.has_value()) {
            return std::nullopt;
        }

        return make(network, *parsed_type, *parsed_node);
    }

    [[nodiscard]] static Entity t(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::t, node);
    }

    [[nodiscard]] static Entity c(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::c, node);
    }

    [[nodiscard]] static Entity qs(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::qs, node);
    }

    [[nodiscard]] static Entity qa(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::qa, node);
    }

    [[nodiscard]] static Entity qe(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::qe, node);
    }

    [[nodiscard]] static Entity qi(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::qi, node);
    }

    [[nodiscard]] static Entity qr(ThermalNetwork& network, NodeNum node) {
        return make(network, EntityType::qr, node);
    }

    [[nodiscard]] static Entity gl(ThermalNetwork& network, NodeNum node_1,
                                   NodeNum node_2) {
        return make(network, EntityType::gl, node_1, node_2);
    }

    [[nodiscard]] static Entity gr(ThermalNetwork& network, NodeNum node_1,
                                   NodeNum node_2) {
        return make(network, EntityType::gr, node_1, node_2);
    }

    [[nodiscard]] static Entity temperature(ThermalNetwork& network,
                                            NodeNum node) {
        return t(network, node);
    }

    [[nodiscard]] static Entity capacity(ThermalNetwork& network,
                                         NodeNum node) {
        return c(network, node);
    }

    [[nodiscard]] static Entity solar_heat(ThermalNetwork& network,
                                           NodeNum node) {
        return qs(network, node);
    }

    [[nodiscard]] static Entity albedo_heat(ThermalNetwork& network,
                                            NodeNum node) {
        return qa(network, node);
    }

    [[nodiscard]] static Entity earth_ir(ThermalNetwork& network,
                                         NodeNum node) {
        return qe(network, node);
    }

    [[nodiscard]] static Entity internal_heat(ThermalNetwork& network,
                                              NodeNum node) {
        return qi(network, node);
    }

    [[nodiscard]] static Entity other_heat(ThermalNetwork& network,
                                           NodeNum node) {
        return qr(network, node);
    }

    [[nodiscard]] static Entity conductive(ThermalNetwork& network,
                                           NodeNum node_1, NodeNum node_2) {
        return gl(network, node_1, node_2);
    }

    [[nodiscard]] static Entity radiative(ThermalNetwork& network,
                                          NodeNum node_1, NodeNum node_2) {
        return gr(network, node_1, node_2);
    }

    [[nodiscard]] EntityType type() const noexcept { return _type; }
    [[nodiscard]] std::string_view token() const noexcept {
        return _ops != nullptr ? _ops->token : std::string_view{};
    }
    [[nodiscard]] NodeNum node_1() const noexcept { return _node_1; }
    [[nodiscard]] NodeNum node_2() const noexcept { return _node_2; }
    [[nodiscard]] int node_count() const noexcept {
        return _ops != nullptr ? static_cast<int>(_ops->node_count) : 0;
    }
    [[nodiscard]] bool writable() const noexcept {
        return _ops != nullptr && _ops->writable;
    }

    bool operator==(const Entity& other) const noexcept {
        return _network == other._network && _type == other._type &&
               _node_1 == other._node_1 && _node_2 == other._node_2;
    }

    [[nodiscard]] bool is_same_as(const Entity& other) const noexcept {
        return *this == other;
    }

    [[nodiscard]] bool exists() const {
        return _network != nullptr && _ops != nullptr &&
               _ops->exists(*_network, _node_1, _node_2);
    }

    [[nodiscard]] double get_value() const {
        if (_network == nullptr || _ops == nullptr) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return _ops->get_value(*_network, _node_1, _node_2);
    }

    [[nodiscard]] double* get_value_ref() const {
        if (_network == nullptr || _ops == nullptr || !_ops->writable) {
            return nullptr;
        }

        return _ops->get_value_ref(*_network, _node_1, _node_2);
    }

    [[nodiscard]] bool set_value(double value) const {
        if (_network == nullptr || _ops == nullptr || !_ops->writable) {
            return false;
        }

        return _ops->set_value(*_network, _node_1, _node_2, value);
    }

    [[nodiscard]] std::string string_representation() const {
        if (_ops == nullptr) {
            return {};
        }

        if (_ops->node_count == 0U) {
            return std::string(_ops->token);
        }

        if (_ops->node_count == 1U) {
            return std::string(_ops->token) + std::to_string(_node_1);
        }

        return std::string(_ops->token) + "(" + std::to_string(_node_1) + "," +
               std::to_string(_node_2) + ")";
    }

    struct Hash {
        [[nodiscard]] std::size_t operator()(const Entity& entity) const {
            std::size_t seed =
                std::hash<const ThermalNetwork*>()(entity._network);
            seed ^= static_cast<std::size_t>(entity._type) + 0x9e3779b9U +
                    (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<NodeNum>()(entity._node_1) + 0x9e3779b9U +
                    (seed << 6U) + (seed >> 2U);
            seed ^= std::hash<NodeNum>()(entity._node_2) + 0x9e3779b9U +
                    (seed << 6U) + (seed >> 2U);
            return seed;
        }
    };

  private:
    ThermalNetwork* _network{nullptr};
    EntityType _type{EntityType::t};
    NodeNum _node_1{invalid_node};
    NodeNum _node_2{invalid_node};
    const detail::EntityOps* _ops{nullptr};
};

}  // namespace pycanha
