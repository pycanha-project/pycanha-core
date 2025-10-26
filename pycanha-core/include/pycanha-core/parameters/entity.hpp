
#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha {

namespace detail {

template <typename Key, typename Value, std::size_t Size>
struct EntityMap {
    std::array<std::pair<Key, Value>, Size> data;

    [[nodiscard]] constexpr Value at(const Key& key) const {
        const auto it = std::find_if(
            data.begin(), data.end(),
            [&key](const auto& pair) { return pair.first == key; });
        if (it == data.end()) {
            throw std::out_of_range("Unknown thermal entity attribute: " +
                                    std::string(key));
        }
        return it->second;
    }
};

using ValueAccessor = double* (*)(ThermalNetwork&, int, int);

using std::literals::string_view_literals::operator""sv;

constexpr std::array<std::pair<std::string_view, ValueAccessor>, 9>
    attribute_table{{
        {"T"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_T_value_ref(node_1);
         }},
        {"C"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_C_value_ref(node_1);
         }},
        {"QS"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_qs_value_ref(node_1);
         }},
        {"QE"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_qe_value_ref(node_1);
         }},
        {"QA"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_qa_value_ref(node_1);
         }},
        {"QI"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_qi_value_ref(node_1);
         }},
        {"QR"sv,
         [](ThermalNetwork& network, int node_1, int /*unused*/) {
             return network.nodes().get_qr_value_ref(node_1);
         }},
        {"GL"sv,
         [](ThermalNetwork& network, int node_1, int node_2) {
             return network.conductive_couplings().get_coupling_value_ref(
                 node_1, node_2);
         }},
        {"GR"sv,
         [](ThermalNetwork& network, int node_1, int node_2) {
             return network.radiative_couplings().get_coupling_value_ref(
                 node_1, node_2);
         }},
    }};

constexpr EntityMap<std::string_view, ValueAccessor, attribute_table.size()>
    attribute_lookup{{attribute_table}};

[[nodiscard]] inline ValueAccessor resolve_accessor(const std::string& type) {
    return attribute_lookup.at(type);
}

}  // namespace detail

class ThermalEntity {
  public:
    ThermalEntity(const ThermalEntity&) = default;
    ThermalEntity& operator=(const ThermalEntity&) = default;
    ThermalEntity(ThermalEntity&&) noexcept = default;
    ThermalEntity& operator=(ThermalEntity&&) noexcept = default;
    virtual ~ThermalEntity() = default;

    [[nodiscard]] const std::string& type() const noexcept { return _type; }
    [[nodiscard]] int node_index_1() const noexcept { return _node_1; }
    [[nodiscard]] int node_index_2() const noexcept { return _node_2; }

    bool operator==(const ThermalEntity& other) const {
        return string_representation() == other.string_representation();
    }

    [[nodiscard]] bool is_same_as(const ThermalEntity& other) const {
        return *this == other;
    }

    [[nodiscard]] virtual std::string string_representation() const = 0;
    [[nodiscard]] virtual double get_value() = 0;
    [[nodiscard]] virtual double* get_value_ref() = 0;
    virtual void set_value(double value) = 0;

    [[nodiscard]] virtual std::unique_ptr<ThermalEntity> clone() const = 0;

  protected:
    ThermalEntity(ThermalNetwork& network, std::string type)
        : _network(&network), _type(std::move(type)) {
        if (_network == nullptr) {
            throw std::invalid_argument("ThermalEntity requires a network");
        }
    }

    [[nodiscard]] ThermalNetwork& network() noexcept { return *_network; }
    [[nodiscard]] const ThermalNetwork& network() const noexcept {
        return *_network;
    }

    void set_nodes(int node_1, int node_2 = -1) noexcept {
        _node_1 = node_1;
        _node_2 = node_2;
    }

    [[nodiscard]] detail::ValueAccessor accessor() const {
        return detail::resolve_accessor(_type);
    }

  private:
    ThermalNetwork* _network{};
    std::string _type;
    int _node_1{-1};
    int _node_2{-1};
};

class AttributeEntity final : public ThermalEntity {
  public:
    AttributeEntity(ThermalNetwork& network, std::string type, int node)
        : ThermalEntity(network, std::move(type)) {
        set_nodes(node);
    }

    [[nodiscard]] std::string string_representation() const override {
        return type() + std::to_string(node_index_1());
    }

    double get_value() override {
        auto* value_ptr = accessor()(network(), node_index_1(), -1);
        if (value_ptr == nullptr) {
            throw std::runtime_error("AttributeEntity has no value pointer");
        }
        return *value_ptr;
    }

    double* get_value_ref() override {
        auto* value_ptr = accessor()(network(), node_index_1(), -1);
        if (value_ptr == nullptr) {
            throw std::runtime_error("AttributeEntity has no value pointer");
        }
        return value_ptr;
    }

    void set_value(double value) override {
        auto* value_ptr = get_value_ref();
        *value_ptr = value;
    }

    [[nodiscard]] std::unique_ptr<ThermalEntity> clone() const override {
        return std::make_unique<AttributeEntity>(*this);
    }
};

class CouplingEntity : public ThermalEntity {
  public:
    CouplingEntity(const CouplingEntity&) = default;
    CouplingEntity& operator=(const CouplingEntity&) = default;
    CouplingEntity(CouplingEntity&&) noexcept = default;
    CouplingEntity& operator=(CouplingEntity&&) noexcept = default;
    ~CouplingEntity() override = default;

    [[nodiscard]] std::string string_representation() const override {
        return type() + "(" + std::to_string(node_index_1()) + ", " +
               std::to_string(node_index_2()) + ")";
    }

    double get_value() override {
        auto* value_ptr = accessor()(network(), node_index_1(), node_index_2());
        if (value_ptr == nullptr) {
            throw std::runtime_error("CouplingEntity has no value pointer");
        }
        return *value_ptr;
    }

    double* get_value_ref() override {
        auto* value_ptr = accessor()(network(), node_index_1(), node_index_2());
        if (value_ptr == nullptr) {
            throw std::runtime_error("CouplingEntity has no value pointer");
        }
        return value_ptr;
    }

    void set_value(double value) override {
        auto* value_ptr = get_value_ref();
        *value_ptr = value;
    }

  protected:
    CouplingEntity(ThermalNetwork& network, std::string type, int node_1,
                   int node_2)
        : ThermalEntity(network, std::move(type)) {
        const auto [first, second] = std::minmax(node_1, node_2);
        set_nodes(first, second);
    }
};

class ConductiveCouplingEntity final : public CouplingEntity {
  public:
    ConductiveCouplingEntity(ThermalNetwork& network, int node_1, int node_2)
        : CouplingEntity(network, "GL", node_1, node_2) {}

    double get_value() override {
        return network().conductive_couplings().get_coupling_value(
            node_index_1(), node_index_2());
    }

    double* get_value_ref() override {
        auto* value_ptr =
            network().conductive_couplings().get_coupling_value_ref(
                node_index_1(), node_index_2());
        if (value_ptr == nullptr) {
            throw std::runtime_error(
                "ConductiveCouplingEntity has no value pointer");
        }
        return value_ptr;
    }

    void set_value(double value) override {
        network().conductive_couplings().set_coupling_value(
            node_index_1(), node_index_2(), value);
    }

    [[nodiscard]] std::unique_ptr<ThermalEntity> clone() const override {
        return std::make_unique<ConductiveCouplingEntity>(*this);
    }
};

class RadiativeCouplingEntity final : public CouplingEntity {
  public:
    RadiativeCouplingEntity(ThermalNetwork& network, int node_1, int node_2)
        : CouplingEntity(network, "GR", node_1, node_2) {}

    double get_value() override {
        return network().radiative_couplings().get_coupling_value(
            node_index_1(), node_index_2());
    }

    double* get_value_ref() override {
        auto* value_ptr =
            network().radiative_couplings().get_coupling_value_ref(
                node_index_1(), node_index_2());
        if (value_ptr == nullptr) {
            throw std::runtime_error(
                "RadiativeCouplingEntity has no value pointer");
        }
        return value_ptr;
    }

    void set_value(double value) override {
        network().radiative_couplings().set_coupling_value(
            node_index_1(), node_index_2(), value);
    }

    [[nodiscard]] std::unique_ptr<ThermalEntity> clone() const override {
        return std::make_unique<RadiativeCouplingEntity>(*this);
    }
};

}  // namespace pycanha
