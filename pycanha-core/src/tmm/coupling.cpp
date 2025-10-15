
#include "pycanha-core/tmm/coupling.hpp"

#include "pycanha-core/parameters.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

Coupling::Coupling(Index node_1, Index node_2, double value) noexcept
    : _conductor(node_1, node_2, value) {}

Index Coupling::get_node_1() const noexcept { return _conductor.row(); }

Index Coupling::get_node_2() const noexcept { return _conductor.col(); }

double Coupling::get_value() const noexcept { return _conductor.value(); }

void Coupling::set_node_1(Index node_1) noexcept {
    const Index node_2 = _conductor.col();
    const double value = _conductor.value();
    _conductor = Eigen::Triplet<double, Index>(node_1, node_2, value);
}

void Coupling::set_node_2(Index node_2) noexcept {
    const Index node_1 = _conductor.row();
    const double value = _conductor.value();
    _conductor = Eigen::Triplet<double, Index>(node_1, node_2, value);
}

void Coupling::set_value(double value) noexcept {
    const Index node_1 = _conductor.row();
    const Index node_2 = _conductor.col();
    _conductor = Eigen::Triplet<double, Index>(node_1, node_2, value);
}