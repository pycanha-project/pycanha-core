
#include "pycanha-core/tmm/coupling.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

Coupling::Coupling(int node_1, int node_2, double value) noexcept
    : _conductor(node_1, node_2, value) {}

int Coupling::get_node_1() const noexcept {
    return static_cast<int>(_conductor.row());
}

int Coupling::get_node_2() const noexcept {
    return static_cast<int>(_conductor.col());
}

double Coupling::get_value() const noexcept { return _conductor.value(); }

void Coupling::set_node_1(int node_1) noexcept {
    const int node_2 = _conductor.col();
    const double value = _conductor.value();
    _conductor = Eigen::Triplet<double>(node_1, node_2, value);
}

void Coupling::set_node_2(int node_2) noexcept {
    const int node_1 = _conductor.row();
    const double value = _conductor.value();
    _conductor = Eigen::Triplet<double>(node_1, node_2, value);
}

void Coupling::set_value(double value) noexcept {
    const int node_1 = _conductor.row();
    const int node_2 = _conductor.col();
    _conductor = Eigen::Triplet<double>(node_1, node_2, value);
}