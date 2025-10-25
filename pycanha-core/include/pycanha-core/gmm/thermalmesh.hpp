#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/id.hpp"
#include "pycanha-core/gmm/materials.hpp"

namespace pycanha::gmm {
using pycanha::MeshIndex;

class ThermalMesh : public UniqueID {
    // Private members with default values
    bool _side1_activity = true;
    bool _side2_activity = true;

    double _side1_thick = 0.0;
    double _side2_thick = 0.0;

    Color _side1_color = Color(std::array<uint8_t, 3>{0, 127, 255});
    Color _side2_color = Color(std::array<uint8_t, 3>{127, 0, 255});

    std::shared_ptr<BulkMaterial> _side1_material =
        std::make_shared<BulkMaterial>();
    std::shared_ptr<BulkMaterial> _side2_material =
        std::make_shared<BulkMaterial>();

    std::shared_ptr<OpticalMaterial> _side1_optical =
        std::make_shared<OpticalMaterial>();
    std::shared_ptr<OpticalMaterial> _side2_optical =
        std::make_shared<OpticalMaterial>();

    std::vector<double> _dir1_mesh = std::vector<double>{0.0, 1.0};
    std::vector<double> _dir2_mesh = std::vector<double>{0.0, 1.0};

    // Validate function that throws exception.
    void validate() const {
        if (!is_valid()) {
            throw std::invalid_argument("Invalid thermal mesh");
        }
    }

  public:
    // Using default values and test if the mesh is valid
    ThermalMesh() { validate(); }

    // Getters
    [[nodiscard]] bool get_side1_activity() const { return _side1_activity; }
    [[nodiscard]] bool get_side2_activity() const { return _side2_activity; }

    [[nodiscard]] double get_side1_thick() const { return _side1_thick; }
    [[nodiscard]] double get_side2_thick() const { return _side2_thick; }

    [[nodiscard]] Color get_side1_color() const { return _side1_color; }
    [[nodiscard]] Color get_side2_color() const { return _side2_color; }

    [[nodiscard]] std::shared_ptr<BulkMaterial> get_side1_material() const {
        return _side1_material;
    }
    [[nodiscard]] std::shared_ptr<BulkMaterial> get_side2_material() const {
        return _side2_material;
    }

    [[nodiscard]] std::shared_ptr<OpticalMaterial> get_side1_optical() const {
        return _side1_optical;
    }
    [[nodiscard]] std::shared_ptr<OpticalMaterial> get_side2_optical() const {
        return _side2_optical;
    }

    [[nodiscard]] std::vector<double> get_dir1_mesh() const {
        return _dir1_mesh;
    }

    [[nodiscard]] const std::vector<double>* get_dir1_mesh_ptr() const {
        return &_dir1_mesh;
    }

    [[nodiscard]] MeshIndex get_dir1_mesh_size() const {
        return static_cast<MeshIndex>(_dir1_mesh.size());
    }

    [[nodiscard]] std::vector<double> get_dir2_mesh() const {
        return _dir2_mesh;
    }

    [[nodiscard]] const std::vector<double>* get_dir2_mesh_ptr() const {
        return &_dir2_mesh;
    }

    [[nodiscard]] MeshIndex get_dir2_mesh_size() const {
        return static_cast<MeshIndex>(_dir2_mesh.size());
    }
    // Setters
    void set_side1_activity(bool side1_activity) {
        _side1_activity = side1_activity;
        validate();
    }
    void set_side2_activity(bool side2_activity) {
        _side2_activity = side2_activity;
        validate();
    }

    void set_side1_thick(double side1_thick) {
        _side1_thick = side1_thick;
        validate();
    }
    void set_side2_thick(double side2_thick) {
        _side2_thick = side2_thick;
        validate();
    }

    // cppcheck-suppress passedByValue
    void set_side1_color(Color side1_color) {
        _side1_color = side1_color;
        validate();
    }

    // cppcheck-suppress passedByValue
    void set_side2_color(Color side2_color) {
        _side2_color = side2_color;
        validate();
    }

    void set_side1_material(std::shared_ptr<BulkMaterial> side1_material) {
        _side1_material = std::move(side1_material);
        validate();
    }

    void set_side2_material(std::shared_ptr<BulkMaterial> side2_material) {
        _side2_material = std::move(side2_material);
        validate();
    }

    void set_side1_optical(std::shared_ptr<OpticalMaterial> side1_optical) {
        _side1_optical = std::move(side1_optical);
        validate();
    }

    void set_side2_optical(std::shared_ptr<OpticalMaterial> side2_optical) {
        _side2_optical = std::move(side2_optical);
        validate();
    }

    void set_dir1_mesh(std::vector<double> dir1_mesh) {
        _dir1_mesh = std::move(dir1_mesh);
        validate();
    }

    void set_dir2_mesh(std::vector<double> dir2_mesh) {
        _dir2_mesh = std::move(dir2_mesh);
        validate();
    }
    /**
     * @brief Check if the mesh is valid
     * @return True if the mesh is valid, false otherwise
     */

    [[nodiscard]] bool is_valid() const {
        // Check that dir1_mesh and dir2_mesh have at least 2 elements, that
        // they start at 0 and end at 1.0 and that they are sorted.
        // We also check that 2 * dir1_mesh_size * dir2_mesh_size is less than
        // MeshIndex (uint32_t) max value to avoid overflow.
        const bool mesh_valid =
            (2 * _dir1_mesh.size() * _dir2_mesh.size() <=
             std::numeric_limits<MeshIndex>::max()) &&
            (_dir1_mesh.size() >= 2) && (_dir2_mesh.size() >= 2) &&
            (std::abs(_dir1_mesh[0]) <= LENGTH_TOL) &&
            (std::abs(_dir1_mesh[_dir1_mesh.size() - 1] - 1.0) <= LENGTH_TOL) &&
            (std::abs(_dir2_mesh[0]) <= 0.0) &&
            (std::abs(_dir2_mesh[_dir2_mesh.size() - 1] - 1.0) <= LENGTH_TOL) &&
            (std::is_sorted(_dir1_mesh.begin(), _dir1_mesh.end())) &&
            (std::is_sorted(_dir2_mesh.begin(), _dir2_mesh.end()));

        return mesh_valid;
    }

    /**
     * @brief Return number of pair of faces
     * @return Number of pair of faces
     */
    [[nodiscard]] MeshIndex get_number_of_pair_faces() const {
        return static_cast<MeshIndex>((_dir1_mesh.size() - 1) *
                                      (_dir2_mesh.size() - 1));
    }
};
}  // namespace pycanha::gmm
