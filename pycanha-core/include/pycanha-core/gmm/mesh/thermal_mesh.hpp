#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/materials/color.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"

namespace pycanha::gmm {

/**
 * @brief Per-primitive thermal discretization with full legacy parity.
 *
 * Carries the UV cut vectors in the two parametric directions plus per-side
 * (side 1 = front, side 2 = back) activity, thickness, color, bulk material
 * and optical material. Adds four int32 fields driving the face -> tmm-node
 * assignment (see node_of()).
 *
 * NOTE on the public surface: there is intentionally no `Side` enum and no
 * `face_id(i, j, side)` accessor. Internal face numbering (even = side 1,
 * odd = side 2) is an implementation detail of the mesher.
 *
 * Validation is enforced on every setter: an invalid state throws
 * std::invalid_argument (same contract as the legacy implementation). A
 * ThermalMesh shared between two GeometryItems yields the SAME per-cell node
 * number on both (the node fields are inputs, not per-item state).
 */
class ThermalMesh {
  public:
    /// Default: unit square, one face pair, all defaults.
    ThermalMesh();

    /// Convenience: build directly from the two cut vectors.
    ThermalMesh(std::vector<double> dir1_mesh, std::vector<double> dir2_mesh);

    // --- UV cuts ---
    [[nodiscard]] std::span<const double> get_dir1_mesh() const noexcept;
    [[nodiscard]] std::span<const double> get_dir2_mesh() const noexcept;
    void set_dir1_mesh(std::vector<double> dir1_mesh);
    void set_dir2_mesh(std::vector<double> dir2_mesh);

    [[nodiscard]] bool is_valid() const noexcept;

    /// (n1 - 1) * (n2 - 1) face pairs.
    [[nodiscard]] MeshIndex get_number_of_pair_faces() const noexcept;

    // --- Per-side metadata (side1 = front, side2 = back) ---
    [[nodiscard]] bool get_side1_activity() const noexcept {
        return _side1_activity;
    }
    [[nodiscard]] bool get_side2_activity() const noexcept {
        return _side2_activity;
    }
    void set_side1_activity(bool activity) noexcept {
        _side1_activity = activity;
    }
    void set_side2_activity(bool activity) noexcept {
        _side2_activity = activity;
    }

    [[nodiscard]] double get_side1_thick() const noexcept {
        return _side1_thick;
    }
    [[nodiscard]] double get_side2_thick() const noexcept {
        return _side2_thick;
    }
    void set_side1_thick(double thick);
    void set_side2_thick(double thick);

    [[nodiscard]] const Color& get_side1_color() const noexcept {
        return _side1_color;
    }
    [[nodiscard]] const Color& get_side2_color() const noexcept {
        return _side2_color;
    }
    void set_side1_color(Color color) noexcept {
        _side1_color = color;
    }
    void set_side2_color(Color color) noexcept {
        _side2_color = color;
    }

    [[nodiscard]] const std::shared_ptr<BulkMaterial>& get_side1_material()
        const noexcept {
        return _side1_material;
    }
    [[nodiscard]] const std::shared_ptr<BulkMaterial>& get_side2_material()
        const noexcept {
        return _side2_material;
    }
    void set_side1_material(std::shared_ptr<BulkMaterial> material) noexcept {
        _side1_material = std::move(material);
    }
    void set_side2_material(std::shared_ptr<BulkMaterial> material) noexcept {
        _side2_material = std::move(material);
    }

    [[nodiscard]] const std::shared_ptr<OpticalMaterial>& get_side1_optical()
        const noexcept {
        return _side1_optical;
    }
    [[nodiscard]] const std::shared_ptr<OpticalMaterial>& get_side2_optical()
        const noexcept {
        return _side2_optical;
    }
    void set_side1_optical(std::shared_ptr<OpticalMaterial> optical) noexcept {
        _side1_optical = std::move(optical);
    }
    void set_side2_optical(std::shared_ptr<OpticalMaterial> optical) noexcept {
        _side2_optical = std::move(optical);
    }

    // --- Per-side tmm-node assignment ---
    // Cell index k for cell (i in dir1, j in dir2): k = i*(n2-1) + j.
    //   node_side1(k) = node1_start + k * node1_step
    //   node_side2(k) = node2_start + k * node2_step
    // node_step == 0 means every face on that side shares the same node.
    // All four default to 0 ("no node assigned").
    [[nodiscard]] std::int32_t get_node1_start() const noexcept {
        return _node1_start;
    }
    [[nodiscard]] std::int32_t get_node1_step() const noexcept {
        return _node1_step;
    }
    [[nodiscard]] std::int32_t get_node2_start() const noexcept {
        return _node2_start;
    }
    [[nodiscard]] std::int32_t get_node2_step() const noexcept {
        return _node2_step;
    }
    void set_node1_start(std::int32_t value) noexcept { _node1_start = value; }
    void set_node1_step(std::int32_t value) noexcept { _node1_step = value; }
    void set_node2_start(std::int32_t value) noexcept { _node2_start = value; }
    void set_node2_step(std::int32_t value) noexcept { _node2_step = value; }

    /// Node number for cell (i, j) on @p side (1 or 2).
    [[nodiscard]] NodeNum node_of(MeshIndex i, MeshIndex j,
                                  unsigned side) const noexcept;

  private:
    void validate() const;

    bool _side1_activity = true;
    bool _side2_activity = true;

    double _side1_thick = 0.0;
    double _side2_thick = 0.0;

    Color _side1_color{0, 127, 255};
    Color _side2_color{127, 0, 255};

    std::shared_ptr<BulkMaterial> _side1_material =
        std::make_shared<BulkMaterial>();
    std::shared_ptr<BulkMaterial> _side2_material =
        std::make_shared<BulkMaterial>();

    std::shared_ptr<OpticalMaterial> _side1_optical =
        std::make_shared<OpticalMaterial>();
    std::shared_ptr<OpticalMaterial> _side2_optical =
        std::make_shared<OpticalMaterial>();

    std::vector<double> _dir1_mesh{0.0, 1.0};
    std::vector<double> _dir2_mesh{0.0, 1.0};

    std::int32_t _node1_start = 0;
    std::int32_t _node1_step = 0;
    std::int32_t _node2_start = 0;
    std::int32_t _node2_step = 0;
};

}  // namespace pycanha::gmm
