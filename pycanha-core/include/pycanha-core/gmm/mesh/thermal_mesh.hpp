#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/materials/color.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"

namespace pycanha::gmm {

/**
 * @brief Which sides of a shell take part in one physics.
 *
 * The four states are STEP-TAS's `mgm_active_side_type`. A ThermalMesh carries
 * two independent selectors, one for radiation and one for conduction, so a
 * side can radiate, conduct, do both, or do neither. That pair spans exactly
 * the four activity values ESATAN uses ("Active", "Inactive", "Radiative",
 * "Conductive"), which a single selector could not express.
 *
 * The underlying values are a bitmask over the two sides (side 1 = bit 0,
 * side 2 = bit 1), which is what makes active_side_includes() a bit test.
 */
enum class ActiveSide : std::uint8_t {
    None = 0,
    Side1 = 1,
    Side2 = 2,
    Both = 3,
};

/// True if @p side (1 or 2) is selected by @p sides. Any other side is false.
[[nodiscard]] constexpr bool active_side_includes(ActiveSide sides,
                                                  unsigned side) noexcept {
    if (side != 1U && side != 2U) {
        return false;
    }
    const auto mask = static_cast<std::uint8_t>(1U << (side - 1U));
    return (static_cast<std::uint8_t>(sides) & mask) != 0U;
}

/**
 * @brief Per-primitive thermal discretization with full legacy parity.
 *
 * Carries the UV cut vectors in the two parametric directions plus the two
 * per-physics active-side selectors and, per side (side 1 = front, side 2 =
 * back), thickness, color, bulk material and optical material. Adds four int32
 * fields driving the face -> tmm-node assignment (see node_of()).
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

    // --- Activity, one selector per physics ---
    // Radiative activity gates the optical/view-factor path; conductive
    // activity gates conductor generation. They are independent: neither
    // implies the other. Taking part in either is what makes a side exist for
    // the tmm, so that is what gates its nodes and their capacitance.
    [[nodiscard]] ActiveSide get_radiative_active_side() const noexcept {
        return _radiative_active_side;
    }
    [[nodiscard]] ActiveSide get_conductive_active_side() const noexcept {
        return _conductive_active_side;
    }
    void set_radiative_active_side(ActiveSide sides) noexcept {
        _radiative_active_side = sides;
    }
    void set_conductive_active_side(ActiveSide sides) noexcept {
        _conductive_active_side = sides;
    }

    /// Per-side predicates. @p side must be 1 or 2, else std::invalid_argument.
    [[nodiscard]] bool is_radiative_active(unsigned side) const;
    [[nodiscard]] bool is_conductive_active(unsigned side) const;
    /// True if the side takes part in either physics — the "this side of the
    /// shell exists at all" test.
    [[nodiscard]] bool is_side_active(unsigned side) const;

    // --- Per-side metadata (side1 = front, side2 = back) ---

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
    void set_side1_color(const Color& color) noexcept { _side1_color = color; }
    void set_side2_color(const Color& color) noexcept { _side2_color = color; }

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
    // Cell index k for cell (i in dir1, j in dir2): k = i + j*(n1-1), i.e.
    // direction 1 varies fastest. That is the order STEP-TAS lists a meshed
    // surface's faces in, so k is also the face's index in an exchanged model.
    //   node_side1(k) = node1_start + k * node1_step
    //   node_side2(k) = node2_start + k * node2_step
    // node_step == 0 means every face on that side shares the same node.
    // The start fields default to NO_NODE (-1) and the steps to 0, so an
    // unconfigured ThermalMesh maps every face to "no node assigned". Note 0
    // is a legal user node number; only NO_NODE means unassigned.
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
    /// Throws std::invalid_argument if side is not 1/2 or if (i, j) is outside
    /// the cell grid ((dir1-1) x (dir2-1)).
    [[nodiscard]] NodeNum node_of(MeshIndex i, MeshIndex j,
                                  unsigned side) const;

  private:
    void validate() const;

    ActiveSide _radiative_active_side = ActiveSide::Both;
    ActiveSide _conductive_active_side = ActiveSide::Both;

    double _side1_thick = 0.0;
    double _side2_thick = 0.0;

    Color _side1_color{0, 127, 255};
    Color _side2_color{127, 0, 255};

    // Materials are nullable: nullptr = "no material assigned". Presence is
    // validated later by the thermal-analysis layer (a nullptr on an active
    // side is an error there), not here. Keeps ThermalMesh{} cheap (no
    // per-mesh material allocation) and makes "unassigned" explicit.
    std::shared_ptr<BulkMaterial> _side1_material;
    std::shared_ptr<BulkMaterial> _side2_material;

    std::shared_ptr<OpticalMaterial> _side1_optical;
    std::shared_ptr<OpticalMaterial> _side2_optical;

    std::vector<double> _dir1_mesh{0.0, 1.0};
    std::vector<double> _dir2_mesh{0.0, 1.0};

    std::int32_t _node1_start = NO_NODE;
    std::int32_t _node1_step = 0;
    std::int32_t _node2_start = NO_NODE;
    std::int32_t _node2_step = 0;
};

}  // namespace pycanha::gmm
