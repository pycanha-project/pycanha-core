#include "pycanha-core/io/esatan.hpp"

// HDF5 C++ API symbols are provided through umbrella headers/macros.
// NOLINTBEGIN(misc-include-cleaner)
#include <H5Cpp.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {
namespace {

constexpr std::size_t k_num_real_node_attrs = 16;

enum class RealNodeAttrIndex : std::uint8_t {
    T = 0,
    C = 1,
    QA = 2,
    QE = 3,
    QI = 4,
    QR = 5,
    QS = 6,
    A = 7,
    APH = 8,
    EPS = 9,
    FX = 13,
    FY = 14,
    FZ = 15,
};

void require_rank(const H5::DataSet& dataset, int expected_rank,
                  const std::string& dataset_name) {
    const int rank = dataset.getSpace().getSimpleExtentNdims();
    if (rank != expected_rank) {
        throw std::runtime_error("Unexpected rank for dataset '" +
                                 dataset_name + "'.");
    }
}

std::vector<int> read_int_column_2d(H5::DataSet& dataset, hsize_t column_index,
                                    const std::string& dataset_name) {
    require_rank(dataset, 2, dataset_name);

    const H5::DataSpace file_space = dataset.getSpace();
    std::array<hsize_t, 2> dims{0, 0};
    file_space.getSimpleExtentDims(dims.data(), nullptr);

    const hsize_t num_rows = dims[0];
    if (column_index >= dims[1]) {
        throw std::runtime_error("Column index out of bounds in dataset '" +
                                 dataset_name + "'.");
    }

    std::vector<int> out(num_rows);
    if (num_rows == 0) {
        return out;
    }

    const std::array<hsize_t, 2> count{num_rows, 1};
    const std::array<hsize_t, 2> offset{0, column_index};
    file_space.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

    const H5::DataSpace mem_space(1, &num_rows);
    dataset.read(out.data(), H5::PredType::NATIVE_INT, mem_space, file_space);
    return out;
}

std::vector<double> read_double_attr_3d(H5::DataSet& dataset,
                                        hsize_t attr_index,
                                        const std::string& dataset_name) {
    require_rank(dataset, 3, dataset_name);

    const H5::DataSpace file_space = dataset.getSpace();
    std::array<hsize_t, 3> dims{0, 0, 0};
    file_space.getSimpleExtentDims(dims.data(), nullptr);

    const hsize_t num_nodes = dims[1];
    if (attr_index >= dims[2]) {
        throw std::runtime_error("Attribute index out of bounds in dataset '" +
                                 dataset_name + "'.");
    }

    std::vector<double> out(num_nodes);
    if (num_nodes == 0) {
        return out;
    }

    const std::array<hsize_t, 3> count{1, num_nodes, 1};
    const std::array<hsize_t, 3> offset{0, 0, attr_index};
    file_space.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

    const H5::DataSpace mem_space(1, &num_nodes);
    dataset.read(out.data(), H5::PredType::NATIVE_DOUBLE, mem_space,
                 file_space);
    return out;
}

std::vector<char> read_node_types(H5::DataSet& dataset,
                                  const std::string& dataset_name) {
    require_rank(dataset, 3, dataset_name);

    const H5::DataSpace file_space = dataset.getSpace();
    std::array<hsize_t, 3> dims{0, 0, 0};
    file_space.getSimpleExtentDims(dims.data(), nullptr);

    const hsize_t num_nodes = dims[1];
    if (dims[2] == 0) {
        throw std::runtime_error(
            "Node type dataset has empty third dimension.");
    }

    std::vector<char> out(num_nodes, 'D');
    if (num_nodes == 0) {
        return out;
    }

    const std::array<hsize_t, 3> count{1, num_nodes, 1};
    const std::array<hsize_t, 3> offset{0, 0, 0};
    file_space.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

    const H5::DataSpace mem_space(1, &num_nodes);

    const H5::StrType string_type = dataset.getStrType();
    const std::size_t string_size = string_type.getSize();
    if (string_size == 0) {
        throw std::runtime_error("Node type strings have invalid size.");
    }

    std::string buffer;
    buffer.resize(num_nodes * string_size, '\0');
    dataset.read(buffer.data(), string_type, mem_space, file_space);

    for (hsize_t i = 0; i < num_nodes; ++i) {
        out[i] = buffer[i * string_size];
    }

    return out;
}

std::vector<int> map_index_links_to_node_numbers(
    const std::vector<int>& link_idx, const std::vector<int>& node_nums,
    const std::string& link_name) {
    std::vector<int> out;
    out.reserve(link_idx.size());

    for (const int idx_base1 : link_idx) {
        if (idx_base1 <= 0 ||
            static_cast<std::size_t>(idx_base1) > node_nums.size()) {
            throw std::runtime_error("Invalid node index in link dataset '" +
                                     link_name + "'.");
        }
        out.push_back(node_nums[static_cast<std::size_t>(idx_base1 - 1)]);
    }

    return out;
}

}  // namespace

ESATANReader::ESATANReader(ThermalMathematicalModel& model) : _model(model) {}

void ESATANReader::read_tmd(const std::string& filepath) {
    const H5::H5File tmd_file(filepath, H5F_ACC_RDONLY);

    const H5::Group analysis_group = tmd_file.openGroup("AnalysisSet1");
    const H5::Group data_group = analysis_group.openGroup("DataGroup1");

    H5::DataSet nodes_dataset = analysis_group.openDataSet("thermalNodes");
    H5::DataSet gl_dataset = analysis_group.openDataSet("conductorsGL");
    H5::DataSet gr_dataset = analysis_group.openDataSet("conductorsGR");

    H5::DataSet node_real_data_dataset =
        data_group.openDataSet("thermalNodesRealData");
    H5::DataSet node_string_data_dataset =
        data_group.openDataSet("thermalNodesStringData");
    H5::DataSet gl_data_dataset = data_group.openDataSet("conductorDataGL");
    H5::DataSet gr_data_dataset = data_group.openDataSet("conductorDataGR");

    const std::vector<int> node_numbers =
        read_int_column_2d(nodes_dataset, 0, "thermalNodes");
    const std::vector<char> node_types =
        read_node_types(node_string_data_dataset, "thermalNodesStringData");

    if (node_numbers.size() != node_types.size()) {
        throw std::runtime_error(
            "Mismatch between thermalNodes and thermalNodesStringData sizes.");
    }

    std::array<std::vector<double>, k_num_real_node_attrs> node_attrs;
    for (std::size_t i = 0; i < node_attrs.size(); ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
        node_attrs[i] =
            read_double_attr_3d(node_real_data_dataset, static_cast<hsize_t>(i),
                                "thermalNodesRealData");
    }

    std::unordered_set<int> inactive_nodes;

    for (std::size_t i = 0; i < node_numbers.size(); ++i) {
        const int node_number = node_numbers[i];
        const char node_type = node_types[i];

        if (node_type == 'X') {
            inactive_nodes.insert(node_number);
            continue;
        }

        Node node(node_number);
        if (node_type == 'B') {
            node.set_type('B');
        }

        node.set_T(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::T)][i] +
            273.15);
        node.set_C(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::C)][i]);
        node.set_qa(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::QA)][i]);
        node.set_qe(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::QE)][i]);
        node.set_qi(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::QI)][i]);
        node.set_qr(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::QR)][i]);
        node.set_qs(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::QS)][i]);
        node.set_a(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::A)][i]);
        node.set_aph(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::APH)][i]);
        node.set_eps(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::EPS)][i]);
        node.set_fx(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::FX)][i]);
        node.set_fy(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::FY)][i]);
        node.set_fz(
            node_attrs[static_cast<std::size_t>(RealNodeAttrIndex::FZ)][i]);

        _model.add_node(node);
    }

    const std::vector<int> gl_idx_1 =
        read_int_column_2d(gl_dataset, 0, "conductorsGL");
    const std::vector<int> gl_idx_2 =
        read_int_column_2d(gl_dataset, 1, "conductorsGL");
    const std::vector<double> gl_values =
        read_double_attr_3d(gl_data_dataset, 0, "conductorDataGL");

    const std::vector<int> gl_node_1 =
        map_index_links_to_node_numbers(gl_idx_1, node_numbers, "conductorsGL");
    const std::vector<int> gl_node_2 =
        map_index_links_to_node_numbers(gl_idx_2, node_numbers, "conductorsGL");

    if (gl_node_1.size() != gl_values.size() ||
        gl_node_2.size() != gl_values.size()) {
        throw std::runtime_error("Mismatch in GL dataset sizes.");
    }

    for (std::size_t i = 0; i < gl_values.size(); ++i) {
        const int node_1 = gl_node_1[i];
        const int node_2 = gl_node_2[i];
        if (inactive_nodes.contains(node_1) ||
            inactive_nodes.contains(node_2)) {
            continue;
        }

        _model.add_conductive_coupling(node_1, node_2, gl_values[i]);
    }

    const std::vector<int> gr_idx_1 =
        read_int_column_2d(gr_dataset, 0, "conductorsGR");
    const std::vector<int> gr_idx_2 =
        read_int_column_2d(gr_dataset, 1, "conductorsGR");
    const std::vector<double> gr_values =
        read_double_attr_3d(gr_data_dataset, 0, "conductorDataGR");

    const std::vector<int> gr_node_1 =
        map_index_links_to_node_numbers(gr_idx_1, node_numbers, "conductorsGR");
    const std::vector<int> gr_node_2 =
        map_index_links_to_node_numbers(gr_idx_2, node_numbers, "conductorsGR");

    if (gr_node_1.size() != gr_values.size() ||
        gr_node_2.size() != gr_values.size()) {
        throw std::runtime_error("Mismatch in GR dataset sizes.");
    }

    for (std::size_t i = 0; i < gr_values.size(); ++i) {
        const int node_1 = gr_node_1[i];
        const int node_2 = gr_node_2[i];
        if (inactive_nodes.contains(node_1) ||
            inactive_nodes.contains(node_2)) {
            continue;
        }

        _model.add_radiative_coupling(node_1, node_2, gr_values[i]);
    }

    if (verbose) {
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "Read TMD: {}", filepath);
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "  Nodes: {}",
                           node_numbers.size());
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "  GLs: {}",
                           gl_values.size());
        SPDLOG_LOGGER_INFO(pycanha::get_logger(), "  GRs: {}",
                           gr_values.size());
    }
}

}  // namespace pycanha
// NOLINTEND(misc-include-cleaner)
