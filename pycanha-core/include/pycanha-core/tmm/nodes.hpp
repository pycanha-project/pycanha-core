
/// C++ Implementation of Nodes
/**
 * Contain all the information regarding the thermal nodes in such a way that it
 * can be easily an efficiently readed by the solvers. The class itself ensures
 * that the nodes are always ordered in the same way according to their user
 * node number and their type.
 *
 * There are two type of storage for the properties of the nodes:
 * - Dense (std::vector<Type>)
 * - Sparse (Eigen::SparseVector<Type>)
 *
 * The dense storage is reserved for node attributes that are always defined, as
 * the user number, the temperature and the thermal capacity.
 * The dense container might change in the future but is guarantee to be memory
 * contiguous and ordered according to the internal numbering scheme explained
 * below.
 *
 * The sparse storage is used for node attributes that are tipically zero,
 * including the qs, qa, qe, qi, qr, a, fx, fy, fz, eps and aph. This type of
 * storage is also used for the literals. Tipically the attributes are just
 * numbers, but some of them might be defined by formula. The formula in these
 * cases is saved as a string in a sparse vector. The order of these vector is
 * also based on the internal node number but the storage is handled by an Eigen
 * Sparse Vector (see
 * https://eigen.tuxfamily.org/dox/classEigen_1_1SparseVector.html for more
 * info).
 *
 * The class handles automatically the order of the vectors. The order (also
 * called internal node number) can not be changed and is based on the user node
 * number and the type of the node. The 'diffusive' nodes are placed before the
 * boundary and both 'blocks' are ordered according to the user node number.
 * Therefore inserting a lot of nodes in a different order as the class handle
 * it might be slow as the containers needs to be resized and reordered.
 *
 * As an example, the following nodes: D100, D101, B102, D103, D500, B600, D700
 * would be stored internally:
 *
 * - T_vector: [T100, T101, T103, T500, T700, T102, T600]
 * - C_vector: [C100, C101, C103, C500, C700, C102, C600]
 * - etc..
 */

#pragma once
#include <Eigen/Sparse>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/literalstring.hpp"

namespace pycanha {

class Couplings;
class Node;
class Nodes {
    friend class Node;
    friend class Couplings;
    friend class ThermalNetwork;

  public:
    int estimated_number_of_nodes{100};  // TODO: Is this useful? Delete?

  private:
    /**
     * Variable to store the shared_pointer to the instance.
     * When the instance is destroyed, so is the shared pointer.
     * All the references to the instance in Node are through weak_ptr
     * so the node automatically knows that it doesn't belong to any Nodes
     */
    std::shared_ptr<Nodes> _self_pointer;

    /**
     * Vector with the user node number of the diffusive nodes. The vector is
     * ordered from lower user node number to higher.
     */
    // TODO: Consistent nomenclature
    std::vector<NodeNum> _diff_node_num_vector;

    /**
     * Vector with the user node number of the boundary nodes. The vector is
     * ordered from lower user node number to higher.
     */
    // TODO: Consistent nomenclature
    std::vector<NodeNum> _bound_node_num_vector;

    // The concatenated vector [diff_node_num_vector, bound_node_num_vector]
    // would contain the user node number of all the nodes of the model ordered
    // according to the internal number.

  public:
    // NOLINTBEGIN(readability-identifier-naming)
    // Dense storage for vectors used directly by the solvers
    std::vector<double>
        T_vector;  ///< Temperature [K] vector ordered by internal node number.
    std::vector<double> C_vector;  ///< Thermal capacity [J/K] vector ordered by
                                   ///< internal node number.
    // NOLINTEND(readability-identifier-naming)

    // Sparse storage for attributes that are typically 0
    Eigen::SparseVector<double>
        qs_vector;  ///< Solar load [W] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double>
        qa_vector;  ///< Albedo load [W] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double>
        qe_vector;  ///< Earth IR load [W] sparse vector ordered by internal
                    ///< node number.
    Eigen::SparseVector<double>
        qi_vector;  ///< Internal load [W] sparse vector ordered by internal
                    ///< node number.
    Eigen::SparseVector<double>
        qr_vector;  ///< Other load [W] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double> a_vector;  ///< Area [m^2] sparse vector ordered
                                           ///< by internal node number.
    Eigen::SparseVector<double>
        fx_vector;  ///< X coordinate [m] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double>
        fy_vector;  ///< Y coordinate [m] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double>
        fz_vector;  ///< Z coordinate [m] sparse vector ordered by internal node
                    ///< number.
    Eigen::SparseVector<double>
        eps_vector;  ///< IR emissivity sparse vector ordered by internal node
                     ///< number.
    Eigen::SparseVector<double>
        aph_vector;  ///< Solar absortivity sparse vector ordered by internal
                     ///< node number.

    // Literals storage, typically empty
    // NOLINTBEGIN(readability-identifier-naming)
    Eigen::SparseVector<LiteralString>
        literals_C;  ///< Thermal capacity literal sparse vector ordered by
                     ///< internal node number.
    // NOLINTEND(readability-identifier-naming)
    Eigen::SparseVector<LiteralString>
        literals_qs;  ///< Solar load literal sparse vector ordered by internal
                      ///< node number.
    Eigen::SparseVector<LiteralString>
        literals_qa;  ///< Albedo load literal sparse vector ordered by internal
                      ///< node number.
    Eigen::SparseVector<LiteralString>
        literals_qe;  ///< Earth IR load literal sparse vector ordered by
                      ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_qi;  ///< Internal load literal sparse vector ordered by
                      ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_qr;  ///< Other load literal sparse vector ordered by internal
                      ///< node number.
    Eigen::SparseVector<LiteralString>
        literals_a;  ///< Area literal sparse vector ordered by internal node
                     ///< number.
    Eigen::SparseVector<LiteralString>
        literals_fx;  ///< X coordinate literal sparse vector ordered by
                      ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_fy;  ///< Y coordinate literal sparse vector ordered by
                      ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_fz;  ///< Z coordinate literal sparse vector ordered by
                      ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_eps;  ///< IR emissivity literal sparse vector ordered by
                       ///< internal node number.
    Eigen::SparseVector<LiteralString>
        literals_aph;  ///< Solar absortivity literal sparse vector ordered by
                       ///< internal node number.

  private:
    /// User node number and internal node number mapping
    /**
     * Anytime the user ask for an attribute of a node, it is typically done by
     * using the user node number. But internally, attributes are ordered
     * according to the internal number. The unordered_map allow O(1) access to
     * node attributes. The map is dinamically updated anytime the node
     * structure change.
     *
     * The opposite map (internal to user) is not necessary as the user number
     * are stored in two ordered vectors (diff_node_num_vector and
     * bound_node_num_vector).
     */
    // TODO: Consistent nomenclature
    mutable std::unordered_map<NodeNum, Index> _usr_to_int_node_num;

    /**
     * The same map as a flat table, used instead of the unordered_map when the
     * node numbers are dense enough: entry (number - _dense_base) holds the
     * internal index, or -1 for a number that is not a node. One memory read
     * per lookup, and a rebuild is a single pass over the nodes.
     */
    mutable std::vector<std::int32_t> _dense_node_index;
    mutable NodeNum _dense_base{0};
    mutable bool _dense_mapped{true};

    /**
     * Bumped by every change of the node structure: a node added, removed or
     * moved. Whatever depends on the structure (the node number map here, the
     * coupling matrix sizes, an initialised solver) remembers the version it
     * was built for and compares.
     */
    std::uint64_t _structure_version{0};

    /// The structure version the node number map was built for.
    mutable std::uint64_t _mapped_version{0};

    /**
     * Coupling containers that index their matrices by this instance's
     * internal node order, linked through Couplings itself (an intrusive list,
     * so registering never allocates). A node inserted in the middle of its
     * block, or removed, moves the internal index of the nodes after it; each
     * of them is told so right away and remaps its stored couplings.
     */
    Couplings* _first_observer{nullptr};

  public:
    // Constructors

    /// Default constructor.
    /**
     * Create a blank instance of Thermal Nodes. It also creates blank instances
     * of conductive and radiative couplings. The couplings can be accessed
     * through the pointers or the get methods.
     */
    Nodes();

    /// Destructor
    ~Nodes();

    // Copy Constructor
    Nodes(const Nodes& other);

    // Copy Assignment Operator
    Nodes& operator=(const Nodes& other);

    // Move Constructor
    Nodes(Nodes&& other) noexcept;

    // Move Assignment Operator
    Nodes& operator=(Nodes&& other) noexcept;

    // Node handlers

    /// Method to add a new node.
    /**
     * The information of the node is copied to Nodes. The structure of Nodes
     * is automatically rearanged to follow the correct internal order.
     * Additionally the struture of the conductive and radiative couplings is
     * also rearranged to match the new order.
     *
     * IMPORTANT: The method will modify the instance of the node used as
     * argument. Typically the node input argument is not associated with Nodes,
     * and the node attributes are locally stored in the node. After copying the
     * attributes of the node to Nodes, the local storage of the input node is
     * deleted and the node is associated to the Nodes instance.
     */
    void add_node(Node& node);

    /// Method to add a several nodes contained in a std::vector.
    /**
     * Currently, the implementation is just an iteration from the begining to
     * the end of the node vector, calling Nodes::add_node for each element.
     *
     * In the future, some optimization might be implemented, so inserting a
     * large number of nodes will be more efficient through this method.
     */
    void add_nodes(std::vector<Node>& node_vector);

    /// Method to add a batch of nodes of one type from arrays.
    /**
     * Nothing is copied from the batch except into the node storage itself.
     * A batch sorted by increasing number, all above the numbers already
     * stored in its block, is appended: O(k), plus O(boundary nodes) for a
     * diffusive batch. Any other batch is sorted and merged in, one linear
     * pass over every stored attribute.
     *
     * Rejected, and reported: every copy of a number repeated in the batch, a
     * number that is already a node, and the whole batch when its type is not
     * 'D' or 'B' or an attribute span does not match the numbers in length.
     * The node number map is updated in place on an append, and left for the
     * next lookup to rebuild otherwise. Pointers into the attribute storage
     * (the *_value_ref getters) are invalidated.
     */
    BulkReport add_nodes(const NodeBatch& batch);

    /// Reserves storage for @p num_nodes nodes in total, so that adding them
    /// one by one or in several batches never reallocates the dense vectors.
    void reserve(Index num_nodes);

    /// Method to delete a node of the model.
    /**
     * The node is deleted and the structure of Nodes is rearranged. The
     * corresponding rows and columns of the linear and radiative couplings are
     * also deleted, so any information regarding the links of the deleted node
     * with other nodes of the model is also deleted.
     */
    void remove_node(NodeNum node_num);

    // Attribute getters and setters
    char get_type(NodeNum node_num);  ///< Type getter.
    // NOLINTBEGIN(readability-identifier-naming)
    double get_T(NodeNum node_num);  ///< Temperature [K] getter.
    double get_C(NodeNum node_num);  ///< Thermal capacity [J/K] getter.
    // NOLINTEND(readability-identifier-naming)
    double get_qs(NodeNum node_num);   ///< Solar load [W] getter.
    double get_qa(NodeNum node_num);   ///< Albedo load [W] getter.
    double get_qe(NodeNum node_num);   ///< Earth IR load [W] getter.
    double get_qi(NodeNum node_num);   ///< Internal load [W] getter.
    double get_qr(NodeNum node_num);   ///< Other load [W] getter.
    double get_a(NodeNum node_num);    ///< Area [m^2] getter.
    double get_fx(NodeNum node_num);   ///< X coordinate [m] getter.
    double get_fy(NodeNum node_num);   ///< Y coordinate [m] getter.
    double get_fz(NodeNum node_num);   ///< Z coordinate [m] getter.
    double get_eps(NodeNum node_num);  ///< IR emissivity getter.
    double get_aph(NodeNum node_num);  ///< Solar absortivity getter.

    // NOLINTBEGIN(readability-identifier-naming)
    std::string get_literal_C(
        NodeNum node_num) const;  ///< Literal thermal capacity getter.
    // NOLINTEND(readability-identifier-naming)

    /// Type setter: 'D' diffusive, 'B' boundary. The node moves to the other
    /// block, which reorders every node and coupling once; for many nodes
    /// use set_types, which does it once for all of them.
    bool set_type(NodeNum node_num, char type);

    /**
     * Changes the type of many nodes at once ('D' diffusive, 'B' boundary).
     * Each node keeps its attributes and couplings; the nodes and the
     * coupling matrices are reordered in a single pass, O(nodes + couplings).
     * Unknown node numbers are rejected and reported; a node that already
     * has the type is accepted unchanged.
     */
    BulkReport set_types(std::span<const NodeNum> node_nums, char type);
    // NOLINTBEGIN(readability-identifier-naming)
    bool set_T(NodeNum node_num, double T);  ///< Temperature [K] setter.
    bool set_C(NodeNum node_num, double C);  ///< Thermal capacity [J/K] setter.
    // NOLINTEND(readability-identifier-naming)
    bool set_qs(NodeNum node_num,
                double value);  ///< Solar load [W] setter.
    bool set_qa(NodeNum node_num,
                double value);  ///< Albedo load [W] setter.
    bool set_qe(NodeNum node_num,
                double value);  ///< Earth IR load [W] setter.
    bool set_qi(NodeNum node_num,
                double value);                    ///< Internal load [W] setter.
    bool set_qr(NodeNum node_num, double value);  ///< Other load [W] setter.
    bool set_a(NodeNum node_num, double value);   ///< Area [m^2] setter.
    bool set_fx(NodeNum node_num, double value);  ///< X coordinate [m] setter.
    bool set_fy(NodeNum node_num, double value);  ///< Y coordinate [m] setter.
    bool set_fz(NodeNum node_num, double value);  ///< Z coordinate [m] setter.
    bool set_eps(NodeNum node_num,
                 double value);  ///< IR emissivity setter.
    bool set_aph(NodeNum node_num,
                 double value);  ///< Solar absortivity setter.

    // NOLINTBEGIN(readability-identifier-naming)
    bool set_literal_C(
        NodeNum node_num,
        const std::string& str);  ///< Literal thermal capacity setter.

    double* get_T_value_ref(
        NodeNum node_num);  ///< Pointer where the temperature value is stored.
    double* get_C_value_ref(
        NodeNum node_num);  ///< Pointer where the capacity value is stored.
    // NOLINTEND(readability-identifier-naming)
    double* get_qs_value_ref(
        NodeNum node_num);  ///< Solar load [W] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_qa_value_ref(
        NodeNum node_num);  ///< Albedo load [W] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_qe_value_ref(
        NodeNum node_num);  ///< Earth IR load [W] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_qi_value_ref(
        NodeNum node_num);  ///< Internal load [W] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_qr_value_ref(
        NodeNum node_num);  ///< Other load [W] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_a_value_ref(
        NodeNum node_num);  ///< Area [m^2] pointer to the value. Note: Values
                            ///< are store in sparse vectors. Calling this
                            ///< function will create a zero value in the matrix
                            ///< if not exist.
    double* get_fx_value_ref(
        NodeNum node_num);  ///< X coordinate [m] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_fy_value_ref(
        NodeNum node_num);  ///< Y coordinate [m] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_fz_value_ref(
        NodeNum node_num);  ///< Z coordinate [m] pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_eps_value_ref(
        NodeNum node_num);  ///< IR emissivity pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.
    double* get_aph_value_ref(
        NodeNum node_num);  ///< Solar absortivity pointer to the value. Note:
                            ///< Values are store in sparse vectors. Calling
                            ///< this function will create a zero value in the
                            ///< matrix if not exist.

    std::optional<Index> get_idx_from_node_num(NodeNum node_num) const;
    ///< Get internal node numbre from user number.
    std::optional<NodeNum> get_node_num_from_idx(Index idx) const;
    ///< Get internal node numbre from user number.
    bool is_node(NodeNum node_num) const;  ///< Check if node is stored.
    Node get_node_from_node_num(
        NodeNum node_num);              ///< Get node object from node number.
    Node get_node_from_idx(Index idx);  ///< Get node object from idx.

    // TODO: select one I/F num_nodes or get_num_nodes
    Index num_nodes() const;      ///< Number of nodes of the model.
    Index get_num_nodes() const;  ///< Number of nodes of the model.
    Index get_num_diff_nodes()
        const;  ///< Number of diffusive nodes of the model.
    Index get_num_bound_nodes()
        const;  ///< Number of boundary nodes of the model.

    bool is_mapped() const;  ///< Check if the internal node structure has
                             ///< already been mapped.

    /// Changes whenever a node is added, removed or moved. See
    /// _structure_version.
    [[nodiscard]] std::uint64_t structure_version() const noexcept;

    /// User numbers of every node, in internal order.
    [[nodiscard]] std::vector<NodeNum> node_numbers() const;

    /// Sets one attribute of many nodes. Unknown nodes are skipped and
    /// reported; for a node given twice the last value wins. A sparse
    /// attribute is rebuilt once, dropping values at or below ZERO_THR_ATTR.
    BulkReport set_values(NodeAttribute attribute,
                          std::span<const NodeNum> node_nums,
                          std::span<const double> values);

    /// Reads one attribute of many nodes into @p values (same length as
    /// @p node_nums). Unknown nodes read as NaN and are reported.
    BulkReport get_values(NodeAttribute attribute,
                          std::span<const NodeNum> node_nums,
                          std::span<double> values) const;

    /// One attribute of every node, in internal order.
    [[nodiscard]] std::vector<double> get_values(NodeAttribute attribute) const;

  private:
    /**
     * Remake the map that tracks user node numbers -> internal node numbers
     * (the flat table when the numbers are dense, the unordered_map
     * otherwise) and mark it valid for the current structure version. Called
     * automatically before a lookup when the structure changed since the last
     * rebuild. It is marked as const because the node structure and the node
     * attributes are not modified; only the mutable map members are.
     */
    void create_node_num_map() const;

    void ensure_node_map() const;
    [[nodiscard]] std::optional<Index> lookup_node_index(
        NodeNum node_num) const;
    [[nodiscard]] bool store_node_index(NodeNum node_num, Index index) const;
    [[nodiscard]] bool shift_boundary_node_indices(Index shift) const;
    void update_map_after_append(char type, std::span<const NodeNum> numbers,
                                 bool map_was_valid);
    [[nodiscard]] bool has_node_number(NodeNum node_num) const;

    void add_observer(Couplings* couplings) noexcept;
    void remove_observer(Couplings* couplings) noexcept;
    void notify_block_remap(char type, std::span<const Index> old_to_new);
    void notify_single_insertion(char type, Index local_index,
                                 Index old_block_size);

    // Marks the nodes set_types has to move to the other block.
    [[nodiscard]] std::vector<std::uint8_t> retype_marks(
        std::span<const NodeNum> node_nums, char type,
        BulkReport& report) const;
    // Puts every node and attribute in the order new_to_old gives (the old
    // internal index of each new one), the first diff_count diffusive;
    // `numbers` is the old internal order of the node numbers.
    void reorder_storage(std::span<const NodeNum> numbers,
                         std::span<const Index> new_to_old, Index diff_count);

    [[nodiscard]] std::vector<double>* dense_storage(
        NodeAttribute attribute) noexcept;
    [[nodiscard]] const std::vector<double>* dense_storage(
        NodeAttribute attribute) const noexcept;
    [[nodiscard]] Eigen::SparseVector<double>* sparse_storage(
        NodeAttribute attribute) noexcept;
    [[nodiscard]] const Eigen::SparseVector<double>* sparse_storage(
        NodeAttribute attribute) const noexcept;
    bool find_node_index(NodeNum node_num, Index& index,
                         const char* error_prefix) const;
    double resolve_get_dense_attr(NodeNum node_num,
                                  const std::vector<double>& storage) const;
    bool resolve_set_dense_attr(NodeNum node_num, std::vector<double>& storage,
                                double value);
    double* resolve_get_dense_attr_ref(NodeNum node_num,
                                       std::vector<double>& storage);
    double resolve_get_sparse_attr(
        NodeNum node_num, const Eigen::SparseVector<double>& storage) const;
    bool resolve_set_sparse_attr(NodeNum node_num,
                                 Eigen::SparseVector<double>& storage,
                                 double value);
    double* resolve_get_sparse_attr_ref(NodeNum node_num,
                                        Eigen::SparseVector<double>& storage);
    std::string resolve_get_literal_attr(
        NodeNum node_num,
        const Eigen::SparseVector<LiteralString>& storage) const;
    bool resolve_set_literal_attr(NodeNum node_num,
                                  Eigen::SparseVector<LiteralString>& storage,
                                  const std::string& value);

    // Insert methods for SparseVectors

    /**
     * Helper method to insert a LiteralString value in the middle of a Sparse
     * vector. The size of the vector is increased by one, and the elements
     * after the inserted one are displaced one position.
     */
    static void insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                                Index index, const LiteralString& string);

    /**
     * Helper method to insert a string value in the middle of a Sparse
     * vector. The size of the vector is increased by one, and the elements
     * after the inserted one are displaced one position.
     */
    static void insert_displace(Eigen::SparseVector<LiteralString>& sparse,
                                Index index, const std::string& string);

    /**
     * Helper method to insert a double value in the middle of a Sparse vector.
     * The size of the vector is increased by one, and the elements after the
     * inserted one are displaced one position.
     */
    static void insert_displace(Eigen::SparseVector<double>& sparse,
                                Index index, double value);

    // Delete methods for SparseVectors

    /**
     * Helper method to delete an entry at position 'index' in a Sparse vector
     * of LiteralString. The size of the vector is decreased by one, and the
     * elements after the deleted one are displaced one position.
     */
    static void delete_displace(Eigen::SparseVector<LiteralString>& sparse,
                                Index index);

    /**
     * Helper method to delete an entry at position 'index' in a Sparse vector
     * of doubles. The size of the vector is decreased by one, and the elements
     * after the deleted one are displaced one position.
     */
    static void delete_displace(Eigen::SparseVector<double>& sparse,
                                Index index);

    /**
     * Insert the node given the positions.
     * This is an internal function to be called from
     * add_node. No checks are performed here.
     */
    void add_node_insert_idx(Node& node, Index insert_idx);
};

}  // namespace pycanha
