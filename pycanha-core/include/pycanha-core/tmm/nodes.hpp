
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../parameters.hpp"
#include "./couplingmatrices.hpp"
#include "./literalstring.hpp"
#include "./node.hpp"

using namespace pycanha;  // NOLINT

class Node;
class Nodes {
    friend class Node;
    friend class Couplings;
    friend class ThermalNetwork;

  public:
    int estimated_number_of_nodes;  // TODO: Is this useful? Delete?

  private:
    /**
     * Variable to store the shared_pointer to the instance.
     * When the instance is destroyed, so is the shared pointer.
     * All the references to the instance in Node are through weak_ptr
     * so the node automatically knows that it doesn't belong to any Nodes
     */
    std::shared_ptr<Nodes> self_pointer;

    /**
     * Vector with the user node number of the diffusive nodes. The vector is
     * ordered from lower user node number to higher.
     */
    // TODO: Consistent nomenclature
    std::vector<int> _diff_node_num_vector;

    /**
     * Vector with the user node number of the boundary nodes. The vector is
     * ordered from lower user node number to higher.
     */
    // TODO: Consistent nomenclature
    std::vector<int> _bound_node_num_vector;

    // The concatenated vector [diff_node_num_vector, bound_node_num_vector]
    // would contain the user node number of all the nodes of the model ordered
    // according to the internal number.

  public:
    // Dense storage for vectors used directly by the solvers
    std::vector<double>
        T_vector;  ///< Temperature [K] vector ordered by internal node number.
    std::vector<double> C_vector;  ///< Thermal capacity [J/K] vector ordered by
                                   ///< internal node number.

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
    Eigen::SparseVector<LiteralString>
        literals_C;  ///< Thermal capacity literal sparse vector ordered by
                     ///< internal node number.
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
    mutable std::unordered_map<int, int> _usr_to_int_node_num;

    /**
     * Variable to track changes in the structure of the nodes.
     * Anytime the node order changes, or a node is added or removed, the
     * variable is set to false. Always, before using the node number map
     * (_usr_to_int_node_num) this variable is checked. If false the map is updated
     * before accesing it.
     */
    mutable bool _node_num_mapped;

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
    void add_node(Node &node);

    /// Method to add a several nodes contained in a std::vector.
    /**
     * Currently, the implementation is just an iteration from the begining to
     * the end of the node vector, calling Nodes::add_node for each element.
     *
     * In the future, some optimization might be implemented, so inserting a
     * large number of nodes will be more efficient through this method.
     */
    void add_nodes(std::vector<Node> &node_vector);

    /// Method to delete a node of the model.
    /**
     * The node is deleted and the structure of Nodes is rearranged. The
     * corresponding rows and columns of the linear and radiative couplings are
     * also deleted, so any information regarding the links of the deleted node
     * with other nodes of the model is also deleted.
     */
    void remove_node(int node_num);

    // Attribute getters and setters
    char get_type(int node_num);   ///< Type getter.
    double get_T(int node_num);    ///< Temperature [K] getter.
    double get_C(int node_num);    ///< Thermal capacity [J/K] getter.
    double get_qs(int node_num);   ///< Solar load [W] getter.
    double get_qa(int node_num);   ///< Albedo load [W] getter.
    double get_qe(int node_num);   ///< Earth IR load [W] getter.
    double get_qi(int node_num);   ///< Internal load [W] getter.
    double get_qr(int node_num);   ///< Other load [W] getter.
    double get_a(int node_num);    ///< Area [m^2] getter.
    double get_fx(int node_num);   ///< X coordinate [m] getter.
    double get_fy(int node_num);   ///< Y coordinate [m] getter.
    double get_fz(int node_num);   ///< Z coordinate [m] getter.
    double get_eps(int node_num);  ///< IR emissivity getter.
    double get_aph(int node_num);  ///< Solar absortivity getter.

    std::string get_literal_C(
        int node_num) const;  ///< Literal thermal capacity getter.

    bool set_type(int node_num, char type);  ///< Type setter.
    bool set_T(int node_num, double T);      ///< Temperature [K] setter.
    bool set_C(int node_num, double C);      ///< Thermal capacity [J/K] setter.
    bool set_qs(int node_num, double qs);    ///< Solar load [W] setter.
    bool set_qa(int node_num, double qa);    ///< Albedo load [W] setter.
    bool set_qe(int node_num, double qe);    ///< Earth IR load [W] setter.
    bool set_qi(int node_num, double qi);    ///< Internal load [W] setter.
    bool set_qr(int node_num, double qr);    ///< Other load [W] setter.
    bool set_a(int node_num, double a);      ///< Area [m^2] setter.
    bool set_fx(int node_num, double fx);    ///< X coordinate [m] setter.
    bool set_fy(int node_num, double fy);    ///< Y coordinate [m] setter.
    bool set_fz(int node_num, double fz);    ///< Z coordinate [m] setter.
    bool set_eps(int node_num, double eps);  ///< IR emissivity setter.
    bool set_aph(int node_num, double aph);  ///< Solar absortivity setter.

    bool set_literal_C(int node_num,
                       std::string str);  ///< Literal thermal capacity setter.

    double *get_T_value_ref(
        int node_num);  ///< Pointer where the temperature value is stored.
    double *get_C_value_ref(
        int node_num);  ///< Pointer where the capacity value is stored.
    double *get_qs_value_ref(
        int node_num);  ///< Solar load [W] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_qa_value_ref(
        int node_num);  ///< Albedo load [W] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_qe_value_ref(
        int node_num);  ///< Earth IR load [W] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_qi_value_ref(
        int node_num);  ///< Internal load [W] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_qr_value_ref(
        int node_num);  ///< Other load [W] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_a_value_ref(
        int node_num);  ///< Area [m^2] pointer to the value. Note: Values
                        ///< are store in sparse vectors. Calling this
                        ///< function will create a zero value in the matrix
                        ///< if not exist.
    double *get_fx_value_ref(
        int node_num);  ///< X coordinate [m] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_fy_value_ref(
        int node_num);  ///< Y coordinate [m] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_fz_value_ref(
        int node_num);  ///< Z coordinate [m] pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_eps_value_ref(
        int node_num);  ///< IR emissivity pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.
    double *get_aph_value_ref(
        int node_num);  ///< Solar absortivity pointer to the value. Note:
                        ///< Values are store in sparse vectors. Calling
                        ///< this function will create a zero value in the
                        ///< matrix if not exist.

    Index get_idx_from_node_num(
        int node_num) const;  ///< Get internal node numbre from user number.
    int get_node_num_from_idx(
        Index idx) const;        ///< Get internal node numbre from user number.
    bool is_node(int node_num);  ///< Check if node is stored.
    Node get_node_from_node_num(
        int node_num);                  ///< Get node object from node number.
    Node get_node_from_idx(Index idx);  ///< Get node object from idx.

    // TODO: select one I/F num_nodes or get_num_nodes
    int num_nodes() const;      ///< Number of nodes of the model.
    int get_num_nodes() const;  ///< Number of nodes of the model.
    int get_num_diff_nodes()
        const;  ///< Number of diffusive nodes of the model.
    int get_num_bound_nodes()
        const;  ///< Number of boundary nodes of the model.

    bool is_mapped() const;  ///< Check if the internal node structure has
                             ///< already been mapped.

  private:
    /**
     * Remake the map that tracks user node numbers -> internal node numbers,
     * after that set _node_num_mapped to true. The method is called automatically
     * when trying to obtain an attribute but _node_num_mapped is false. It is
     * marked as const because the node struture and the node attributes are not
     * modified. However the (mutable) members _usr_to_int_node_num and _node_num_mapped
     * are modified.
     */
    void create_node_num_map() const;

    /**
     * Change the type of the node from diffusive to boundary. Because of how
     * the internal order is defined, the node structure needs to be rearranged.
     */
    void diffusive_to_boundary(int node_num);

    /**
     * Change the type of the node from boundary to diffusive. Because of how
     * the internal order is defined, the node structure needs to be rearranged.
     */
    void boundary_to_diffusive(int node_num);

    // Insert methods for SparseVectors

    /**
     * Helper method to insert a LiteralString value in the middle of a Sparse
     * vector. The size of the vector is increased by one, and the elements
     * after the inserted one are displaced one position.
     */
    void insert_displace(Eigen::SparseVector<LiteralString> &sparse,
                         Index index, const LiteralString &string);

    /**
     * Helper method to insert a string value in the middle of a Sparse
     * vector. The size of the vector is increased by one, and the elements
     * after the inserted one are displaced one position.
     */
    void insert_displace(Eigen::SparseVector<LiteralString> &sparse,
                         Index index, const std::string &string);

    /**
     * Helper method to insert a double value in the middle of a Sparse vector.
     * The size of the vector is increased by one, and the elements after the
     * inserted one are displaced one position.
     */
    void insert_displace(Eigen::SparseVector<double> &sparse, Index index,
                         double value);

    // Delete methods for SparseVectors

    /**
     * Helper method to delete an entry at position 'index' in a Sparse vector
     * of LiteralString. The size of the vector is decreased by one, and the
     * elements after the deleted one are displaced one position.
     */
    void delete_displace(Eigen::SparseVector<LiteralString> &sparse, int index);

    /**
     * Helper method to delete an entry at position 'index' in a Sparse vector
     * of doubles. The size of the vector is decreased by one, and the elements
     * after the deleted one are displaced one position.
     */
    void delete_displace(Eigen::SparseVector<double> &sparse, int index);

    /**
     * Insert the node given the positions.
     * This is an internal function to be called from
     * add_node. No checks are performed here.
     */
    void _add_node_insert_idx(Node &node, Index insert_idx);
};
