/// C++ Implementation of a Node
/**
 * This class provides a structured representation of an individual thermal
 * node. The instance works differently depending if the node is linked to an
 * instance of Nodes or not.
 *
 * If the node is instantiated directly from Node without specifing a Nodes
 * instance the attributes of the node are dynamically stored only in the node
 * instance. Changing any attribute will only affect the instance.
 *
 * If the node is associated to a Nodes instance, the node tracks it.
 * In this case Node just links to the information stored in Nodes.
 * When getting/setting an attribute, Node will first check if it is associated
 * to a Nodes instance. If associated, calls the appropiated getter/setter of
 * Nodes, otherwise return the data stored locally in the instance.
 *
 * A standalone node can be associated to a Nodes anytime. When a standalone
 * node is associated, the information of the node is passed to the Nodes
 * instance. Nodes reorganize its structure to accomodate the new node and Node
 * deletes its internal storage, so the information of the node is not
 * duplicated. The node can be associated to Nodes by calling Nodes::add_node
 * with the node instance as argument.
 *
 * Some considerations:
 * - A node can be only associated to a single Nodes instance.
 * - Deleting the instance of an already associated node won't delete the node
 * from Nodes.
 * - Because an associated node is only a link to Nodes, there can be multiple
 * Node instances pointing to the same node. The instances behaves as they are
 * the same node, but the node is not duplicated in Nodes. The behaviour is
 * similar to several regular pointers that point to the same location.
 * - If the Nodes instance is deleted, the already associated node instances
 * become invalid. A warning is displayed (if VERBOSE == 1) and the node
 * instance returns 'nan' for double attributes and 'static_cast<char>(0)' as
 * node type.
 */

#pragma once
#include <memory>
#include <string>

#include "./nodes.hpp"

using namespace pycanha;  // NOLINT

enum NodeType : unsigned char { DIFFUSIVE_NODE = 'D', BOUNDARY_NODE = 'B' };

class Nodes;

class Node {
    friend class Nodes;

    // Public attributes
  public:
    // Private attributes
  private:
    /**
     * Stores a weak pointer to the parent Nodes instance where the node
     * belongs. The weak pointer is created from a shared pointer in TNs. When
     * the TNs instance is destroyed, so is the shared pointer. And TN knows
     * that the parent instance is not accesible any more
     */
    std::weak_ptr<Nodes> _parent_pointer;

    /**
     * Local storage structure to save the information of a non-associated node.
     * The storage is dynamically de/allocated.
     */
    struct LocalStorage {
        char type;  // Type
        // NOLINTBEGIN(readability-identifier-naming)
        double T;  // Temperature
        double C;  // Thermal capacity
        // NOLINTEND(readability-identifier-naming)
        double qs;   // Solar load
        double qa;   // Albedo load
        double qe;   // Earth IR load
        double qi;   // Internal load
        double qr;   // Other load
        double a;    // Area
        double fx;   // X coordinate
        double fy;   // Y coordinate
        double fz;   // Z coordinate
        double eps;  // IR emissivity
        double aph;  // Solar absortivity

        // NOLINTBEGIN(readability-identifier-naming)
        std::string literal_C;  // Literal Thermal capacity
        // NOLINTEND(readability-identifier-naming)

        std::string literal_qs;   // Literal Solar load
        std::string literal_qa;   // Literal Albedo load
        std::string literal_qe;   // Literal Earth IR load
        std::string literal_qi;   // Literal Internal load
        std::string literal_qr;   // Literal Other load
        std::string literal_a;    // Literal Area
        std::string literal_fx;   // Literal X coordinate
        std::string literal_fy;   // Literal Y coordinate
        std::string literal_fz;   // Literal Z coordinate
        std::string literal_eps;  // Literal IR emissivity
        std::string literal_aph;  // Literal Solar absortivity
    };

    /**
     * A pointer to the LocalStorage struct. If the node is not associated to
     * the TNs class it is a valid pointer to an existing struct. Otherwise is a
     * nullptr.
     */
    LocalStorage* _local_storage_ptr;

    /**
     * Node attribute: User node number
     */
    int _node_num;

    // Public methods
  public:
    //-----------------------------------------
    //             Constructors
    //-----------------------------------------

    /// Local node (non-associated) constructor.
    /**
     * The constructor to create a non-associated node. Only the user node
     * number is required. The node is diffusive and the attribute values are
     * set to zero. The node type and attributes can be changed through the
     * setters member functions.
     */
    explicit Node(int node_num);

    /// Associated node constructor 1.
    /**
     * Similar to the previous one, but given directly the weak pointer to the
     * Nodes instance.
     */
    Node(int node_num, std::weak_ptr<Nodes> parent_pointer);

    // Move constructor
    Node(Node&& other_node) noexcept;

    // Copy constructor
    Node(const Node& other_node);

    // Asignment operator
    Node& operator=(const Node& other_node);

    // Move assignment operator
    Node& operator=(Node&& other_node) noexcept;

    // Destructor
    ~Node();

    //-----------------------------------------
    //    Attribute getters and setters
    //-----------------------------------------

    // TODO: Inconsistent nomenclature
    int get_node_num();      ///< User node number getter.
    int get_int_node_num();  ///< Internal node number getter.
    /**
     * Two valid types:
     * - 'D': Diffusive
     * - 'B': Boundary
     */
    char get_type();  ///< Type getter.
    // NOLINTBEGIN(readability-identifier-naming)
    double get_T();  ///< Temperature [K] getter.
    double get_C();  ///< Thermal capacity [J/K] getter.
    // NOLINTEND(readability-identifier-naming)
    double get_qs();   ///< Solar load [W] getter.
    double get_qa();   ///< Albedo load [W] getter.
    double get_qe();   ///< Earth IR load [W] getter.
    double get_qi();   ///< Internal load [W] getter.
    double get_qr();   ///< Other load [W] getter.
    double get_a();    ///< Area [m^2] getter.
    double get_fx();   ///< X coordinate [m] getter.
    double get_fy();   ///< Y coordinate [m] getter.
    double get_fz();   ///< Z coordinate [m] getter.
    double get_eps();  ///< IR emissivity getter.
    double get_aph();  ///< Solar absortivity getter.

    // NOLINTBEGIN(readability-identifier-naming)
    [[nodiscard]] std::string get_literal_C()
        const;  ///< Literal thermal capacity getter.
    // NOLINTEND(readability-identifier-naming)

    // TODO: Inconsistent nomenclature
    void set_node_num(int node_num);  ///< Internal node number setter.
    /**
     * Two valid types:
     * - 'D': Diffusive
     * - 'B': Boundary
     */
    void set_type(char type);  ///< Type setter.
    // NOLINTBEGIN(readability-identifier-naming)
    void set_T(double T);  ///< Temperature [K] setter.
    void set_C(double C);  ///< Thermal capacity [J/K] setter.
    // NOLINTEND(readability-identifier-naming)
    void set_qs(double qs);    ///< Solar load [W] setter.
    void set_qa(double qa);    ///< Albedo load [W] setter.
    void set_qe(double qe);    ///< Earth IR load [W] setter.
    void set_qi(double qi);    ///< Internal load [W] setter.
    void set_qr(double qr);    ///< Other load [W] setter.
    void set_a(double a);      ///< Area [m^2] setter.
    void set_fx(double fx);    ///< X coordinate [m] setter.
    void set_fy(double fy);    ///< Y coordinate [m] setter.
    void set_fz(double fz);    ///< Z coordinate [m] setter.
    void set_eps(double eps);  ///< IR emissivity setter.
    void set_aph(double aph);  ///< Solar absortivity setter.

    // NOLINTBEGIN(readability-identifier-naming)
    void set_literal_C(std::string str);  ///< Literal thermal capacity setter.
    // NOLINTEND(readability-identifier-naming)

    // Other getters
    //-------------

    /// Get a weak pointer to Nodes parent instance.
    /**
     * If the node is associated, the weak pointer should referene a Nodes
     * instance where the information of the node is stored. If the node isn't
     * associated, the weak pointer doesn't reference any instance.
     */
    // TODO: Consistent nomenclature
    std::weak_ptr<Nodes> get_parent_pointer();

    /// Get an unsigned 64 bit integer representation of the address of the
    /// parent instance.
    /**
     * If the node is associated, the weak pointer should referene a Nodes
     * instance where the information of the node is stored. If the node isn't
     * associated, the weak pointer doesn't reference any instance.
     */
    // TODO: Consistent nomenclature
    // TODO: Undefined behaviour for non-associated nodes?????
    uint64_t get_int_parent_pointer();

    // Other setters
    //-------------

    /// Set the parent instance.
    /**
     * Associate the node to a Nodes instance. The information of the node is
     * not copied!!
     */
    // TODO: is really useful this method????
    void set_thermal_nodes_parent(
        std::weak_ptr<Nodes> thermal_nodes_parent_ptr);

    // Private methods
  private:
    /**
     * Deallocate the memory of the local information of the node and set the
     * local storage pointer to null.
     */
    void _local_storage_destructor();
};
