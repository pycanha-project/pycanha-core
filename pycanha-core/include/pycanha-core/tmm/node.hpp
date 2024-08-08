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

enum NodeType : unsigned char { DiffusiveNode = 'D', BoundaryNode = 'B' };

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
    std::weak_ptr<Nodes> ParentPointer;

    /**
     * Local storage structure to save the information of a non-associated node.
     * The storage is dynamically de/allocated.
     */
    struct local_storage {
        char m_type;   // Type
        double m_T;    // Temperature
        double m_C;    // Thermal capacity
        double m_qs;   // Solar load
        double m_qa;   // Albedo load
        double m_qe;   // Earth IR load
        double m_qi;   // Internal load
        double m_qr;   // Other load
        double m_a;    // Area
        double m_fx;   // X coordinate
        double m_fy;   // Y coordinate
        double m_fz;   // Z coordinate
        double m_eps;  // IR emissivity
        double m_aph;  // Solar absortivity

        std::string m_literal_C;    // Literal Thermal capacity
        std::string m_literal_qs;   // Literal Solar load
        std::string m_literal_qa;   // Literal Albedo load
        std::string m_literal_qe;   // Literal Earth IR load
        std::string m_literal_qi;   // Literal Internal load
        std::string m_literal_qr;   // Literal Other load
        std::string m_literal_a;    // Literal Area
        std::string m_literal_fx;   // Literal X coordinate
        std::string m_literal_fy;   // Literal Y coordinate
        std::string m_literal_fz;   // Literal Z coordinate
        std::string m_literal_eps;  // Literal IR emissivity
        std::string m_literal_aph;  // Literal Solar absortivity
    };

    /**
     * A pointer to the local_storage struct. If the node is not associated to
     * the TNs class it is a valid pointer to an existing struct. Otherwise is a
     * nullptr.
     */
    local_storage* m_local_storage_ptr;

    /**
     * Node attribute: User node number
     */
    int UsrNodeNum;

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
    explicit Node(int UsrNodeNum);

    /// Associated node constructor 1.
    /**
     * Similar to the previous one, but given directly the weak pointer to the
     * Nodes instance.
     */
    Node(int UsrNodeNum, std::weak_ptr<Nodes> ParentPointer);

    // Move constructor
    Node(Node&& otherNode);

    // Copy constructor
    Node(const Node& otherNode);

    // Asignment operator
    Node& operator=(const Node& otherNode);

    // Destructor
    ~Node();

    //-----------------------------------------
    //    Attribute getters and setters
    //-----------------------------------------

    // TODO: Inconsistent nomenclature
    int getUsrNodeNum();  ///< User node number getter.
    int getIntNodeNum();  ///< Internal node number getter.
    /**
     * Two valid types:
     * - 'D': Diffusive
     * - 'B': Boundary
     */
    char get_type();   ///< Type getter.
    double get_T();    ///< Temperature [K] getter.
    double get_C();    ///< Thermal capacity [J/K] getter.
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

    std::string get_literal_C() const;  ///< Literal thermal capacity getter.

    // TODO: Inconsistent nomenclature
    void setUsrNodeNum(int UsrNodeNum);  ///< Internal node number setter.
    /**
     * Two valid types:
     * - 'D': Diffusive
     * - 'B': Boundary
     */
    void set_type(char type);  ///< Type setter.
    void set_T(double T);      ///< Temperature [K] setter.
    void set_C(double C);      ///< Thermal capacity [J/K] setter.
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

    void set_literal_C(std::string str);  ///< Literal thermal capacity setter.

    // Other getters
    //-------------

    /// Get a weak pointer to Nodes parent instance.
    /**
     * If the node is associated, the weak pointer should referene a Nodes
     * instance where the information of the node is stored. If the node isn't
     * associated, the weak pointer doesn't reference any instance.
     */
    // TODO: Consistent nomenclature
    std::weak_ptr<Nodes> getParentPointer();

    /// Get an unsigned 64 bit integer representation of the address of the
    /// parent instance.
    /**
     * If the node is associated, the weak pointer should referene a Nodes
     * instance where the information of the node is stored. If the node isn't
     * associated, the weak pointer doesn't reference any instance.
     */
    // TODO: Consistent nomenclature
    // TODO: Undefined behaviour for non-associated nodes?????
    uint64_t getintParentPointer();

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
