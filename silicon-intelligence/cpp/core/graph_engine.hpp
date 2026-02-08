#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <tuple>
#include <boost/graph/adjacency_list.hpp>

namespace si {

// Mirroring CanonicalSiliconGraph.py enums
enum class NodeType {
    CELL,
    MACRO,
    PORT,
    CLOCK,
    POWER,
    SIGNAL
};

enum class EdgeType {
    CONNECTION,
    PHYSICAL_PROXIMITY,
    TIMING_DEPENDENCY,
    POWER_FEED
};

struct NodeAttributes {
    NodeType node_type;
    std::string cell_type = "";
    float area = 0.0f;
    float power = 0.0f;
    float delay = 0.0f;
    float capacitance = 0.0f;
    std::optional<std::pair<float, float>> position;
    std::string region = "";
    std::string voltage_domain = "";
    std::string clock_domain = "";
    bool is_clock_root = false;
    bool is_macro = false;
    float estimated_congestion = 0.0f;
    float timing_criticality = 0.0f;
};

struct EdgeAttributes {
    EdgeType edge_type;
    float resistance = 0.0f;
    float capacitance = 0.0f;
    float delay = 0.0f;
    float length = 0.0f;
    std::vector<std::string> layers_used;
    float congestion = 0.0f;
    float capacity = 1.0f;
};

// Boost Graph Definition
// We use a MultiDiGraph equivalent: bidirectional with multiple edges allowed
typedef boost::adjacency_list<
    boost::vecS,           // OutEdgeList
    boost::vecS,           // VertexList
    boost::bidirectionalS, // Directed/Bidirectional
    NodeAttributes,        // VertexProperty
    EdgeAttributes         // EdgeProperty
> SiliconGraph;

class GraphEngine {
public:
    GraphEngine();
    ~GraphEngine();

    // Node operations
    uint64_t add_node(const std::string& name, const NodeAttributes& attrs);
    bool has_node(const std::string& name) const;
    void update_node(const std::string& name, const NodeAttributes& attrs);
    
    // Edge operations
    void add_edge(const std::string& src, const std::string& dst, const EdgeAttributes& attrs);

    // Stats
    size_t num_nodes() const;
    size_t num_edges() const;

    // Node attribute accessors for optimization
    std::pair<float, float> get_node_position(const std::string& name) const;
    void set_node_position(const std::string& name, float x, float y);
    std::vector<std::string> get_neighbors(const std::string& name) const;
    std::vector<std::string> get_all_node_names() const;

    // Algorithms (Phase 2 & 3)
    std::vector<std::string> get_timing_critical_nodes(float threshold) const;

    // Phase 5 Additions
    std::pair<float, float> get_node_position(const std::string& name) const;
    void set_node_position(const std::string& name, float x, float y);
    std::vector<std::string> get_neighbors(const std::string& name) const;
    std::vector<std::string> get_all_node_names() const;

private:
    SiliconGraph g_;
    std::unordered_map<std::string, SiliconGraph::vertex_descriptor> name_to_vertex_;
    std::unordered_map<SiliconGraph::vertex_descriptor, std::string> vertex_to_name_;
};

} // namespace si
