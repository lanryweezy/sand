#include "graph_engine.hpp"
#include <boost/graph/iteration_macros.hpp>

namespace si {

GraphEngine::GraphEngine() {}
GraphEngine::~GraphEngine() {}

uint64_t GraphEngine::add_node(const std::string& name, const NodeAttributes& attrs) {
    if (has_node(name)) {
        update_node(name, attrs);
        return name_to_vertex_[name];
    }
    
    auto v = boost::add_vertex(attrs, g_);
    name_to_vertex_[name] = v;
    vertex_to_name_[v] = name;
    return v;
}

bool GraphEngine::has_node(const std::string& name) const {
    return name_to_vertex_.find(name) != name_to_vertex_.end();
}

void GraphEngine::update_node(const std::string& name, const NodeAttributes& attrs) {
    if (has_node(name)) {
        auto v = name_to_vertex_[name];
        g_[v] = attrs;
    }
}

void GraphEngine::add_edge(const std::string& src, const std::string& dst, const EdgeAttributes& attrs) {
    if (!has_node(src) || !has_node(dst)) {
        return; // In production we might throw an error or add the nodes
    }
    
    auto u = name_to_vertex_[src];
    auto v = name_to_vertex_[dst];
    boost::add_edge(u, v, attrs, g_);
}

size_t GraphEngine::num_nodes() const {
    return boost::num_vertices(g_);
}

size_t GraphEngine::num_edges() const {
    return boost::num_edges(g_);
}

std::vector<std::string> GraphEngine::get_timing_critical_nodes(float threshold) const {
    std::vector<std::string> result;
    
    // Explicit vertex iteration
    auto [vi, vi_end] = boost::vertices(g_);
    for (; vi != vi_end; ++vi) {
        if (g_[*vi].timing_criticality >= threshold) {
            result.push_back(vertex_to_name_.at(*vi));
        }
    }
    
    return result;
}

std::pair<float, float> GraphEngine::get_node_position(const std::string& name) const {
    if (has_node(name)) {
        auto v = name_to_vertex_.at(name);
        if (g_[v].position) {
            return *g_[v].position;
        }
    }
    return {0.0f, 0.0f};
}

void GraphEngine::set_node_position(const std::string& name, float x, float y) {
    if (has_node(name)) {
        auto v = name_to_vertex_.at(name);
        g_[v].position = std::make_pair(x, y);
    }
}

std::vector<std::string> GraphEngine::get_neighbors(const std::string& name) const {
    std::vector<std::string> neighbors;
    if (has_node(name)) {
        auto u = name_to_vertex_.at(name);
        auto [ai, ai_end] = boost::adjacent_vertices(u, g_);
        for (; ai != ai_end; ++ai) {
            neighbors.push_back(vertex_to_name_.at(*ai));
        }
    }
    return neighbors;
}

std::vector<std::string> GraphEngine::get_all_node_names() const {
    std::vector<std::string> names;
    for (auto const& [name, v] : name_to_vertex_) {
        names.push_back(name);
    }
    return names;
}

} // namespace si
