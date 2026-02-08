#include "optimization_kernels.hpp"
#include <iostream>
#include <omp.h>

namespace si {

OptimizationKernels::OptimizationKernels(std::shared_ptr<GraphEngine> engine) 
    : engine_(engine) {}

float OptimizationKernels::calculate_hpwl() {
    float total_hpwl = 0.0f;
    if (!engine_) return 0.0f;

    auto names = engine_->get_all_node_names();
    for (const auto& name : names) {
        auto p1 = engine_->get_node_position(name);
        auto neighbors = engine_->get_neighbors(name);
        for (const auto& neighbor : neighbors) {
            auto p2 = engine_->get_node_position(neighbor);
            // L1 distance (Manhattan) for HPWL proxy in 2-pin nets
            total_hpwl += std::abs(p1.first - p2.first) + std::abs(p1.second - p2.second);
        }
    }
    return total_hpwl / 2.0f; // Undirected edges or bidirectional edges counted twice
}

void OptimizationKernels::run_global_placement(const PlacementConfig& config) {
    if (!engine_) return;

    std::cout << "Starting Global Placement (Stochastic + Force Directed)..." << std::endl;
    std::cout << "  Threads: " << config.threads << std::endl;
    
    size_t num_nodes = engine_->num_nodes();
    float temp = config.initial_temp;
    float k = std::sqrt((config.area_width * config.area_height) / (num_nodes + 1.0f));

    for (int i = 0; i < config.iterations; ++i) {
        // 1. Parallel Simulated Annealing (Phase-based independent sets)
        auto sets = find_independent_sets();
        for (const auto& set : sets) {
            #pragma omp parallel for num_threads(config.threads)
            for (int s = 0; s < (int)set.size(); ++s) {
                // In a real implementation, we'd do per-node moves here
                // For this demo, we use a single thread step logic for simplicity but multi-threaded across nodes
                // Note: Each thread needs its own RNG for true correctness
            }
        }
        
        // 2. Stochastic Anneal Step
        anneal_step(temp, config.area_width, config.area_height);

        // 3. Force-Directed Refinement
        apply_forces(k, k);
        
        temp *= config.cooling_rate;
        
        if (i % 100 == 0) {
            std::cout << "  Iteration " << i << " HPWL: " << calculate_hpwl() << std::endl;
        }
    }
    std::cout << "Placement Complete." << std::endl;
}

void OptimizationKernels::anneal_step(float temp, float width, float height) {
    auto nodes = engine_->get_all_node_names();
    if (nodes.empty()) return;

    static std::mt19937 gen(42);
    std::uniform_int_distribution<> node_dis(0, nodes.size() - 1);
    std::uniform_real_distribution<> move_dis(-temp, temp);
    std::uniform_real_distribution<> prob_dis(0.0, 1.0);

    for (int i = 0; i < (int)nodes.size(); ++i) {
        std::string node = nodes[node_dis(gen)];
        auto old_pos = engine_->get_node_position(node);
        float old_hpwl = calculate_hpwl();

        float new_x = std::clamp(old_pos.first + move_dis(gen), 0.0f, width);
        float new_y = std::clamp(old_pos.second + move_dis(gen), 0.0f, height);

        engine_->set_node_position(node, new_x, new_y);
        float new_hpwl = calculate_hpwl();

        float delta = new_hpwl - old_hpwl;
        if (delta > 0 && prob_dis(gen) > std::exp(-delta / temp)) {
            // Reject move
            engine_->set_node_position(node, old_pos.first, old_pos.second);
        }
    }
}

std::vector<std::vector<std::string>> OptimizationKernels::find_independent_sets() {
    auto nodes = engine_->get_all_node_names();
    std::vector<std::vector<std::string>> sets;
    if (nodes.empty()) return sets;

    // Greedy Coloring for Independent Sets
    std::map<std::string, int> colors;
    for (const auto& name : nodes) {
        std::set<int> neighbor_colors;
        for (const auto& neighbor : engine_->get_neighbors(name)) {
            if (colors.count(neighbor)) {
                neighbor_colors.insert(colors[neighbor]);
            }
        }
        
        int color = 0;
        while (neighbor_colors.count(color)) color++;
        colors[name] = color;
    }

    int max_color = 0;
    for (auto const& [name, color] : colors) max_color = std::max(max_color, color);

    sets.resize(max_color + 1);
    for (auto const& [name, color] : colors) {
        sets[color].push_back(name);
    }

    return sets;
}

float OptimizationKernels::calculate_distance(const std::string& u, const std::string& v) {
    auto pos1 = engine_->get_node_position(u);
    auto pos2 = engine_->get_node_position(v);
    float dx = pos1.first - pos2.first;
    float dy = pos1.second - pos2.second;
    return std::sqrt(dx*dx + dy*dy);
}

void OptimizationKernels::apply_forces(float attraction_k, float repulsion_k) {
    auto names = engine_->get_all_node_names();
    size_t n = names.size();
    
    std::vector<std::pair<float, float>> forces(n, {0.0f, 0.0f});

    // 1. Repulsive forces (All pairs - O(N^2) for now, manageable for small designs)
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            
            auto p1 = engine_->get_node_position(names[i]);
            auto p2 = engine_->get_node_position(names[j]);
            
            float dx = p1.first - p2.first;
            float dy = p1.second - p2.second;
            float dist_sq = dx*dx + dy*dy + 0.01f; // Avoid division by zero
            float dist = std::sqrt(dist_sq);
            
            // Repulsion formula: FR = (k^2 / d)
            float fr = (repulsion_k * repulsion_k) / dist;
            forces[i].first += (dx / dist) * fr;
            forces[i].second += (dy / dist) * fr;
        }
    }

    // 2. Attractive forces (Edges only - O(E))
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        auto neighbors = engine_->get_neighbors(names[i]);
        auto p1 = engine_->get_node_position(names[i]);
        
        for (const auto& neighbor : neighbors) {
            auto p2 = engine_->get_node_position(neighbor);
            
            float dx = p1.first - p2.first;
            float dy = p1.second - p2.second;
            float dist = std::sqrt(dx*dx + dy*dy + 0.01f);
            
            // Attraction formula: FA = (d^2 / k)
            float fa = (dist * dist) / attraction_k;
            forces[i].first -= (dx / dist) * fa;
            forces[i].second -= (dy / dist) * fa;
        }
    }

    // 3. Apply displacement
    for (size_t i = 0; i < n; ++i) {
        auto p = engine_->get_node_position(names[i]);
        // Limit max displacement
        float fx = std::clamp(forces[i].first, -50.0f, 50.0f);
        float fy = std::clamp(forces[i].second, -50.0f, 50.0f);
        engine_->set_node_position(names[i], p.first + fx, p.second + fy);
    }
}

} // namespace si
