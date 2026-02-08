#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include "graph_engine.hpp"

namespace si {

struct PlacementConfig {
    int iterations = 1000;
    float initial_temp = 1000.0f;
    float cooling_rate = 0.95f;
    int threads = 4;
    float area_width = 1000.0f;
    float area_height = 1000.0f;
};

class OptimizationKernels {
public:
    OptimizationKernels(std::shared_ptr<GraphEngine> engine);
    
    // Simulated Annealing for Placement
    // Updates node positions in the graph engine directly
    void run_global_placement(const PlacementConfig& config);
    
    // Independent set finding for parallel moves
    std::vector<std::vector<std::string>> find_independent_sets();

    // Cost function calculation (Total Wirelength)
    float calculate_hpwl();

private:
    std::shared_ptr<GraphEngine> engine_;
    
    // Internal helper for single threaded annealing step
    void anneal_step(float temp, float width, float height);

    // Phase 5: Force-Directed Logic
    void apply_forces(float attraction_k, float repulsion_k);
    float calculate_distance(const std::string& u, const std::string& v);
};

} // namespace si
