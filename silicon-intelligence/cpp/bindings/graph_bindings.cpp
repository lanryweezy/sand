#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "graph_engine.hpp"
#include "graph_engine.hpp"
#include "rtl_transformer.hpp"
#include "optimization_kernels.hpp"

namespace py = pybind11;

PYBIND11_MODULE(silicon_intelligence_cpp, m) {
    m.doc() = "Silicon Intelligence C++ Core acceleration module";

    // Enums
    py::enum_<si::NodeType>(m, "NodeType")
        .value("CELL", si::NodeType::CELL)
        .value("MACRO", si::NodeType::MACRO)
        .value("PORT", si::NodeType::PORT)
        .value("CLOCK", si::NodeType::CLOCK)
        .value("POWER", si::NodeType::POWER)
        .value("SIGNAL", si::NodeType::SIGNAL)
        .export_values();

    py::enum_<si::EdgeType>(m, "EdgeType")
        .value("CONNECTION", si::EdgeType::CONNECTION)
        .value("PHYSICAL_PROXIMITY", si::EdgeType::PHYSICAL_PROXIMITY)
        .value("TIMING_DEPENDENCY", si::EdgeType::TIMING_DEPENDENCY)
        .value("POWER_FEED", si::EdgeType::POWER_FEED)
        .export_values();

    // Attributes
    py::class_<si::NodeAttributes>(m, "NodeAttributes")
        .def(py::init<>())
        .def_readwrite("node_type", &si::NodeAttributes::node_type)
        .def_readwrite("cell_type", &si::NodeAttributes::cell_type)
        .def_readwrite("area", &si::NodeAttributes::area)
        .def_readwrite("power", &si::NodeAttributes::power)
        .def_readwrite("delay", &si::NodeAttributes::delay)
        .def_readwrite("capacitance", &si::NodeAttributes::capacitance)
        .def_readwrite("position", &si::NodeAttributes::position)
        .def_readwrite("region", &si::NodeAttributes::region)
        .def_readwrite("voltage_domain", &si::NodeAttributes::voltage_domain)
        .def_readwrite("clock_domain", &si::NodeAttributes::clock_domain)
        .def_readwrite("is_clock_root", &si::NodeAttributes::is_clock_root)
        .def_readwrite("is_macro", &si::NodeAttributes::is_macro)
        .def_readwrite("estimated_congestion", &si::NodeAttributes::estimated_congestion)
        .def_readwrite("timing_criticality", &si::NodeAttributes::timing_criticality);

    py::class_<si::EdgeAttributes>(m, "EdgeAttributes")
        .def(py::init<>())
        .def_readwrite("edge_type", &si::EdgeAttributes::edge_type)
        .def_readwrite("resistance", &si::EdgeAttributes::resistance)
        .def_readwrite("capacitance", &si::EdgeAttributes::capacitance)
        .def_readwrite("delay", &si::EdgeAttributes::delay)
        .def_readwrite("length", &si::EdgeAttributes::length)
        .def_readwrite("layers_used", &si::EdgeAttributes::layers_used)
        .def_readwrite("congestion", &si::EdgeAttributes::congestion)
        .def_readwrite("capacity", &si::EdgeAttributes::capacity);

    // Graph Engine
    py::class_<si::GraphEngine, std::shared_ptr<si::GraphEngine>>(m, "GraphEngine")
        .def(py::init([]() { return std::make_shared<si::GraphEngine>(); }))
        .def("add_node", &si::GraphEngine::add_node)
        .def("has_node", &si::GraphEngine::has_node)
        .def("update_node", &si::GraphEngine::update_node)
        .def("add_edge", &si::GraphEngine::add_edge)
        .def("num_nodes", &si::GraphEngine::num_nodes)
        .def("num_edges", &si::GraphEngine::num_edges)
        .def("get_timing_critical_nodes", &si::GraphEngine::get_timing_critical_nodes)
        .def("get_node_position", &si::GraphEngine::get_node_position)
        .def("set_node_position", &si::GraphEngine::set_node_position)
        .def("get_neighbors", &si::GraphEngine::get_neighbors)
        .def("get_all_node_names", &si::GraphEngine::get_all_node_names);

    // RTL Transformer Bindings
    py::class_<silicon_intelligence::Module, std::shared_ptr<silicon_intelligence::Module>>(m, "Module")
        .def(py::init<std::string>())
        .def_readwrite("name", &silicon_intelligence::Module::name)
        .def("to_verilog", &silicon_intelligence::Module::to_verilog);

    py::class_<silicon_intelligence::RTLTransformer>(m, "RTLTransformer")
        .def(py::init<>())
        .def("parse_verilog", &silicon_intelligence::RTLTransformer::parse_verilog)
        .def("parse_file", &silicon_intelligence::RTLTransformer::parse_file)
        .def("add_pipeline_stage", &silicon_intelligence::RTLTransformer::add_pipeline_stage)
        .def("insert_clock_gate", &silicon_intelligence::RTLTransformer::insert_clock_gate)
        .def("generate_verilog", &silicon_intelligence::RTLTransformer::generate_verilog)
        .def("apply_logic_merging", &silicon_intelligence::RTLTransformer::apply_logic_merging)
        .def("apply_fanout_buffering", &silicon_intelligence::RTLTransformer::apply_fanout_buffering)
        .def("apply_input_isolation", &silicon_intelligence::RTLTransformer::apply_input_isolation);

    // Optimization Kernels Bindings
    py::class_<si::PlacementConfig>(m, "PlacementConfig")
        .def(py::init<>())
        .def_readwrite("iterations", &si::PlacementConfig::iterations)
        .def_readwrite("initial_temp", &si::PlacementConfig::initial_temp)
        .def_readwrite("cooling_rate", &si::PlacementConfig::cooling_rate)
        .def_readwrite("threads", &si::PlacementConfig::threads)
        .def_readwrite("area_width", &si::PlacementConfig::area_width)
        .def_readwrite("area_height", &si::PlacementConfig::area_height);

    py::class_<si::OptimizationKernels, std::shared_ptr<si::OptimizationKernels>>(m, "OptimizationKernels")
        .def(py::init([](std::shared_ptr<si::GraphEngine> engine) {
            return std::make_shared<si::OptimizationKernels>(engine);
        }))
        .def("run_global_placement", &si::OptimizationKernels::run_global_placement)
        .def("calculate_hpwl", &si::OptimizationKernels::calculate_hpwl)
        .def("apply_forces", &si::OptimizationKernels::apply_forces);
}
