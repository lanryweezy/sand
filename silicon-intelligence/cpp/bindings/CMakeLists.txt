# CMake for Silicon Intelligence Python Bindings
pybind11_add_module(silicon_intelligence_cpp
    graph_bindings.cpp
)

target_link_libraries(silicon_intelligence_cpp PRIVATE si_core)
