# CMake for Silicon Intelligence Core Libs
add_library(si_core STATIC
    graph_engine.cpp
    rtl_transformer.cpp
    optimization_kernels.cpp
    # rtl_ast.cpp
)

# Link against dependencies like Boost or TBB later
# target_link_libraries(si_core PRIVATE ...)
