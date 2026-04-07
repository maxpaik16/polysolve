if(TARGET magma::magma)
    return()
endif()

message(STATUS "Third-party: creating target 'magma::magma'")

include(CPM)
set(GPU_TARGET "Ampere" CACHE STRING "Target GPU architecture for MAGMA")
add_compile_definitions(ADD_)

CPMAddPackage(
    NAME magma
    VERSION 2.10.0
    URL https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.10.0.tar.gz
    EXCLUDE_FROM_ALL True
    OPTIONS
        "GPU_TARGET ${GPU_TARGET}"
        "MAGMA_ENABLE_CUDA ON"     # Explicitly enforce CUDA backend
        "BUILD_SHARED_LIBS OFF"    # Build as static to embed directly
        "BUILD_TESTING OFF" # Skip tests to save build time
)
