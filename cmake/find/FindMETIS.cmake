# cmake/FindMETIS.cmake

find_path(METIS_INCLUDE_DIR NAMES metis.h)
find_library(METIS_LIBRARY NAMES metis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS DEFAULT_MSG METIS_LIBRARY METIS_INCLUDE_DIR)

if(METIS_FOUND)
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
    
    # Create an imported target (Best Practice)
    if(NOT TARGET METIS::METIS)
        add_library(METIS::METIS UNKNOWN IMPORTED)
        set_target_properties(METIS::METIS PROPERTIES
            IMPORTED_LOCATION "${METIS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
        )
    endif()
endif()