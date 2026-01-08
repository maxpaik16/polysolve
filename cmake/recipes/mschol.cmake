if(TARGET mschol::mschol)
    return()
endif()

message(STATUS "Third-party: creating target 'mschol::mschol'")

include(CPM)
CPMAddPackage(
    NAME mschol
    GIT_REPOSITORY https://github.com/maxpaik16/mschol
    GIT_TAG ff36036b07f92397ff8975c2503e4d845c6222e7
)

#add_library(mschol::mschol ALIAS mschol)
#target_link_libraries(mschol PRIVATE Eigen3::Eigen)
target_include_directories(mschol SYSTEM PUBLIC ${mschol_SOURCE_DIR}/src)