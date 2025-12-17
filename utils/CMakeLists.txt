# Utils executables

add_executable(fvecs_to_bin fvecs_to_bin.cpp)
add_executable(ivecs_to_bin ivecs_to_bin.cpp)
add_executable(compute_groundtruth compute_groundtruth.cpp)

target_include_directories(fvecs_to_bin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(ivecs_to_bin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(compute_groundtruth
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(compute_groundtruth
    PRIVATE
        OpenMP::OpenMP_CXX
)

