cmake_minimum_required(VERSION 3.14)
project(hydramarker_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Dependencies
# ============================================================

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)

# ============================================================
# Include directories
# ============================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================
# Source files
# ============================================================

set(HYDRAMARKER_SOURCES

    # Core
    src/marker_field.cpp
    src/checkerboard_detector.cpp
    src/dot_detector.cpp
    src/patch_extractor.cpp
    src/patch_decoder.cpp
    src/marker_geometry.cpp
    src/correspondence_builder.cpp

    # Marker generation
    src/generator_HydraMarker.cpp

    # Checkerboard pipeline
    src/corner_detection.cpp
    src/corner_refinement.cpp
    src/lattice_model.cpp
    src/grid_builder.cpp
    src/lk_tracker.cpp
    src/tracking_validator.cpp
    src/geometry_utils.cpp
)

# ============================================================
# Python module
# ============================================================

pybind11_add_module(hydramarker_cpp
    bindings.cpp
    ${HYDRAMARKER_SOURCES}
)

target_link_libraries(hydramarker_cpp
    PRIVATE
    ${OpenCV_LIBS}
)

# ============================================================
# Example executable
# ============================================================

add_executable(example
    example.cpp
    ${HYDRAMARKER_SOURCES}
)

target_link_libraries(example
    PRIVATE
    ${OpenCV_LIBS}
)
