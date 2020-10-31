cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 14)             # available options are [98, 11, 14, 17. 20]

project(geodesic-sandbox)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           OFF)
find_package(LIBIGL REQUIRED)

### vtk
find_package(VTK REQUIRED NO_MODULE)
if (NOT VTK_FOUND)
  message(FATAL_ERROR "VTK is required, but could not be found! Make sure path to VTKConfig.cmake is included in CMAKE_PREFIX_PATH")
endif()
include(${VTK_USE_FILE})

### Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

## openmp
find_package(OpenMP REQUIRED)


file(GLOB CPP_SRC
        "src/*.cpp"
        )
add_executable(geotest ${CPP_SRC})
target_include_directories(geotest PUBLIC src/)


target_link_libraries(geotest PUBLIC OpenMP::OpenMP_CXX ${VTK_LIBRARIES} igl::core)
