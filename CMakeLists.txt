cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_STANDARD 14)             # available options are [98, 11, 14, 17. 20]

project(geodesic-sandbox)

### vtk
find_package(VTK REQUIRED)
if (NOT VTK_FOUND)
  message(FATAL_ERROR "VTK is required, but could not be found! Make sure path to VTKConfig.cmake is included in CMAKE_PREFIX_PATH")
endif()
include(${VTK_USE_FILE})

### vxl
FIND_PACKAGE(VXL REQUIRED)
IF(VXL_FOUND)
  # this adds deprecated VXL variables to the cmake namespace (TODO: use the new ones)
  INCLUDE (${VXL_CMAKE_DIR}/UseVXL.cmake)
ELSE(VXL_FOUND)
  MESSAGE(SEND_ERROR "VXL is required, but could not be found! Make sure path to VXLConfig.cmake is included in CMAKE_PREFIX_PATH")
ENDIF(VXL_FOUND)

### ITK
FIND_PACKAGE(ITK REQUIRED)
IF (ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE()
  MESSAGE(SEND_ERROR "ITK (The Insight Toolkit) is required, but could not be found! Make sure path to ITKConfig.cmake is included in CMAKE_PREFIX_PATH")
ENDIF()

### Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(geotest src/main.cpp)
target_include_directories(geotest PUBLIC src/)
target_link_libraries(geotest ${VTK_LIBRARIES})
