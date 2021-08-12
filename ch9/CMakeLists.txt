cmake_minimum_required(VERSION 3.20)

project(ch9)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
#
#LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(g2o REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
#Find_Package(CSparse REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${g2o_INCLUDE_DIR}
        )
#
add_library(bal_common common.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
#
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common)
target_link_libraries(bundle_adjustment_g2o g2o_stuff g2o_core
        g2o_csparse_extension  cxsparse
        bal_common fmt)
