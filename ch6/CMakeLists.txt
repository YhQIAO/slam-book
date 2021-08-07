cmake_minimum_required(VERSION 3.20)
project(ch6)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_CXX_STANDARD 11)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${g2o_INCLUDE_DIRS})

add_executable(ch6 g2oCurveFitting.cpp)
target_link_libraries(ch6
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
