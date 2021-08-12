cmake_minimum_required(VERSION 3.20)
project(ch6)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS})

#add_executable(ceresCurveFitting ceresCurveFitting.cpp)
#target_link_libraries(ceresCurveFitting
#        ${OpenCV_LIBS}
#        ${CERES_LIBRARIES}
#        )

add_executable(g2oCurveFitting g2oCurveFitting.cpp)
target_link_libraries(g2oCurveFitting ${OpenCV_LIBS}
        ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} )
