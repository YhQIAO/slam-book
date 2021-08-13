cmake_minimum_required(VERSION 3.20)
# learn how to use ceres
# the ceres on my linux computer is compiled by myself
# and installed in /usr/local
project(usage_of_ceres)

set(CMAKE_CXX_STANDARD 14)

# find ceres and include header files
find_package(Ceres)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp )
target_link_libraries(ceresCurveFitting ${OpenCV_LIBS} Ceres::ceres)

add_executable(official_demo official_demo.cpp)
target_link_libraries(official_demo Ceres::ceres)