cmake_minimum_required(VERSION 3.17)
project(ch3)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRECTORIES})

add_executable(ch3  plotTrajectory.cpp)
target_link_libraries(ch3 ${Pangolin_LIBRARIES})