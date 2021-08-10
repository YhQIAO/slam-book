cmake_minimum_required(VERSION 3.17)
project(ch3)

set(CMAKE_CXX_STANDARD 11)
include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRECTORIES})
message("include: ${Pangolin_INCLUDE_DIRECTORIES}")

add_executable(ch3  plotTrajectory.cpp)
target_link_libraries(ch3 ${Pangolin_LIBRARIES})