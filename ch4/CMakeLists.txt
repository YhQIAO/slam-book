cmake_minimum_required(VERSION 3.20)
project(ch4)

set(CMAKE_CXX_STANDARD 14)

find_package(Sophus REQUIRED)
find_package(Pangolin)
include_directories(${Sophus_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(ch4 trajectoryError.cpp)
target_link_libraries(ch4 ${Sophus_LIBRARIES} fmt)
target_link_libraries(ch4 ${Pangolin_LIBRARIES})
