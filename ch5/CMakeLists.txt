cmake_minimum_required(VERSION 3.20)
project(ch5)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS})

add_executable(ch5 stereoVision.cpp)
target_link_libraries(ch5
        ${OpenCV_LIBRARIES}
        ${Pangolin_LIBRARIES} fmt)
