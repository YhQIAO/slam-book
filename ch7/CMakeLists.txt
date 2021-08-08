cmake_minimum_required(VERSION 3.20)
project(ch7)

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INLCUDE_DIRS})

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )

add_executable(orb_cv orb_cv.cpp triangulation.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
