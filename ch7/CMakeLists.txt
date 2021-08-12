cmake_minimum_required(VERSION 3.20)
project(ch7)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3)
include_directories(Eigen3_INCLUDE_DIRS)

# 添加Pangolin依赖
find_package( Pangolin )
include_directories( ${Pangolin_INCLUDE_DIRS} )
# opencv
find_package(OpenCV REQUIRED)
include_directories(OpenCV_INCLUDE_DIRS)
#g2o
find_package(g2o)
include_directories(G2O_INCLUDE_DIRS)

find_package(Ceres)

add_executable(orb_cv orb_cv.cpp)
target_link_libraries(orb_cv ${OpenCV_LIBS})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS} ${Pangolin_LIBRARIES})

add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d
        ${OpenCV_LIBS}
        g2o_core g2o_stuff
        fmt) # when use sophus remember to use fmt

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS}
        g2o_core g2o_stuff
        fmt)

add_executable(my_gaussNewton_3d2d my_gaussNewton_3d2d.cpp)
target_link_libraries(my_gaussNewton_3d2d
        ${OpenCV_LIBS}
        g2o_core g2o_stuff
        fmt)