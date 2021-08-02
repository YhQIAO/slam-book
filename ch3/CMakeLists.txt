cmake_minimum_required(VERSION 3.17)
project(ch3)

set(CMAKE_CXX_STANDARD 11)
include_directories("/usr/include/eigen3")

add_executable(ch3 main.cpp)