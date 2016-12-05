cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

set(PROJECT_NAME EigenPCA)
project (${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Debug)

find_package( Eigen3 REQUIRED )
include_directories(EIGEN3_INCLUDE_DIR)

add_library(${PROJECT_NAME} SHARED src/pca.cpp include/pca.hpp)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
