cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
set(PROJECT_NAME EigenPCA)
project (${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Debug)


find_package( Eigen3 REQUIRED )
include_directories(EIGEN3_INCLUDE_DIR)

add_library(${PROJECT_NAME} SHARED pca.cpp pca.h)
add_executable(${PROJECT_NAME}_test pca_run.cpp)
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME})