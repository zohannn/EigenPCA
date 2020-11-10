cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

set(PROJECT_NAME EigenPCA)
project (${PROJECT_NAME})

include_directories( EIGEN3_INCLUDE_DIR )

## Defult build type: Debug ##
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
## Debug definition: ##
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_definitions("-DDEBUG")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O0")


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_library(${PROJECT_NAME}d SHARED src/pca.cpp include/pca.hpp)
	install(TARGETS ${PROJECT_NAME}d DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
	add_library(${PROJECT_NAME} SHARED src/pca.cpp include/pca.hpp)
	install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)
else()
	MESSAGE(FATAL_ERROR "Please specify -DCMAKE_BUILD_TYPE=Debug|Release")
endif()


