set(CMAKE_VERBOSE_MAKEFILE ON)

cmake_minimum_required(VERSION 2.8.12)
project(augmentations_backend)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(pybind11)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

pybind11_add_module(augmentations_backend cpp/augmentations.cpp cpp/transformations.cpp)
target_link_libraries(augmentations_backend PUBLIC ${OpenCV_LIBS})
target_link_libraries(augmentations_backend PUBLIC ${CMAKE_THREAD_LIBS_INIT})
