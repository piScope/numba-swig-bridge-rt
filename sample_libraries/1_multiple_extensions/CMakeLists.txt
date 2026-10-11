cmake_minimum_required(VERSION 3.24)
project(numba_swig_bridge_sample_library LANGUAGES CXX)

add_library(nsb_sample1_core STATIC sample1.cpp)
target_compile_features(nsb_sample1_core PUBLIC cxx_std_17)
target_include_directories(nsb_sample1_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(nsb_sample1_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
