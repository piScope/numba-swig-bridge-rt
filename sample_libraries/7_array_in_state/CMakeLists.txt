cmake_minimum_required(VERSION 3.24)
project(numba_swig_bridge_sample_library LANGUAGES CXX)

add_library(nsb_sample7_core STATIC sample7.cpp)
target_compile_features(nsb_sample7_core PUBLIC cxx_std_17)
target_include_directories(nsb_sample7_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(nsb_sample7_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
