cmake_minimum_required(VERSION 3.24)
project(nsb_sample3 LANGUAGES CXX)
# A minimal base/derived library for the setuptools demonstration.
add_library(nsb_sample3_core STATIC inheritance.cpp)
target_compile_features(nsb_sample3_core PUBLIC cxx_std_17)
target_include_directories(nsb_sample3_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(nsb_sample3_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
