cmake_minimum_required(VERSION 3.15)
project(hw02)

set(CMAKE_CXX_STANDARD 14)

add_executable(hw02 main.cpp input_data.cpp input_data.h output_data.cpp output_data.h bayes_classifier.cpp bayes_classifier.h beta_distribution.cpp beta_distribution.h)