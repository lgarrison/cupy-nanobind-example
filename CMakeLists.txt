cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

enable_language(CUDA)

nanobind_add_module(cuda src/example/cuda.cu)

install(TARGETS cuda LIBRARY DESTINATION example)
