cmake_minimum_required(VERSION 3.22)

project(${SKBUILD_PROJECT_NAME} LANGUAGES C VERSION 0.1.0 DESCRIPTION "HCI implementation for PySCF")
set(CMAKE_C_FLAGS "-O3 -march=native")

add_subdirectory(${SKBUILD_PROJECT_NAME}/lib)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION lib)