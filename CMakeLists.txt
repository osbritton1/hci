cmake_minimum_required(VERSION 3.22)

project(${SKBUILD_PROJECT_NAME} VERSION 0.1.0 DESCRIPTION "HCI implementation for PySCF")

add_subdirectory(${SKBUILD_PROJECT_NAME}/lib)

install(TARGETS ${SKBUILD_PROJECT_NAME} DESTINATION "lib")