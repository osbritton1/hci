cmake_minimum_required(VERSION 3.8)

project(libhci VERSION 0.1.0 DESCRIPTION "C library for PySCF HCI implementation")
include(GNUInstallDirs)
add_library(libhci SHARED "hci_rank.c" "hci_rank.h" "hci_store.c" "hci_store.h" "hci_enlarge.c" "hci_enlarge.h" "hci_contract.c" "hci_contract.h")
set_target_properties(libhci PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# add_executable(scratch "test.c" "hci_rank.c" "hci_rank.h" "hci_store.c" "hci_store.h" "hci_enlarge.c" "hci_enlarge.h" "hci_contract.c" "hci_contract.h")