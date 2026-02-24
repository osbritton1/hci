cmake_minimum_required(VERSION 3.8)

project("libhci")

add_executable(libhci "test.c" "hci_rank.c" "hci_rank.h" "hci_store.c" "hci_store.h" "hci_enlarge.c" "hci_enlarge.h" "hci_contract.c" "hci_contract.h")