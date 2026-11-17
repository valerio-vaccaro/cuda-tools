cmake_minimum_required(VERSION 3.24)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
# Build for the GPU in this machine unless told otherwise
# (e.g. -DCMAKE_CUDA_ARCHITECTURES="75;86").
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
# Lets older CUDA toolkits work with a newer host g++ than they officially know.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Wno-deprecated-gpu-targets"
    CACHE STRING "" FORCE)

project(two_mnemonic LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Embed the BIP39 English word list into the binary.
set(WORDLIST ${CMAKE_SOURCE_DIR}/data/english.txt)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${WORDLIST})
file(SHA256 ${WORDLIST} WORDLIST_HASH)
if(NOT WORDLIST_HASH STREQUAL "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda")
  message(FATAL_ERROR "data/english.txt is not the official BIP39 English word list")
endif()
file(STRINGS ${WORDLIST} WORDS)
set(WORDLIST_H "// Generated from data/english.txt - do not edit.\nstatic const char* const BIP39_WORDS[2048] = {\n")
foreach(W IN LISTS WORDS)
  string(APPEND WORDLIST_H "\"${W}\",\n")
endforeach()
string(APPEND WORDLIST_H "};\n")
file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/wordlist.h CONTENT "${WORDLIST_H}" @ONLY)

add_executable(two-mnemonic
  src/main.cpp
  src/bip39.cpp
  src/gpu_search.cu)
target_include_directories(two-mnemonic PRIVATE src ${CMAKE_BINARY_DIR}/generated)
target_compile_options(two-mnemonic PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

find_package(Threads REQUIRED)
target_link_libraries(two-mnemonic PRIVATE Threads::Threads)

enable_testing()
add_test(NAME selftest COMMAND two-mnemonic --selftest)
