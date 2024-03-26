cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Export compile commants to help VS Code.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  "elasticnet_af"
  VERSION 0.1
)

# Register git hooks directory.
execute_process(
  COMMAND "git" "config" "core.hooksPath" "${CMAKE_CURRENT_SOURCE_DIR}/git-hooks"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# --------------
# Find Libraries
# --------------

find_package(ArrayFire REQUIRED)
find_package(fmt REQUIRED)

# --------------
# Compiler Flags
# --------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wmissing-field-initializers -pedantic ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# ---------------
# Compile targets
# ---------------

add_executable(example example.cpp)
target_link_libraries(example elasticnet_af)
