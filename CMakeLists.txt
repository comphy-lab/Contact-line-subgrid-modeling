cmake_minimum_required(VERSION 3.12)
project(GLE_Solver_SUNDIALS C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(MPI)

# Find SUNDIALS
find_path(SUNDIALS_INCLUDE_DIR 
    NAMES sundials/sundials_config.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES sundials
)

find_library(SUNDIALS_CVODE_LIB
    NAMES sundials_cvode
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

find_library(SUNDIALS_KINSOL_LIB
    NAMES sundials_kinsol
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

find_library(SUNDIALS_NVECSERIAL_LIB
    NAMES sundials_nvecserial
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

find_library(SUNDIALS_CORE_LIB
    NAMES sundials_core
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

find_library(SUNDIALS_SUNLINSOLDENSE_LIB
    NAMES sundials_sunlinsoldense
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

find_library(SUNDIALS_SUNMATRIXDENSE_LIB
    NAMES sundials_sunmatrixdense
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /opt/homebrew/lib
)

# Check if SUNDIALS was found
if(NOT SUNDIALS_INCLUDE_DIR OR NOT SUNDIALS_CVODE_LIB OR NOT SUNDIALS_KINSOL_LIB OR NOT SUNDIALS_NVECSERIAL_LIB OR NOT SUNDIALS_CORE_LIB)
    message(WARNING "SUNDIALS not found. Please install SUNDIALS library (>= 6.0) with CVODE and KINSOL modules.")
    message(STATUS "On Ubuntu/Debian: sudo apt-get install libsundials-dev")
    message(STATUS "On CentOS/RHEL: sudo yum install sundials-devel")
    message(STATUS "Or build from source: https://computing.llnl.gov/projects/sundials")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
if(SUNDIALS_INCLUDE_DIR)
    include_directories(${SUNDIALS_INCLUDE_DIR})
endif()

# Add math library
find_library(MATH_LIB m)

# Source files
set(GLE_MATH_SOURCES
    src/gle_math.c
)

set(GLE_SOLVER_SOURCES
    src/GLE_solver-SUNDIALS.c
    src/csv_output.c
)

# Create library for mathematical functions
add_library(gle_math STATIC ${GLE_MATH_SOURCES})
target_link_libraries(gle_math ${MATH_LIB})

# Create main solver library
if(SUNDIALS_CVODE_LIB AND SUNDIALS_KINSOL_LIB AND SUNDIALS_NVECSERIAL_LIB AND SUNDIALS_CORE_LIB)
    add_library(gle_solver STATIC ${GLE_SOLVER_SOURCES})
    target_link_libraries(gle_solver 
        gle_math
        ${SUNDIALS_CVODE_LIB}
        ${SUNDIALS_KINSOL_LIB}
        ${SUNDIALS_NVECSERIAL_LIB}
        ${SUNDIALS_SUNLINSOLDENSE_LIB}
        ${SUNDIALS_SUNMATRIXDENSE_LIB}
        ${SUNDIALS_CORE_LIB}
        ${MATH_LIB}
    )
    
    # Link MPI if available and SUNDIALS was built with MPI
    if(MPI_FOUND)
        target_link_libraries(gle_solver ${MPI_C_LIBRARIES})
        target_include_directories(gle_solver PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_compile_definitions(gle_solver PRIVATE SUNDIALS_MPI_ENABLED=1)
    endif()

    # Create main executable
    add_executable(gle_solver_main src/main.c)
    target_link_libraries(gle_solver_main gle_solver)

    # Enable testing
    enable_testing()

    # Add C unit tests
    add_executable(test_gle_math_c test/c_tests/test_gle_math_c.c)
    target_link_libraries(test_gle_math_c gle_math)
    add_test(NAME test_gle_math_c COMMAND test_gle_math_c)

    add_executable(test_gle_solver_c test/c_tests/test_gle_solver_c.c)
    target_link_libraries(test_gle_solver_c gle_solver)
    add_test(NAME test_gle_solver_c COMMAND test_gle_solver_c)
else()
    message(WARNING "SUNDIALS libraries not found. Skipping solver and main executable build.")
    message(STATUS "Only mathematical functions library will be built.")
    
    # Still build math library and its tests
    enable_testing()
    add_executable(test_gle_math_c test/c_tests/test_gle_math_c.c)
    target_link_libraries(test_gle_math_c gle_math)
    add_test(NAME test_gle_math_c COMMAND test_gle_math_c)
endif()

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG")

# Print configuration summary
message(STATUS "=== GLE Solver SUNDIALS Configuration ===")
message(STATUS "SUNDIALS Include Dir: ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "SUNDIALS CVODE Lib: ${SUNDIALS_CVODE_LIB}")
message(STATUS "SUNDIALS KINSOL Lib: ${SUNDIALS_KINSOL_LIB}")
message(STATUS "SUNDIALS NVECSERIAL Lib: ${SUNDIALS_NVECSERIAL_LIB}")
message(STATUS "SUNDIALS CORE Lib: ${SUNDIALS_CORE_LIB}")
message(STATUS "SUNDIALS SUNLINSOLDENSE Lib: ${SUNDIALS_SUNLINSOLDENSE_LIB}")
message(STATUS "SUNDIALS SUNMATRIXDENSE Lib: ${SUNDIALS_SUNMATRIXDENSE_LIB}")
message(STATUS "Math Library: ${MATH_LIB}")
message(STATUS "==========================================")
