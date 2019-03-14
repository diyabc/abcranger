cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

project(abcrangertest VERSION 0.2.0 LANGUAGES CXX)

include_directories(../src)

if(NOT RFTEST_TOLERANCE)
    set(RFTEST_TOLERANCE 1e-3)
endif()

add_definitions(-DRFTEST_TOLERANCE=${RFTEST_TOLERANCE})

### HDF5 et HighFive
### HDF5
set(HDF5_USE_STATIC_LIBRARIES 1)
if(HDF5_DIR_ROOT)
    set(ENV{HDF5_DIR} "${HDF5_DIR_ROOT}/${CMAKE_BUILD_TYPE}/cmake")
endif()
message("HDF5_DIR : $ENV{HDF5_DIR}")
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

#find_package(hdf5 CONFIG REQUIRED)
find_package(HighFive CONFIG REQUIRED)

find_package(Catch2 CONFIG REQUIRED)
include(Catch)
set(DATADIR ${CMAKE_SOURCE_DIR}/test/data)

file(GLOB cxx_files *.cpp)
foreach(filename ${cxx_files})
    get_filename_component(target ${filename} NAME_WE)
    add_executable(${target} ${filename})
    target_link_libraries(${target} Catch2::Catch2)
    catch_discover_tests(${target} WORKING_DIRECTORY ${DATADIR})
endforeach(filename)

target_link_libraries(readreftable-test HighFive hdf5::hdf5_cpp-shared)

target_sources(forestclass-test PUBLIC ${RANGER_SOURCE_DIR}/ForestClassification.cpp)
target_sources(forestreg-test PUBLIC ${RANGER_SOURCE_DIR}/ForestRegression.cpp)
