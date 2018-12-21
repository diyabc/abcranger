cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

project(abcrangertest VERSION 0.2.0 LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(../src)

### HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/HighFive/include)


if (WIN32)
    set(TestENV "PATH=${Boost_LIBRARY_DIRS}\\;${HDF5_ROOT}/bin\\;${LAPACK_BIN}\\;$ENV{PATH}")
endif()

set(DATADIR ${CMAKE_SOURCE_DIR}/test/data)

file(GLOB cxx_files *.cpp)
foreach(filename ${cxx_files})
    get_filename_component(target ${filename} NAME_WE)
    add_executable(${target} ${filename})
    target_link_libraries(${target} Boost::unit_test_framework)
    add_test(NAME ${target} COMMAND ${target} WORKING_DIRECTORY ${DATADIR})
    set_tests_properties(${target} PROPERTIES ENVIRONMENT "${TestENV}")
endforeach(filename)

target_link_libraries(readreftable-test ${HDF5_LIBRARIES})
target_compile_definitions(readreftable-test PRIVATE ${HDF5_DEFINITIONS})

target_sources(forestclass-test PUBLIC ${RANGER_SOURCE_DIR}/ForestClassification.cpp)
target_sources(forestreg-test PUBLIC ${RANGER_SOURCE_DIR}/ForestRegression.cpp)
