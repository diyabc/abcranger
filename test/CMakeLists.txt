cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

project(abcrangertest VERSION 0.2.0 LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(../src)

if(!Boost_USE_STATIC_LIBS)
    add_definitions(-DBOOST_TEST_DYN_LINK)

### HDF5
set(HDF5_USE_STATIC_LIBRARIES 1)
if(HDF5_DIR_ROOT)
    set(ENV{HDF5_DIR} "${HDF5_DIR_ROOT}/${CMAKE_BUILD_TYPE}/cmake")
endif()
message("HDF5_DIR : $ENV{HDF5_DIR}")
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/HighFive/include)

# add_definitions(-DBOOST_TEST_DYN_LINK)

set(DATADIR ${CMAKE_SOURCE_DIR}/test/data)

file(GLOB cxx_files *.cpp)
foreach(filename ${cxx_files})
    get_filename_component(target ${filename} NAME_WE)
    add_executable(${target} ${filename})
    target_link_libraries(${target} Boost::unit_test_framework)
    add_test(NAME ${target} COMMAND ${target} WORKING_DIRECTORY ${DATADIR})
endforeach(filename)

target_link_libraries(readreftable-test ${HDF5_CXX_LIBRARIES})
# target_link_libraries(readreftable-test hdf5::hdf5_cpp-static)
target_compile_definitions(readreftable-test PRIVATE ${HDF5_DEFINITIONS})

target_sources(forestclass-test PUBLIC ${RANGER_SOURCE_DIR}/ForestClassification.cpp)
target_sources(forestreg-test PUBLIC ${RANGER_SOURCE_DIR}/ForestRegression.cpp)
