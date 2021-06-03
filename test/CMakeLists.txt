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
# find_package(hdf5 CONFIG REQUIRED)
find_package(HDF5 COMPONENTS CXX)
find_package(HighFive CONFIG REQUIRED)

find_package(Catch2 CONFIG REQUIRED)
include(Catch)
set(DATADIR ${CMAKE_SOURCE_DIR}/test/data)

file(GLOB cxx_files *.cpp)
foreach(filename ${cxx_files})
    get_filename_component(target ${filename} NAME_WE)
    add_executable(${target} ${filename})
    target_link_libraries(${target} Catch2::Catch2)
    if(target MATCHES "^forest.*")
        target_link_libraries(${target} ${BOOSTLIBS} abcrangerlib fmt::fmt)
    endif()
    if(target MATCHES ".*eigen.*")
	target_link_libraries(${target} ${TBBLIB})
    endif()
    catch_discover_tests(${target} WORKING_DIRECTORY ${DATADIR})
endforeach(filename)

target_sources(readreftable-test PRIVATE ${ABCRANGER_SOURCE_DIR}/readreftable.cpp ${ABCRANGER_SOURCE_DIR}/tqdm.cpp)
target_sources(parse_parexpr-test PRIVATE ${ABCRANGER_SOURCE_DIR}/parse_parexpr.cpp)
target_sources(quantile-test PRIVATE ${ABCRANGER_SOURCE_DIR}/forestQuantiles.cpp)
target_sources(forestestimparam-ks-test PRIVATE ${ABCRANGER_SOURCE_DIR}/EstimParam.cpp)
target_sources(forestmodelchoice-ks-test PRIVATE ${ABCRANGER_SOURCE_DIR}/ModelChoice.cpp)
target_sources(pls-eigen-test PRIVATE ${ABCRANGER_SOURCE_DIR}/tqdm.cpp)

target_link_libraries(readreftable-test ${TBBLIB} ${BOOSTLIBS} HighFive hdf5::hdf5_cpp-static fmt::fmt)
target_link_libraries(forestestimparam-ks-test ${TBBLIB} cxxopts::cxxopts abcrangerlib fmt::fmt)
target_link_libraries(forestmodelchoice-ks-test ${TBBLIB} cxxopts::cxxopts abcrangerlib fmt::fmt)
target_link_libraries(cxxopts-test cxxopts::cxxopts)
target_link_libraries(pls-eigen-test ${TBBLIB} fmt::fmt)

target_sources(forestclass-test PRIVATE ${RANGER_SOURCE_DIR}/ForestClassification.cpp)
target_sources(forestreg-test PRIVATE ${RANGER_SOURCE_DIR}/ForestRegression.cpp)
