cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

project(abcrangertest VERSION 0.2.0 LANGUAGES CXX)

include(CTest)
enable_testing()

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories(../src)

### HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/HighFive/include)


if (WIN32)
    set(TestENV "PATH=${Boost_LIBRARY_DIRS}\\;${HDF5_ROOT}/bin\\;${LAPACK_BIN}\\;$ENV{PATH}")
# set(TestENV "PATH=${BOOST_TEST_BIN};${HDF5_BIN};${LAPACK_BIN};$ENV{PATH}")
endif()

set(DATADIR data)
link_libraries(Boost::unit_test_framework)

add_executable(Eigen eigen-test.cpp)
target_link_libraries(Eigen)
add_test(NAME EigenTest COMMAND Eigen WORKING_DIRECTORY ${DATADIR})
set_tests_properties(EigenTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(LDAEigen lda-eigen-test.cpp)
target_link_libraries(LDAEigen)
add_test(NAME LDAEigenTest COMMAND LDAEigen WORKING_DIRECTORY ${DATADIR})
set_tests_properties(LDAEigenTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(PLSEigen pls-eigen-test.cpp)
target_link_libraries(PLSEigen)
add_test(NAME PLSEigenTest COMMAND PLSEigen WORKING_DIRECTORY ${DATADIR})
set_tests_properties(PLSEigenTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(readreftableTest readreftable-test.cpp)
target_link_libraries(readreftableTest ${HDF5_LIBRARIES})
target_compile_definitions(readreftableTest PRIVATE ${HDF5_DEFINITIONS})
add_test(NAME ReadRF COMMAND readreftableTest WORKING_DIRECTORY ${DATADIR})
set_tests_properties(ReadRF PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(matutils matutils-test.cpp)
target_link_libraries(matutils)
add_test(NAME MatutilsTest COMMAND matutils WORKING_DIRECTORY ${DATADIR})
set_tests_properties(MatutilsTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(ForestClass forestclass-test.cpp ${RANGER_SOURCE_DIR}/ForestClassification.cpp)
target_link_libraries(ForestClass)
add_test(NAME ForestClassTest COMMAND ForestClass WORKING_DIRECTORY ${DATADIR})
set_tests_properties(ForestClassTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(ForestOnlineClass forestonlineclass-test.cpp)
target_link_libraries(ForestOnlineClass)
add_test(NAME ForestOnlineClassTest COMMAND ForestOnlineClass WORKING_DIRECTORY ${DATADIR})
set_tests_properties(ForestOnlineClassTest PROPERTIES ENVIRONMENT "${TestENV}")


add_executable(ForestReg forestreg-test.cpp ${RANGER_SOURCE_DIR}/ForestRegression.cpp)
target_link_libraries(ForestReg)
add_test(NAME ForestRegTest COMMAND ForestReg WORKING_DIRECTORY ${DATADIR})
set_tests_properties(ForestRegTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(ForestOnlineReg forestonlinereg-test.cpp)
target_link_libraries(ForestOnlineReg)
add_test(NAME ForestOnlineRegTest COMMAND ForestOnlineReg WORKING_DIRECTORY ${DATADIR})
set_tests_properties(ForestOnlineRegTest PROPERTIES ENVIRONMENT "${TestENV}")