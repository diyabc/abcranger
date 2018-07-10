cmake_minimum_required(VERSION 3.0.0)
project(abcranger VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

set(RANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/src)
include_directories("${RANGER_SOURCE_DIR}")
set(PROJECT_abcranger_SRCS
  ${RANGER_SOURCE_DIR}/Data.cpp
  ${RANGER_SOURCE_DIR}/utility.cpp
  a.cpp
  )

find_package (Threads REQUIRED)

find_package(HDF5 REQUIRED COMPONENTS CXX)
# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem regex)

if (WIN32)
    set(TestENV "PATH=${Boost_LIBRARY_DIRS}\\;${HDF5_ROOT}\\bin;$ENV{PATH}")
endif()

include_directories(Boost::boost)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/HighFive/include)

add_executable(abcranger main.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(abcranger Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)

add_executable(readreftableTest readreftable-test.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(readreftableTest Threads::Threads ${HDF5_LIBRARIES} Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
target_compile_definitions(readreftableTest PRIVATE ${HDF5_DEFINITIONS})
target_compile_features(readreftableTest PRIVATE cxx_thread_local)

# target_link_libraries(arraytest Boost::unit_test_framework)
add_test(NAME ReadRF COMMAND readreftableTest WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(ReadRF PROPERTIES ENVIRONMENT "${TestENV}")


add_executable(simpleregex regex-test.cpp)
target_link_libraries(simpleregex Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
# target_link_libraries(arraytest Boost::unit_test_framework)
add_test(SimpleRegex simpleregex)
set_tests_properties(SimpleRegex PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(matutils matutils-test.cpp readreftable.cpp)
target_link_libraries(matutils Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
# target_link_libraries(arraytest Boost::unit_test_framework)
add_test(NAME MatutilsTest COMMAND matutils WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(MatutilsTest PROPERTIES ENVIRONMENT "${TestENV}")

