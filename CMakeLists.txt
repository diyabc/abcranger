cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)


project(abcranger VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

set(RANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/src)
include_directories("${RANGER_SOURCE_DIR}")
set(PROJECT_abcranger_SRCS
${RANGER_SOURCE_DIR}/Data.cpp
${RANGER_SOURCE_DIR}/DataFloat.cpp
${RANGER_SOURCE_DIR}/Forest.cpp
${RANGER_SOURCE_DIR}/Tree.cpp
${RANGER_SOURCE_DIR}/TreeClassification.cpp
${RANGER_SOURCE_DIR}/TreeRegression.cpp
${RANGER_SOURCE_DIR}/utility.cpp
a.cpp
  )

add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING -D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING)
find_package(Threads REQUIRED)
# set(LAPACK_ROOT "C:/tools/miniconda3/pkgs/mkl-2018.0.3-1/Library/lib")
find_package(LAPACK)

# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem regex)

find_package(HDF5 REQUIRED COMPONENTS CXX)


if (WIN32)
    set(HDF5_ROOT "C:/Program Files/HDF_Group/HDF5/1.10.2")
    set(TestENV "PATH=${Boost_LIBRARY_DIRS}\\;${HDF5_ROOT}/bin\\;${LAPACK_BIN};$ENV{PATH}")
    # set(TestENV "PATH=${BOOST_TEST_BIN};${HDF5_BIN};${LAPACK_BIN};$ENV{PATH}")
endif()


include_directories(Boost::boost)
include_directories(AFTER ${CMAKE_SOURCE_DIR}/boost-numeric-bindings)
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/HighFive/include)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(${CMAKE_SOURCE_DIR}/eigen)
include_directories(${LAPACK_ROOT}/../../include)

add_executable(abcranger main.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(abcranger Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)

add_executable(readreftableTest readreftable-test.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(readreftableTest Threads::Threads ${HDF5_LIBRARIES} Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
target_compile_definitions(readreftableTest PRIVATE ${HDF5_DEFINITIONS})
target_compile_features(readreftableTest PRIVATE cxx_thread_local)

add_test(NAME ReadRF COMMAND readreftableTest WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(ReadRF PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(simpleregex regex-test.cpp)
target_link_libraries(simpleregex Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(SimpleRegex simpleregex)
set_tests_properties(SimpleRegex PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(matutils matutils-test.cpp readreftable.cpp)
target_link_libraries(matutils Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME MatutilsTest COMMAND matutils WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(MatutilsTest PROPERTIES ENVIRONMENT "${TestENV}")

# add_executable(ForestClass forestclass-test.cpp ForestOnlineClassification.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
add_executable(ForestClass forestclass-test.cpp ${RANGER_SOURCE_DIR}/ForestClassification.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(ForestClass Threads::Threads Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME ForestClassTest COMMAND ForestClass WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(ForestClassTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(ForestReg forestreg-test.cpp ForestOnlineRegression.cpp readreftable.cpp ${PROJECT_abcranger_SRCS})
target_link_libraries(ForestReg Threads::Threads Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME ForestRegTest COMMAND ForestReg WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(ForestRegTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(LDA lda-test.cpp)
target_link_libraries(LDA ${LAPACK_LIBRARIES} Boost::unit_test_framework Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME LDAtest COMMAND LDA WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(LDAtest PROPERTIES ENVIRONMENT "${TestENV}")
target_compile_features(LDA PRIVATE cxx_std_17)

add_executable(Lapack lapack-test.cpp)
target_link_libraries(Lapack ${LAPACK_LIBRARIES} Boost::unit_test_framework Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME Lapacktest COMMAND Lapack WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(Lapacktest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(Eigen eigen-test.cpp)
target_compile_definitions(Eigen PRIVATE EIGEN_USE_MKL_ALL)
target_link_libraries(Eigen ${LAPACK_LIBRARIES} Boost::unit_test_framework Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME EigenTest COMMAND Eigen WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(EigenTest PROPERTIES ENVIRONMENT "${TestENV}")

add_executable(LDAEigen lda-eigen-test.cpp)
target_compile_definitions(LDAEigen PRIVATE EIGEN_USE_MKL_ALL)
target_link_libraries(LDAEigen ${LAPACK_LIBRARIES} Boost::unit_test_framework Boost::dynamic_linking Boost::diagnostic_definitions)
add_test(NAME LDAEigenTest COMMAND LDAEigen WORKING_DIRECTORY "../../../R/dossier_test1_poolseq_TRIS_model_choice_estim_param_pour_FD")
set_tests_properties(LDAEigenTest PROPERTIES ENVIRONMENT "${TestENV}")
