cmake_minimum_required(VERSION 3.0.0)
project(abcranger VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
enable_testing()

# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS unit_test_framework filesystem regex)

if (WIN32)
    set(TestENV "PATH=${Boost_LIBRARY_DIRS}\\;$ENV{PATH}")
endif()

include_directories(Boost::boost)

add_executable(abcranger main.cpp readreftable.cpp)
target_link_libraries(abcranger Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)

add_executable(readreftable readreftable-test.cpp readreftable.cpp)
target_link_libraries(readreftable Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
# target_link_libraries(arraytest Boost::unit_test_framework)
add_test(ReadRF readreftable)
set_tests_properties(ReadRF PROPERTIES ENVIRONMENT "${TestENV}")


add_executable(simpleregex regex-test.cpp)
target_link_libraries(simpleregex Boost::unit_test_framework Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)
# target_link_libraries(arraytest Boost::unit_test_framework)
add_test(SimpleRegex simpleregex)
set_tests_properties(SimpleRegex PROPERTIES ENVIRONMENT "${TestENV}")
