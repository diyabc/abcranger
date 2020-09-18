project(pybind11_tests LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pybind11_tests pybind11_tests.cpp test_eigen.cpp)
target_link_libraries(pybind11_tests PRIVATE ${TBBLIB})

# pybind11_add_module(test_eigen test_eigen.cpp)
# target_link_libraries(test_eigen PRIVATE ${TBBLIB})