cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
set(RANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/src)
set(CPPRANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/cpp_version/src)
set(ABCRANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

project(ranger-standalone VERSION 0.1.0 LANGUAGES CXX)

set(PROJECT_cppranger_SRCS
${CPPRANGER_SOURCE_DIR}/main.cpp
${CPPRANGER_SOURCE_DIR}/utility/ArgumentHandler.cpp
${RANGER_SOURCE_DIR}/Forest.cpp
${RANGER_SOURCE_DIR}/ForestClassification.cpp
${RANGER_SOURCE_DIR}/ForestProbability.cpp
${RANGER_SOURCE_DIR}/ForestRegression.cpp
${RANGER_SOURCE_DIR}/ForestSurvival.cpp
${RANGER_SOURCE_DIR}/Tree.cpp
${RANGER_SOURCE_DIR}/TreeClassification.cpp
${RANGER_SOURCE_DIR}/TreeProbability.cpp
${RANGER_SOURCE_DIR}/TreeRegression.cpp
${RANGER_SOURCE_DIR}/TreeSurvival.cpp
${RANGER_SOURCE_DIR}/Data.cpp
${RANGER_SOURCE_DIR}/DataChar.cpp
${RANGER_SOURCE_DIR}/DataFloat.cpp
${RANGER_SOURCE_DIR}/utility.cpp
)

include_directories("${CPPRANGER_SOURCE_DIR}"
                    "${CPPRANGER_SOURCE_DIR}/utility"
                    "${RANGER_SOURCE_DIR}")

IF(MSVC)
  add_subdirectory (wingetopt)
  link_directories (${wingetopt_BINARY_DIR})
ENDIF()
                  
add_executable(ranger ${PROJECT_cppranger_SRCS})
if (MSVC)
  target_link_libraries(ranger Threads::Threads wingetopt)
  include_directories(${CMAKE_SOURCE_DIR}/wingetopt/src)
else()
  target_link_libraries(ranger Threads::Threads)
endif()


project(abcranger VERSION 0.2.0 LANGUAGES CXX)

add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING -D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

### C++ Threads (required by ranger)
find_package(Threads REQUIRED)

### LAPACK
link_directories(${LAPACK_ROOT})
find_package(LAPACK)
include_directories(${LAPACK_ROOT}/../../include)

### Boost
find_package(Boost REQUIRED COMPONENTS filesystem regex)
include_directories(Boost::boost)

### Eigen
include_directories(${CMAKE_SOURCE_DIR}/eigen)


# if (WIN32)
#     set(WINDOWS_EXPORT_ALL_SYMBOLS True)
# endif()


add_subdirectory(src)
include_directories(src)

link_libraries(abcrangerlib ${LAPACK_LIBRARIES} Threads::Threads  Boost::regex Boost::dynamic_linking Boost::diagnostic_definitions)

add_compile_definitions(PRIVATE EIGEN_USE_MKL_ALL)

add_executable(ModelChoice model_choice.cpp)
add_executable(EstimParam estim_param.cpp)

add_subdirectory(test)
