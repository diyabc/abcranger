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
  add_compile_options(/permissive-)
  add_subdirectory (wingetopt)
  link_directories (${wingetopt_BINARY_DIR})
endif()
                  
add_executable(ranger ${PROJECT_cppranger_SRCS})
if (MSVC)
  target_link_libraries(ranger Threads::Threads wingetopt)
  include_directories(${CMAKE_SOURCE_DIR}/wingetopt/src)
else()
  target_link_libraries(ranger Threads::Threads)
endif()


project(abcranger VERSION 0.2.0 LANGUAGES CXX C)

include(pmm.cmake)

if(NOT USE_MKL)
  set(LAPACK_PKG "openblas")
endif()


if(MSVC AND MAKE_STATIC_EXE)
  set(VCPKG_TARGET_TRIPLET "x64-windows-static")
endif()


pmm(
    # VERBOSE
    # DEBUG
    VCPKG
        REVISION e37cc662ee29852f45e85961124f62d91acb488a
        TRIPLET ${VCPKG_TARGET_TRIPLET}
        REQUIRES range-v3 catch2 cxxopts fmt hdf5[cpp] highfive boost-accumulators boost-regex eigen3-head ${LAPACK_PKG}
        PORTS "${PROJECT_SOURCE_DIR}/eigen3-head"
)

### LAPACK
if(LAPACK_ROOT)
  link_directories(${LAPACK_ROOT})
  include_directories(${LAPACK_ROOT}/../include)
  include_directories(${LAPACK_ROOT}/../../include)
endif()
if(NOT USE_MKL)
  find_package(OpenBLAS CONFIG REQUIRED)
endif()

find_package(LAPACK)

if(MAKE_STATIC_EXE)
  if(MSVC)
    set(CompilerFlags
    CMAKE_CXX_FLAGS
    CMAKE_CXX_FLAGS_DEBUG
    CMAKE_CXX_FLAGS_RELEASE
    CMAKE_C_FLAGS
    CMAKE_C_FLAGS_DEBUG
    CMAKE_C_FLAGS_RELEASE
    )
    foreach(CompilerFlag ${CompilerFlags})
    string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
    set(MAKE_STATIC_FLAG_EXE "")
  else()
    set(MAKE_STATIC_FLAG_EXE "-static")
  endif()
endif()

include(CTest)
enable_testing()

add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING -D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

### C++ Threads (required by ranger)
find_package(Threads REQUIRED)

### Eigen
find_package(Eigen3 CONFIG REQUIRED)
link_libraries(Eigen3::Eigen)
if(USE_MKL)
  add_compile_definitions(PRIVATE EIGEN_USE_MKL_ALL)
else()
  add_compile_definitions(PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
endif()

### Range v3
find_package(range-v3 CONFIG REQUIRED)
link_libraries(meta range-v3)

### Boost::regex with MSVC
if(MSVC)
    find_package(Boost REQUIRED COMPONENTS regex)
endif()

# if (WIN32)
#      set(WINDOWS_EXPORT_ALL_SYMBOLS True)
# endif()

add_subdirectory(src)
include_directories(src)

link_libraries(${LAPACK_LIBRARIES} Threads::Threads)

## CXXOPTS
find_package(cxxopts CONFIG REQUIRED)

## FMT
find_package(fmt CONFIG REQUIRED)

add_subdirectory(test)

add_executable(ModelChoice model_choice.cpp)
add_executable(EstimParam estim_param.cpp)

target_link_libraries(ModelChoice ${MAKE_STATIC_FLAG_EXE} cxxopts::cxxopts abcrangerlib fmt::fmt-header-only)
target_link_libraries(EstimParam ${MAKE_STATIC_FLAG_EXE} cxxopts::cxxopts abcrangerlib fmt::fmt-header-only)



