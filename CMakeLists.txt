cmake_minimum_required(VERSION 3.26)
cmake_policy(SET CMP0074 NEW)

set(CMAKE_CXX_STANDARD 17)
set(RANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/src)
set(CPPRANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ranger/cpp_version/src)
set(ABCRANGER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)


project(ranger-standalone LANGUAGES CXX)

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

message("Compiler ID : ${CMAKE_CXX_COMPILER_ID}")
IF(MSVC)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "[Cc]lang")
    add_compile_options(/permissive-)
  else()
    add_compile_options(/permissive- /Zc:preprocessor /wd5105)
  endif()
  add_subdirectory (wingetopt)
  link_directories (${wingetopt_BINARY_DIR})
endif()
                  
add_executable(ranger ${PROJECT_cppranger_SRCS})
if (MSVC)
  target_link_libraries(ranger Threads::Threads wingetopt)
  include_directories(${CMAKE_SOURCE_DIR}/wingetopt/src)
  set(BOOSTREPKG boost-regex)
else()
  target_link_libraries(ranger Threads::Threads)
endif()

project(abcranger VERSION 1.0.5 LANGUAGES CXX C)

include(pmm.cmake)

if(MSVC AND MAKE_STATIC_EXE)
  set(VCPKG_TARGET_TRIPLET "x64-windows-static")
endif()

if(USE_MKL)
  set(TBBPKG tbb)
endif()

### LAPACK
if(LAPACK_ROOT)
  find_package(LAPACK)
  link_directories(${LAPACK_ROOT})
  link_libraries(${LAPACK_LIBRARIES})
endif()

if(TEST)
  set(TESTDEPS hdf5[cpp] highfive[core])
endif()

if(PYABCRANGER)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  set(PYBIND11PKG pybind11)
endif()

pmm(
    VCPKG
        REVISION 2024.04.26
        TRIPLET ${VCPKG_TARGET_TRIPLET}
        REQUIRES range-v3 catch2 cxxopts fmt  eigen3 ${PYBIND11PKG} ${TBBPKG} ${BOOSTREPKG} ${TESTDEPS}
)

if(MAKE_STATIC_EXE)
  message("Static generation ACTIVATED")
  if(MSVC)
      # add_compile_options(
      #   $<$<CONFIG:>:/MT> #---------|
      #   $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
      #   $<$<CONFIG:Release>:/MT> #--|
      # )
  else()
    set(MAKE_STATIC_FLAG_EXE "-static")

    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        message("USING Clang, checking libc++..")
        include(CheckCXXSourceCompiles)
        check_cxx_source_compiles("#include <iostream>
                                  int main() { return _LIBCPP_VERSION; }" USES_LIBCXX)
      if(USES_LIBCXX)
            message("USING libc++")
      endif()
    endif()

  endif()
endif()

include(CTest)
enable_testing()

if(MSVC)
  find_package(Boost REQUIRED COMPONENTS regex)
  set(BOOSTLIBS Boost::headers Boost::regex)
endif()

add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING 
                -D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING 
                -D_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING
                -D_CRT_SECURE_NO_WARNINGS
                -DTBB_ENABLE_IPO=OFF
                -DNOMINMAX)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

### C++ Threads (required by ranger)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

### Eigen
find_package(Eigen3 CONFIG REQUIRED)
message("Eigen3 found: ${Eigen3_DIR}")
link_libraries(Eigen3::Eigen)
if(USE_MKL)
  add_compile_definitions(PRIVATE EIGEN_USE_MKL_ALL)
  include_directories(${LAPACK_ROOT}/../include)
  include_directories(${LAPACK_ROOT}/../../include)
endif()

### Range v3
find_package(range-v3 CONFIG REQUIRED)
link_libraries(range-v3)

## FMT
find_package(fmt CONFIG REQUIRED)

## CXXOPTS
find_package(cxxopts CONFIG REQUIRED)

## TBBLIB
if(MAKE_STATIC_EXE AND USE_MKL)
  find_package(TBB CONFIG REQUIRED)
  set(TBBLIB TBB::tbb TBB::tbbmalloc)
endif()

include_directories(src)

add_subdirectory(src)

add_executable(abcranger ${ABCRANGER_SOURCE_DIR}/ModelChoice.cpp ${ABCRANGER_SOURCE_DIR}/EstimParam.cpp abcranger.cpp)
set_property(TARGET abcranger PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(abcranger ${MAKE_STATIC_FLAG_EXE} ${TBBLIB} ${BOOSTLIBS} cxxopts::cxxopts abcrangerlib)

if(USE_LIBCXX)
  target_link_libraries(abcranger PRIVATE c++ c++abi)
endif()

if(TEST)
  add_subdirectory(test)
#  add_subdirectory(testpy)
else()
  message("Skipping tests")
endif()

