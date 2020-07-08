cmake_minimum_required(VERSION 3.12.0)
cmake_policy(SET CMP0074 NEW)

project(abcrangerlib VERSION 0.2.0 LANGUAGES CXX)


set(PROJECT_abcranger_SRCS
${RANGER_SOURCE_DIR}/Data.cpp
${RANGER_SOURCE_DIR}/DataFloat.cpp
${RANGER_SOURCE_DIR}/Forest.cpp
${RANGER_SOURCE_DIR}/Tree.cpp
${RANGER_SOURCE_DIR}/TreeClassification.cpp
${RANGER_SOURCE_DIR}/TreeRegression.cpp
${RANGER_SOURCE_DIR}/utility.cpp

${ABCRANGER_SOURCE_DIR}/readreftable.cpp
${ABCRANGER_SOURCE_DIR}/parse_parexpr.cpp
${ABCRANGER_SOURCE_DIR}/forestQuantiles.cpp
${ABCRANGER_SOURCE_DIR}/DataDense.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnline.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnlineClassification.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnlineRegression.cpp
${ABCRANGER_SOURCE_DIR}/tqdm.cpp
)

add_library(abcrangerlib STATIC ${PROJECT_abcranger_SRCS})
target_link_libraries(abcrangerlib fmt::fmt)
if (MSVC)
    add_library(abcrangerlib_so STATIC ${PROJECT_abcranger_SRCS})
else()
    add_library(abcrangerlib_so SHARED ${PROJECT_abcranger_SRCS})
endif()

target_link_libraries(abcrangerlib_so fmt::fmt)

# if(MSVC)
#     target_link_libraries(abcrangerlib PRIVATE Boost::regex)
# endif()

# if (WIN32)
#     set(WINDOWS_EXPORT_ALL_SYMBOLS True)
#     include(GenerateExportHeader)
#     generate_export_header(abcrangerlib)
# endif()
