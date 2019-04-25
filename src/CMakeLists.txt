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
${ABCRANGER_SOURCE_DIR}/DataDense.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnline.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnlineClassification.cpp
${ABCRANGER_SOURCE_DIR}/ForestOnlineRegression.cpp
)

add_library(abcrangerlib STATIC ${PROJECT_abcranger_SRCS})

if(MSVC)
    target_link_libraries(abcrangerlib PRIVATE Boost::regex)
endif()

# if (WIN32)
#     set(WINDOWS_EXPORT_ALL_SYMBOLS True)
#     include(GenerateExportHeader)
#     generate_export_header(abcrangerlib)
# endif()
