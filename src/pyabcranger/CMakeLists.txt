project(pyabcranger)

## Pybind11PYTHO
find_package(Python REQUIRED COMPONENTS Development)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyabcranger ${ABCRANGER_SOURCE_DIR}/ModelChoice.cpp ${ABCRANGER_SOURCE_DIR}/EstimParam.cpp pyabcranger.cpp ${PROJECT_abcranger_SRCS})

target_link_libraries(pyabcranger PRIVATE ${TBBLIB} Python::Python fmt::fmt cxxopts::cxxopts)
target_compile_definitions(pyabcranger PRIVATE PYTHON_OUTPUT=1)

# install pyd files one directory structure UP in the destination as 
# CMAKE_CURRENT_SOURCE_DIR is holding the source files of the MODULE
# which matches the name of CMAKE_CURRENT_SOURCE_DIR
get_filename_component(CURRENT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
if(${SRC_DIR} STREQUAL ${CURRENT_SRC_DIR}) 
    set(DST_DIR ".") # if parent is the base folder
else()
    file(RELATIVE_PATH DST_DIR ${SRC_DIR} ${CURRENT_SRC_DIR})
endif()
install(TARGETS pyabcranger DESTINATION ${DST_DIR} COMPONENT "EXT")