project(pyabcranger)

## Pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyabcranger ${ABCRANGER_SOURCE_DIR}/ModelChoice.cpp ${ABCRANGER_SOURCE_DIR}/EstimParam.cpp pyabcranger.cpp ${PROJECT_abcranger_SRCS})

target_link_libraries(pyabcranger PRIVATE ${TBBLIB} fmt::fmt cxxopts::cxxopts)
target_compile_definitions(pyabcranger PRIVATE PYTHON_OUTPUT=1)
# if (USES_LIBCXX)
#     target_link_options(pyabcranger PRIVATE -nodefaultlibs)
#     target_link_libraries(pyabcranger PRIVATE libc++.a libc++abi.a libgcc_eh.a libgcc.a)
# endif()

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