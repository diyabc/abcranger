project(rabcranger)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
message("Modules : ${CMAKE_MODULE_PATH}")

find_package(R REQUIRED)
message("R libraries: ${R_LIBRARIES}")
message("R include dirs: ${R_INCLUDE_DIR} ${Rcpp_INCLUDE_DIR}")

add_library(rabcranger SHARED rabcranger.cpp)
include_directories(rabcranger ${R_INCLUDE_DIR} ${Rcpp_INCLUDE_DIR})
target_link_libraries(rabcranger ${R_LIBRARIES})

install(TARGETS rabcranger DESTINATION lib)