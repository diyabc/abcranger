include(vcpkg_common_functions)

string(LENGTH "${CURRENT_BUILDTREES_DIR}" BUILDTREES_PATH_LENGTH)
if(BUILDTREES_PATH_LENGTH GREATER 37 AND CMAKE_HOST_WIN32)
    message(WARNING "eigen3's buildsystem uses very long paths and may fail on your system.\n"
        "We recommend moving vcpkg to a short path such as 'C:\\src\\vcpkg' or using the subst command."
    )
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO eigenteam/eigen-git-mirror
    REF e6754c8294b4b981a0786a8e623c490c7ab28221
    SHA512 118a6c2263638458130a73b85f0c6e0615099365e4170b8b90abeea45d1989443a01d20980a5be2f1aa0af4c984f6a6882326c74526271baaf114c1335b60792
    HEAD_REF master
)

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DBUILD_TESTING=OFF
    OPTIONS_RELEASE
        -DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/share/eigen3-head
    OPTIONS_DEBUG
        -DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/debug/share/eigen3-head
)

vcpkg_install_cmake()

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)

file(READ "${CURRENT_PACKAGES_DIR}/share/eigen3-head/Eigen3Targets.cmake" EIGEN_TARGETS)
string(REPLACE "set(_IMPORT_PREFIX " "get_filename_component(_IMPORT_PREFIX \"\${CMAKE_CURRENT_LIST_DIR}/../..\" ABSOLUTE) #" EIGEN_TARGETS "${EIGEN_TARGETS}")
file(WRITE "${CURRENT_PACKAGES_DIR}/share/eigen3-head/Eigen3Targets.cmake" "${EIGEN_TARGETS}")

file(GLOB INCLUDES ${CURRENT_PACKAGES_DIR}/include/eigen3-head/*)
# Copy the eigen header files to conventional location for user-wide MSBuild integration
file(COPY ${INCLUDES} DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Put the licence file where vcpkg expects it
file(COPY ${SOURCE_PATH}/COPYING.README DESTINATION ${CURRENT_PACKAGES_DIR}/share/eigen3-head)
file(RENAME ${CURRENT_PACKAGES_DIR}/share/eigen3-head/COPYING.README ${CURRENT_PACKAGES_DIR}/share/eigen3-head/copyright)
