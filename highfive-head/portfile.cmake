include(vcpkg_common_functions)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO BlueBrain/HighFive
    REF 036cfa12381ef16c79ca8ebec20f7d4b7c5afcc2
    SHA512 2216b366f11c1024d114b43591b4eefb9384a1b38d97b9f0815c247890a19ba6eea513cf068af5655c55e686283ccacd4d7256a8450aa31e3641543f6015dc89
    HEAD_REF master
)

if(${VCPKG_LIBRARY_LINKAGE} MATCHES "static")
    set(HDF5_USE_STATIC_LIBRARIES ON)
endif()

vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA
    OPTIONS
        -DHIGHFIVE_UNIT_TESTS=OFF
        -DHIGHFIVE_EXAMPLES=OFF
        -DUSE_BOOST=OFF
        -DHIGH_FIVE_DOCUMENTATION=OFF
        -DUSE_EIGEN=ON
        -DHDF5_USE_STATIC_LIBRARIES=${HDF5_USE_STATIC_LIBRARIES}
)

vcpkg_install_cmake()

vcpkg_fixup_cmake_targets(CONFIG_PATH share/HighFive/CMake)

file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug)
if(NOT VCPKG_CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
  file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/share/HighFive)
endif()

# Handle copyright
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/highfive-head RENAME copyright)
