#!/bin/bash
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit-2020.0-088
yum install -y python3-devel
pip3 install cmaketools pybind11 cmake ninja
mkdir build 
cd build
cmake --no-warn-unused-cli -DMAKE_STATIC_EXE:BOOL=TRUE -DUSE_MKL:BOOL=TRUE -DLAPACK_ROOT:STRING=/opt/intel/mkl/lib/intel64 -DLAPACK_LIBRARIES:STRING="-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group;pthread;m;dl" -DBLAS_LIBRARIES:STRING="-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group;pthread;m;dl" -DRFTEST_TOLERANCE:STRING=2e-2 -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja ../
