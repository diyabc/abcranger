from cmaketools import setup
import os

setup(
    name="pyabcranger",
    version="0.0.24",
    author="François-David Collin",
    author_email="fradav@gmail.com",
    description="ABC random forests for model choice and parameter estimation, python wrapper",
    url="https://github.com/diyabc/abcranger",
    license="MIT License",
    src_dir="src",
    ext_module_hint=r"pybind11_add_module",
    generator="Ninja",
    has_package_data=False,
    packages=["pyabcranger"],
    configure_opts=["""--no-warn-unused-cli -DMAKE_STATIC_EXE:BOOL=TRUE -DNO_TEST:BOOL=TRUE -DUSE_MKL:BOOL=TRUE -DLAPACK_ROOT:STRING=/opt/intel/mkl/lib/intel64 "-DLAPACK_LIBRARIES:STRING=-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group\;pthread\;m\;dl" "-DBLAS_LIBRARIES:STRING=-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group\;pthread\;m\;dl" -DRFTEST_TOLERANCE:STRING=2e-2 -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja ../"""],
    build_opts=["--target","pyabcranger"]
)