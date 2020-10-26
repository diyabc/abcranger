from cmaketools import setup
import sys
from os import path, getenv
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


build_dir="build"

configure_opts = ["""--no-warn-unused-cli -DMAKE_STATIC_EXE:BOOL=TRUE -DUSE_MKL:BOOL=TRUE -DLAPACK_ROOT:STRING=/opt/intel/mkl/lib/intel64 "-DLAPACK_LIBRARIES:STRING=-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group\;pthread\;m\;dl" "-DBLAS_LIBRARIES:STRING=-Wl,--start-group /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a /opt/intel/mkl/lib/intel64/libmkl_tbb_thread.a /opt/intel/mkl/lib/intel64/libmkl_core.a -Wl,--end-group\;pthread\;m\;dl" -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja ../"""]
if((len(sys.argv) > 1) and (sys.argv[1] == "sdist")):
        configure_opts = ["""--no-warn-unused-cli -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja ../"""]
elif(getenv("CMAKEARGSFILE") and path.isfile(getenv("CMAKEARGSFILE"))):
    with open(path.join(this_directory, getenv("CMAKEARGSFILE")), encoding='utf-8') as f:
        configure_opts = f.read()
        build_dir = getenv("BUILDDIR")

setup(
    name="pyabcranger",
    version="0.0.37",
    author="François-David Collin",
    author_email="fradav@gmail.com",
    description="ABC random forests for model choice and parameter estimation, python wrapper",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/diyabc/abcranger",
    license="MIT License",
    src_dir="src",
    ext_module_hint=r"pybind11_add_module",
    generator="Ninja",
    has_package_data=False,
    packages=["pyabcranger"],
    configure_opts=configure_opts,
    build_opts=["--build",build_dir,"--target","pyabcranger","--config","Release"]
)