from cmaketools import setup
import sys
import platform
from os import path, getenv

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

python_version = str(sys.version_info.major) + "." + str(sys.version_info.minor)
print("PYTHON VERSION : " + python_version)

if((len(sys.argv) > 1) and (sys.argv[1] == "sdist")):
    configure_opts = []
elif sys.platform == "linux":
    configure_opts = ["-DPython_EXECUTABLE="+sys.executable,"-DPYABCRANGER=TRUE","-DUSE_MKL:BOOL=TRUE","-DMAKE_STATIC_EXE:BOOL=TRUE","-DLAPACK_ROOT:STRING=/opt/intel/oneapi/mkl/latest/lib","-DLAPACK_LIBRARIES:STRING=-Wl,--start-group /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_tbb_thread.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a -Wl,--end-group;pthread;m;dl","-DBLAS_LIBRARIES:STRING=-Wl,--start-group /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_tbb_thread.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a -Wl,--end-group;pthread;m;dl"]
elif sys.platform == "darwin":
    if getenv("OSARCH") == "arm":
        configure_opts = ["-DPython_EXECUTABLE="+sys.executable,"-DVCPKG_TARGET_TRIPLET:STRING=arm64-osx","-DCMAKE_OSX_ARCHITECTURES:STRING=arm64","-DPYABCRANGER=TRUE","-DUSE_MKL:BOOL=FALSE","-DCMAKE_BUILD_TYPE:STRING=Release"]
    else:
        configure_opts = ["-DPython_EXECUTABLE="+sys.executable,"-DVCPKG_TARGET_TRIPLET:STRING=x64-osx","-DCMAKE_OSX_ARCHITECTURES:STRING=x86_64","-DPYABCRANGER=TRUE","-DUSE_MKL:BOOL=FALSE","-DCMAKE_BUILD_TYPE:STRING=Release"]
elif sys.platform == "win32":
    configure_opts = ["-DPython_FIND_REGISTRY=NEVER","-DPYABCRANGER=TRUE","-DUSE_MKL:BOOL=FALSE","-DMAKE_STATIC_EXE:BOOL=TRUE","-DVCPKG_TARGET_TRIPLET:STRING=x64-windows-static"]
else:
    exit(1)

setup(
    name="pyabcranger",
    version="0.0.72",
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
    config="Release",
    has_package_data=False,
    packages=["pyabcranger"],
    configure_opts=configure_opts,
    build_opts=["--target","pyabcranger"]
)