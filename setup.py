from cmaketools import setup
import os

os.environ["VCPKG_KEEP_ENV_VARS"] = "CC;CXX;CXXFLAGS"
os.environ["VCPKG_FORCE_SYSTEM_BINARIES"] = "1"

setup(
    name="pyabcranger",
    version="0.0.18",
    author="François-David Collin",
    author_email="fradav@gmail.com",
    description="ABC random forests for model choice and parameter estimation, python wrapper",
    url="https://github.com/diyabc/abcranger",
    license="MIT License",
    src_dir="src",
    ext_module_hint=r"pybind11_add_module",
    generator="Ninja",
    configure_opts=["""-DCMAKE_BUILD_TYPE:STRING=Release"""],
    has_package_data=False,
    packages=["pyabcranger"],
    build_opts=["--target","pyabcranger"]
)