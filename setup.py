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
    build_opts=["--target","pyabcranger"]
)