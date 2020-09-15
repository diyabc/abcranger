from setuptools import setup
from cmake_setup import *

setup(name='pyabcranger',
      description='ABC random forests for model choice and parameter estimation, python wrapper',
      version='0.0.0.dev0',
      ext_modules=[CMakeExtension('abcrangerlib_so')],
      cmdclass={'build_ext': CMakeBuildExt}
      )