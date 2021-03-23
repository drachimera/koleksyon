from setuptools import setup, find_packages
from distutils.core import setup
from Cython.Build import cythonize

setup(name="mcmc", ext_modules=cythonize("./src/koleksyon/mcmc.pyx"))
