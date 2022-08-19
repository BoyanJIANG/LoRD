try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from setuptools.command.build_ext import build_ext
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

numpy_include_dir = numpy.get_include()

# pykdtree (kd tree)
pykdtree = Extension(
    'lib.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'lib/utils/libkdtree/pykdtree/kdtree.c',
        'lib/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)


ext_modules = [
    pykdtree,
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmd_class={
        'build_ext': build_ext
    }
)