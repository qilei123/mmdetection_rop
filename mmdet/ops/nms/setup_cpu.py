from setuptools import setup, Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# extensions
ext_args = dict(
    include_dirs=[numpy_include],
    language='c++',
)

ext_modules = [
    Extension(
        "nms_cpu",
        sources=["src/nms_cpu.cpp"],
        **ext_args
    ),
    Extension(
        "soft_nms_cpu",
        sources=["src/soft_nms_cpu.pyx"],
        **ext_args
    ),
    ]

setup(
    name='nms',
    ext_modules=cythonize(ext_modules),
    # inject our custom trigger
    cmdclass={'build_ext': BuildExtension},
)