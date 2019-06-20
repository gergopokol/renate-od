
from setuptools import setup, find_packages, Extension
import sys
import numpy
import os
import os.path as path
import multiprocessing


use_cython = True
force = False
profile = False

if "--skip-cython" in sys.argv:
    use_cython = False
    del sys.argv[sys.argv.index("--skip-cython")]

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]


source_paths = ['renate']
compilation_includes = [".", numpy.get_include()]
compilation_args = []
cython_directives = {
    'language_level': 3
}
setup_path = path.dirname(path.abspath(__file__))

if use_cython:

    from Cython.Build import cythonize

    # build .pyx extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".pyx":
                    pyx_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(pyx_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [pyx_file], include_dirs=compilation_includes,
                                                extra_compile_args=compilation_args),)

    if profile:
        cython_directives["profile"] = True

    # generate .c files from .pyx
    extensions = cythonize(extensions, nthreads=multiprocessing.cpu_count(), force=force, compiler_directives=cython_directives)

else:

    # build .c extension list
    extensions = []
    for package in source_paths:
        for root, dirs, files in os.walk(path.join(setup_path, package)):
            for file in files:
                if path.splitext(file)[1] == ".c":
                    c_file = path.relpath(path.join(root, file), setup_path)
                    module = path.splitext(c_file)[0].replace("/", ".")
                    extensions.append(Extension(module, [c_file], include_dirs=compilation_includes,
                                                extra_compile_args=compilation_args),)


setup(
    name="renate-od",
    version=1.1.0,
    license="LGPL-3.0",
    description='RENATE Open Diagnostics',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'lxml', 'h5py'],
    packages=['renate'],
    zip_safe=False,
    ext_modules=extensions
)
