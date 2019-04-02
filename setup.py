
from setuptools import setup


setup(
    name="renate",
    version=1.0,
    license="LGPL-3.0",
    description='RENATE Open Diagnostics',
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
    packages=['crm_solver'],
)
