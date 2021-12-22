from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import antiCPy


long_description = 'The ``antiCPy`` package provides tools to monitor destabilization because of varying control parameters or the influence of noise. Based on early warning measures it provides an extrapolation tool to estimate the time horizon in which a critical transition will probably occur.'

setup(
    name='antiCPy',
    version=antiCPy.__version__,
    url='https://github.com/MartinHessler/antiCPy',
    license='GPL',
    author='Martin Heßler',
    author_email='m_hess23@wwu.de',
    install_requires=['numpy', 'matplotlib', 'scipy', 'numba', 'emcee'],
    scripts=[],
    packages=find_packages(),
    description='A package that provides tools to estimate resilience and noise level of a system as well as extrapolate possible transition times.',
    long_description=long_description,
    platforms='any',
    keywords="time series analysis, critical transitions, leading indicators",
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics'
        ]
    )
