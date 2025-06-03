# Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
from setuptools import setup


VERSION = "1.52.5"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name='pyqrack',
    version=VERSION,
    packages=['pyqrack', 'pyqrack.qrack_system', 'pyqrack.stats'],
    description="pyqrack - Pure Python vm6502q/qrack Wrapper",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/pyqrack",
    author="Daniel Strano",
    author_email="dan@unitary.fund",
    license="MIT",
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.3",
        "Programming Language :: Python :: 2.4",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="pyqrack qrack simulator quantum gpu",
    install_requires=[],
    setup_requires=[],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
        ],
    },
    include_package_data=True,
    zip_safe=False
)
