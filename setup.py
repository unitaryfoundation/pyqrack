# Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
from setuptools import setup


requirements = []

VERSION = "1.1.1"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name='pyqrack',
    version=VERSION,
    packages=['pyqrack', 'pyqrack.qrack_system', 'pyqrack.util'],
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    keywords="pyqrack qrack simulator quantum gpu",
    install_requires=requirements,
    setup_requires=[],
    include_package_data=True,
    zip_safe=False,
)
