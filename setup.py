#!/usr/bin/env python
# -*- coding: utf-8 -*-

# partly borrowed from from https://github.com/navdeep-G/setup.py/blob/master/setup.py
import io
import os
from setuptools import setup, find_packages
import pathlib

# Package meta-data.
NAME = 'unifan'
DESCRIPTION = 'Unsupervised cell functional annotation'
URL = "https://github.com/doraadong/UNIFAN"
EMAIL = 'dongshul@andrew.cmu.edu'
AUTHOR = 'Dora Li'
REQUIRES_PYTHON = '>=3.6'
VERSION = '1.0.0'

# What packages are required for this module to be executed?
REQUIRED = ["torch", "numpy>=1.19.2", "pandas>=1.1.5", "scanpy>=1.7.2", "tqdm>=4.61.1",
            "scikit-learn>=0.24.2", "umap-learn>=0.5.1", "matplotlib>=3.3.4", "seaborn>=0.11.0"]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['unifan'],
    scripts=['unifan/main.py'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
