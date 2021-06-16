# Copyright 2021 MIT Probabilistic Computing Project
# See LICENSE.txt

import os
import re
from setuptools import setup

# Specify the requirements.
requirements = {
    'src' : [
        'scipy==1.6.*',
    ],
    'tests'     : [
        'pytest==5.2.*'
    ],
    'examples' : [
        'matplotlib==3.4.*',
        'numpy==1.20.*',
    ]
}
requirements['all'] = [r for v in requirements.values() for r in v]

# Determine the version (hardcoded).
dirname = os.path.dirname(os.path.realpath(__file__))
vre = re.compile('__version__ = \'(.*?)\'')
m = open(os.path.join(dirname, 'src', '__init__.py')).read()
__version__ = vre.findall(m)[0]

setup(
    name='hirm',
    version=__version__,
    description='Hierarchical Infinite Relational Model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
    ],
    packages=[
        'hirm',
        'hirm.tests',
    ],
    package_dir={
        'hirm'           : 'src',
        'hirm.tests'     : 'tests',
    },
    install_requires=requirements['all'],
    extras_require=requirements,
    python_requires='>=3.6',
)
