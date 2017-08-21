#!/usr/bin/env python

from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='bb_binary',
    version='0.2',
    description='BeesBook Binary',
    author='Leon Sixt',
    author_email='mail@leon-sixt.de',
    url='https://github.com/BioroboticsLab/bb_binary/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['bb_binary'],
    package_dir={'': 'python'},
    package_data={'bb_binary': ['*.capnp']},
    entry_points={
        'console_scripts': [
            'bb_gt_to_hdf5 = bb_binary.scripts.gt_to_hdf5:run',
        ]
    }
)
