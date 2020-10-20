#!/usr/bin/env python

from distutils.core import setup

def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]


setup(
    name='bb_binary',
    version='2.0',
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
