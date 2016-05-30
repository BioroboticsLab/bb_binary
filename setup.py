#!/usr/bin/env python

from distutils.core import setup


setup(name='bb_binary',
      version='0.1',
      description='BeesBook Binary',
      author='Leon Sixt',
      author_email='mail@leon-sixt.de',
      url='https://github.com/BioroboticsLab/bb_binary/',
      packages=['bb_binary'],
      package_dir={'': 'python'},
      package_data={'bb_binary': ['*.capnp']}
)
