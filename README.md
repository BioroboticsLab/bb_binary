# bb_binary

[![Build Status](https://secure.travis-ci.org/BioroboticsLab/bb_binary.svg?branch=master)](http://travis-ci.org/BioroboticsLab/bb_binary?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/BioroboticsLab/bb_binary/badge.svg?branch=master)](https://coveralls.io/github/BioroboticsLab/bb_binary?branch=master)
[![Documentation Status](https://readthedocs.org/projects/bb-binary/badge/?version=latest)](http://bb-binary.readthedocs.io/en/latest/?badge=latest)

bb_binary contains the capnproto schema of the BeesBooks detections and the
`bbb` tool. `bbb` is a collection of usefull tools for the  BeesBook detection data.

## Python Interface

To install the python interface simply run:

```
$ pip install git+https://github.com/BioroboticsLab/bb_binary.git@0.1
```

## Java  Interface

There is also a java interface under `java`. But currently there is no nice way
to include it into other projects.


## Copying the Data
It is advisable to use rsync instead of scp when copying the data (or a subset) from the servers. Reason for this is that scp copies links are real files while rsync, with correct options, does not.

**Wrong**:
```
scp -r server_ip_or_name:/path/to/bb_binary/data .
```

**Right**:
```
rsync -av server_ip_or_name:/path/to/bb_binary/data .
```
