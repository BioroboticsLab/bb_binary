# bb_binary

bb_binary contains the capnproto schema of the BeesBooks detections and the
`bbb` tool. `bbb` is a collection of usefull tools for the  BeesBook detection data.

## Python Interface

To install the python interface simply run:

```
$ pip install git+https://github.com/BioroboticsLab/bb_binary.git
```

## Java  Inteface

There is also a java interface under `java`. But currently there is no nice way
to include it into other projects.

## C++ Inteface

Use  [CPM](https://github.com/iauns/cpm) to include the C++ inteface.

In your `CMakeLists.txt`:

```cmake
CPM_AddModule(
    "bb_binary"
    GIT_REPOSITORY "https://github.com/BioroboticsLab/bb_binary.git"
    GIT_TAG "master"
)
```

## C++ Commands Tool

## convert

```
bbb convert -o frames.bbb *.csv
```

Converts CSV files to bbb (BeesBook Binary) files.

```
bbb convert -o frames.bbb.lz4 *.csv
```

Also apply [LZ4](http://cyan4973.github.io/lz4/) compression to the
bbb file. This will use the [lz4frame format](https://cyan4973.github.io/lz4/lz4_Frame_format.html).


#### Examples

```sh
$ bbb select -f json 1240000-1240099
```

Selects all detections from timestamp 1240000 to 1240099 and prints
them as json to the standard output.

## Build and Install

You need to have Boost and [Capnproto](https://capnproto.org/) installed.

```sh
$ mkdir build
$ cmake ..
$ make -j4
$ make install
```
