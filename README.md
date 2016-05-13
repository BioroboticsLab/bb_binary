# bb_binary

bb_binary contains the capnproto schema of the BeesBooks detections and the
`bbb` tool. `bbb` is a collection of usefull tools for the  BeesBook detection data.

## Commands

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


## Not Implemented Commands

### build-index

```sh
$ bbb build-index --file-size [] --output-dir []  dirname
```

Organise all the csv/bbb files in `dirname` into a recursive
time index directory structure.
The directory of `dirname` is flat, the files are then converted to
BBB and organised as:


```
124 ─┰ 1035 ┰ 12410350015.bbb.lz4
     ┃      ┠ 12410351355.bbb.lz4
     ┃      ┠  ...
     ┃      ┖ 12410358490.bbb.lz4
     ┠ 1240 ┰ 12412400015.bbb.lz4
     ┃      ┠ 12410351355.bbb.lz4
     ┃      ┠  ...
     ┃      ┖ 12410351355.bbb.lz4
     ┃
     ┠ .....
     ┃
     ┖ 8940 ┰ 12489400015.bbb.lz4
            ┠ 12489401355.bbb.lz4
            ┖ 12489401355.bbb.lz4
```

### select

```sh
$ bbb select [-f FORMAT] [-n NUMBER] [-o FILE] TIMESTAMP
```

Must be run in a directory organised with `bbb build-index`.

**Options:**

* **-f, --format**: Output format can be either `json` or `bbb`.
* **-n**: Select at maximal this number of detections.
* **-o**: Write result to this file. If it is not specified the
  output will be written to stdou.
* **TIMESTAMP**: Timestamp from there to start and end the selection. Possible
  Formats:
    - `TIME`: Select detections from this timestamp.
    - `FROM-TO`: Select detections from this timespan.

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

## Use Java Package

There exists a java package to read the `bbb` files in java or
scala.

TODO: Document how to build a .jar of the package.
