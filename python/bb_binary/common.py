# -*- coding: utf-8 -*-
"""
These objects are providing access to our :doc:`schema` in
`Python's Cap'n Proto implementation <http://jparyani.github.io/pycapnp/>`_.
"""
import os
import capnp

capnp.remove_import_hook()
_dirname = os.path.dirname(os.path.realpath(__file__))

bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))

FrameContainer = bbb.FrameContainer
"""
FrameContainer are basically the root of our data container.
They represent a video that in beesbook context usually consist of 5 minutes.
Each FrameContainer only has the frames of **one** camera.
"""

Frame = bbb.Frame
"""
A Frame holds all the information about a single image in a video.
"""

DataSource = bbb.DataSource
"""
This is a part of a :obj:`FrameContainer` and references the original video file.
"""

DetectionDP = bbb.DetectionDP
"""
This is the format for a pipeline detection with decoded tag.
"""

DetectionTruth = bbb.DetectionTruth
"""
This is the format for manually generated truth data that might be generated via the
`Editor GUI <https://github.com/BioroboticsLab/bb_analysis/tree/master/system2/editor-gui>`_.
"""

DetectionBee = bbb.DetectionBee
"""
This is the format for a pipeline detection without decoded tag.
"""

CAM_IDX = 0
BEGIN_IDX = 1
TIME_IDX = 1
END_IDX = 2
