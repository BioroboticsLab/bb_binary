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

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 82-98
"""

Frame = bbb.Frame
"""
A Frame holds all the information about a single image in a video.

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 62-74
"""

DataSource = bbb.DataSource
"""
This is a part of a :obj:`FrameContainer` and references the original video file.

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 76-80
"""

HiveMappingData = bbb.HiveMappingData
# TODO(gitmirgut): Add docs

HiveMappedDetection = bbb.HiveMappedDetection
# TODO(gitmirgut): Add docs

DetectionCVP = bbb.DetectionCVP
"""
This is the format of a detection in the old Computer Vision Pipeline format.
It got replaced with :obj:`DetectionDP` in the Summer 2016.

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 8-24
"""

DetectionDP = bbb.DetectionDP
"""
This is the new format for a detection that replaced :obj:`DetectionCVP`.

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 26-42
"""

DetectionTruth = bbb.DetectionTruth
"""
This is the format for manually generated truth data that might be generated via the
`Editor GUI <https://github.com/BioroboticsLab/bb_analysis/tree/master/system2/editor-gui>`_.

.. literalinclude:: ../../bb_binary/bb_binary_schema.capnp
    :lines: 44-59
"""

CAM_IDX = 0
BEGIN_IDX = 1
TIME_IDX = 1
END_IDX = 2
