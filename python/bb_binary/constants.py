# -*- coding: utf-8 -*-
import os
import capnp

capnp.remove_import_hook()
_dirname = os.path.dirname(os.path.realpath(__file__))

bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))
Frame = bbb.Frame
FrameContainer = bbb.FrameContainer
DataSource = bbb.DataSource
DetectionCVP = bbb.DetectionCVP
DetectionDP = bbb.DetectionDP
DetectionTruth = bbb.DetectionTruth


CAM_IDX = 0
BEGIN_IDX = 1
TIME_IDX = 1
END_IDX = 2
