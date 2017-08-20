# -*- coding: utf-8 -*-
"""The Python interface to *bb_binary*.

For more information see <https://github.com/BioroboticsLab/bb_binary>.
"""
from bb_binary.common import Frame, FrameContainer, DataSource, DetectionCVP, DetectionDP,\
    DetectionTruth
from bb_binary.converting import convert_frame_to_numpy, build_frame_container, \
    build_frame_container_from_df
from bb_binary.parsing import dt_to_str, get_fname, get_video_fname, get_timezone, \
    int_id_to_binary, binary_id_to_int, parse_cam_id, parse_fname, parse_image_fname, \
    parse_image_fname_beesbook, parse_video_fname, to_datetime, to_timestamp
from bb_binary.repository import load_frame_container, Repository

# TODO: Add warning about windows symlinks

__all__ = ['Frame', 'FrameContainer', 'DataSource', 'DetectionCVP', 'DetectionDP', 'DetectionTruth',
           'convert_frame_to_numpy', 'build_frame_container', 'build_frame_container_from_df',
           'load_frame_container', 'Repository',
           'dt_to_str', 'get_fname', 'get_video_fname', 'get_timezone',
           'int_id_to_binary', 'binary_id_to_int',
           'parse_cam_id', 'parse_fname', 'parse_image_fname', 'parse_image_fname_beesbook',
           'parse_video_fname', 'to_datetime', 'to_timestamp']
