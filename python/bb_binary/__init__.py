# -*- coding: utf-8 -*-
import bb_binary.constants as constants
import bb_binary.converting as converting
import bb_binary.repository as repository
import bb_binary.parsing as parsing

# TODO: Add warning about windows symlinks

Frame = constants.Frame
FrameContainer = constants.FrameContainer
DataSource = constants.DataSource
DetectionCVP = constants.DetectionCVP
DetectionDP = constants.DetectionDP
DetectionTruth = constants.DetectionTruth


convert_frame_to_numpy = converting.convert_frame_to_numpy
build_frame_container = converting.build_frame_container
build_frame_container_from_df = converting.build_frame_container_from_df

load_frame_container = repository.load_frame_container
Repository = repository.Repository

dt_to_str = parsing.dt_to_str
get_fname = parsing.get_fname
get_video_fname = parsing.get_video_fname
get_timezone = parsing.get_timezone
int_id_to_binary = parsing.int_id_to_binary
parse_cam_id = parsing.parse_cam_id
parse_fname = parsing.parse_fname
parse_image_fname = parsing.parse_image_fname
parse_image_fname_iso = parsing.parse_image_fname
parse_image_fname_beesbook = parsing.parse_image_fname_beesbook
parse_video_fname = parsing.parse_video_fname
to_datetime = parsing.to_datetime
to_timestamp = parsing.to_timestamp
