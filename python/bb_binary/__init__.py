
import capnp
import os
import numpy as np
import json
import math
import datetime
from pytz import timezone

capnp.remove_import_hook()
_dirname = os.path.dirname(os.path.realpath(__file__))

bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))
Frame = bbb.Frame
FrameContainer = bbb.FrameContainer
DataSource = bbb.DataSource
Cam = bbb.Cam
DetectionCVP = bbb.DetectionCVP
DetectionDP = bbb.DetectionDP

_TIMEZONE = timezone('Europe/Berlin')


def get_timezone():
    return _TIMEZONE


def parse_image_fname_readable(fname):
    basename = os.path.basename(fname)
    name = basename.split('.')[0]
    _, camIdxStr, datetimeStr, usStr = name.split('_')

    camIdx = int(camIdxStr)
    year = int(datetimeStr[:4])
    month = int(datetimeStr[4:6])
    day = int(datetimeStr[6:8])

    hour = int(datetimeStr[8:10])
    minute = int(datetimeStr[10:12])
    second = int(datetimeStr[12:14])
    us = int(usStr)

    dt = datetime.datetime(year, month, day, hour, minute, second, us)
    ts = get_timezone().localize(dt).timestamp()
    return camIdx, ts


def parse_image_fname_ts(fname):
    basename = os.path.basename(fname)
    name = basename.split('.')[0]
    _, camIdxStr, ts_str, micros_str = name.split('_')
    ts = float("{}.{}".format(ts_str, micros_str))
    return int(camIdxStr), ts


def parse_image_fname(fname, format='timestamp'):
    assert format in ['readable', 'timestamp']
    if format == 'readable':
        return parse_image_fname_readable(fname)
    else:
        return parse_image_fname_ts(fname)


def parse_video_fname(fname, format='timestamp'):
    begin_name, end_name = fname.split('_TO_')
    (camIdx, begin) = parse_image_fname(begin_name, format)
    (_, end) = parse_image_fname(end_name, format)
    return camIdx, begin, end


def parse_fname(fname):
    fname = os.path.basename(fname)
    try:
        return parse_video_fname(fname)
    except Exception as e:
        camIdx, dtime = parse_image_fname(fname)
        return camIdx, dtime, dtime


def get_fname(camIdx, ts):
    assert type(ts) in (int, float)
    timestamp = str(ts).replace('.', '_')
    if '_' not in timestamp:
        timestamp += '_0'
    return ("Cam_{cam}_{ts}").format(
                cam=camIdx,
                ts=timestamp,
            )


def get_video_fname(camIdx, begin, end):
    return get_fname(camIdx, begin) + "_TO_" + get_fname(camIdx, end)


def get_cam_id(frame_container):
    return frame_container.dataSources[0].cam.camId


def convert_detections_to_numpy(frame):
    """
    Returns the detections as a numpy array from the frame.
    """

    union_type = frame.detectionsUnion.which()
    assert union_type == 'detectionsDP', \
        "Currently only the new pipeline format is supported."
    detections = frame.detectionsUnion.detectionsDP

    nb_bits = len(detections[0].decodedId)
    shape = (len(detections), nb_parameters(nb_bits))
    arr = np.zeros(shape, dtype=np.float32)
    for i, detection in enumerate(detections):
        arr[i, 0] = detection.tagIdx
        arr[i, 1] = detection.xpos
        arr[i, 2] = detection.ypos
        arr[i, 3] = detection.xposHive
        arr[i, 4] = detection.yposHive
        arr[i, 5] = detection.zRotation
        arr[i, 6] = detection.yRotation
        arr[i, 7] = detection.xRotation
        arr[i, 8] = detection.radius
        arr[i, 9:] = np.array(list(detection.decodedId)) / 255.

    return arr


def nb_parameters(nb_bits):
    """Returns the number of parameter of the detections."""
    return _detection_dp_fields_before_ids + nb_bits


_detection_dp_fields_before_ids = 9


def build_frame(
        frame,
        timestamp,
        detections,
        data_source=0,
        detection_format='deeppipeline'
):
    """
    Builds a frame from a numpy array.
    Usage (not tested):
    ```
    frames = [(timestamp, pipeline(image_to_timestamp))]
    nb_frames = len(frames)
    fc = FrameContainer.new_message()
    frames_builder = fc.init('frames', len(frames))
    for i, (timestamp, detections) in enumerate(frames):
        build_frame(frame[i], timestamp, detections)
    ```
    """
    assert detection_format == 'deeppipeline'
    frame.dataSource = int(data_source)
    detec_builder = frame.detectionsUnion.init('detectionsDP',
                                               len(detections))
    for i, detection in enumerate(detections):
        detec_builder[i].tagIdx = int(detection[0])
        detec_builder[i].xpos = int(detection[1])
        detec_builder[i].ypos = int(detection[2])
        detec_builder[i].xposHive = int(detection[3])
        detec_builder[i].yposHive = int(detection[4])
        detec_builder[i].zRotation = float(detection[5])
        detec_builder[i].yRotation = float(detection[6])
        detec_builder[i].xRotation = float(detection[7])
        detec_builder[i].radius = float(detection[8])

        nb_ids = len(detection) - _detection_dp_fields_before_ids
        decodedId = detec_builder[i].init('decodedId', nb_ids)
        for j, bit in enumerate(
                detections[i, _detection_dp_fields_before_ids:]):
            decodedId[j] = int(round(255*bit))


def build_frame_container(from_ts, to_ts, cam_id,
                          data_source_fname=None,
                          video_preview_fname=None,
                          video_first_frame_idx=None,
                          video_last_frame_idx=None,
                          ):
    """
    Builds a FrameContainer

    Args:
        from_ts (int or float): Timestamp of the first frame
        to_ts (int or float): Timestamp of the last frame
        data_source_fname (Optional str): Filename of the data source.
        video_preview_fname (Optional str): Filename of the preview video.
        video_first_frame_idx (Optional str): Index of the frist frame used
            from the video.
        video_last_frame_idx (Optional str): Index of the last frame used
            from the video.
    """
    fc = FrameContainer.new_message()
    fc.fromTimestamp = from_ts
    fc.toTimestamp = to_ts
    data_sources = fc.init('dataSources', 1)

    fc = FrameContainer.new_message()
    fc.fromTimestamp = from_ts
    fc.toTimestamp = to_ts
    data_sources = fc.init('dataSources', 1)
    data_source = data_sources[0]
    cam = data_source.cam
    cam.camId = cam_id
    if data_source_fname is not None:
        data_source.filename = data_source_fname
    if video_preview_fname is not None:
        data_source.videoPreviewFilename = video_preview_fname
    if video_first_frame_idx is not None:
        data_source.videoFirstFrameIdx = video_first_frame_idx
    if video_last_frame_idx is not None:
        data_source.videoFirstFrameIdx = video_last_frame_idx
    return fc


def load_frame_container(fname):
    """Loads frame container from this filename"""
    with open(fname, 'rb') as f:
        return FrameContainer.read(f)


class Repository:
    """
    The Repository class manages multiple bb_binary files. It creates a
    directory layout that enables fast access by the timestamp.
    """
    def __init__(self, root_dir, directory_breadths=None):
        """
        Creates a new repository at `root_dir`.

        Args:
            root_dir (str):  Path where the repository is created
            directory_breadths (Optional list[int]): breaths of the directories
                at the different level. For an int value of `n` there are
                `10**n` possible branch-offs at this level.
                Default value is `[3, 3, 3, 5]`.
        """
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if directory_breadths is None:
            directory_breadths = [3, 3, 3, 5]
        self.directory_breadths = directory_breadths
        if not os.path.exists(self._repo_json_fname()):
            self._save_json()

    @property
    def nb_levels(self):
        """the number of directory levels"""
        # last breaths is the file level
        return len(self.directory_breadths) - 1

    @property
    def max_ts(self):
        """Returns the maximum possible timestamp"""
        return 10**sum(self.directory_breadths) - 1

    def can_contain_ts(self, timestamp):
        """Can this repository contain the timestamp"""
        return 0 <= timestamp < self.max_ts

    def add(self, frame_container):
        """
        Adds the `frame_container` to the repository.
        """
        begin = frame_container.fromTimestamp
        end = frame_container.toTimestamp
        cam_id = get_cam_id(frame_container)
        fname, _ = self._create_file_and_symlinks(begin, end, cam_id, 'bbb')
        with open(fname, 'w') as f:
            frame_container.write(f)

    def open(self, timestamp, cam_id):
        """
        Finds and load the FrameContainer that matches the timestamp and the
        cam_id.
        """
        fnames = self.find(timestamp)
        for fname in fnames:
            cam_id, _, _ = parse_fname(fname)
            if cam_id == cam_id:
                return load_frame_container(fname)

    def find(self, ts):
        """
        Returns all files that includes detections to the given timestamp `ts`.
        """
        path = self._path_for_ts(ts)
        try:
            fnames = self._all_files_in(path)
        except FileNotFoundError:
            return []
        parts = [parse_fname(f) for f in fnames]
        begin_end_fnames = [(p[1], p[2], f) for p, f in zip(parts, fnames)]
        found_files = []
        for begin, end, fname in begin_end_fnames:
            if begin <= ts < end:
                found_files.append(self._join_with_repo_dir(path, fname))
        return found_files

    def get_directory_for_ts(self, timestamp):
        """Returns the directory where this timestamp would be stored."""
        return self._join_with_repo_dir(
            *self._path_for_ts(timestamp))

    @staticmethod
    def load(directory):
        """Load the repository from this directory."""
        config_fname = os.path.join(directory, 'bbb_repo.json')
        assert config_fname, \
            "Tries to load directory: {}, but file {} is missing".\
            format(directory, config_fname)
        with open(config_fname) as f:
            config = json.load(f)
        return Repository(**config)

    def __eq__(self, other):
        return self._to_config() == other._to_config()

    def _save_json(self):
        with open(self._repo_json_fname(), 'w+') as f:
            json.dump(self._to_config(), f)

    def _path_for_ts(self, timestamp):
        assert self.can_contain_ts(timestamp)

        def convert_timestamp_to_path(ts, max_digets):
            format_str = "{{:0{}d}}".format(max_digets)
            return format_str.format(ts)

        def split_number(n):
            for base in self._level_bases():
                n_d = int(math.floor(n / base))
                yield n_d
                n -= n_d * base

        path_pieces = []
        for t, d in zip(split_number(timestamp), self.directory_breadths):
            path_pieces.append(convert_timestamp_to_path(t, d))
        return os.path.join(*path_pieces[:-1])

    def _cumsum_directory_breadths(self):
        return [sum(self.directory_breadths[i:])
                for i in range(0, len(self.directory_breadths))]

    def _level_bases(self):
        exponents = self._cumsum_directory_breadths()[1:] + [1]
        return [10**e for e in exponents]

    def _join_with_repo_dir(self, *paths):
        return os.path.join(self.root_dir, *paths)

    def _all_files_in(self, *paths):
        def isfile_or_link(fname):
            return os.path.isfile(fname) or os.path.islink(fname)
        dirname = self._join_with_repo_dir(*paths)
        return [f for f in os.listdir(dirname)
                if isfile_or_link(os.path.join(dirname, f))]

    def _get_timestamp_from_path(self, path):
        path_splits = path.split(os.path.sep)[-self.nb_levels:]
        path_splits = list(map(int, path_splits))
        ts = 0
        for base, dir_number in zip(self._level_bases(), path_splits):
            ts += dir_number * base
        return ts

    def _one_directory_earlier(self, path):
        if type(path) == int:
            path = self._path_for_ts(path)
        ts = self._get_timestamp_from_path(path)
        ts -= 10**self._cumsum_directory_breadths()[-1]
        return self._path_for_ts(ts)

    def _create_file_and_symlinks(self, begin_ts, end_ts, cam_id,
                                  extension=''):
        def spans_multiple_directories(first_ts, end_ts):
            return self._path_for_ts(first_ts) != \
                self._path_for_ts(end_ts)
        fname = self._get_filename(begin_ts, end_ts, cam_id, extension)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        if not os.path.exists(fname):
            open(fname, 'a').close()
        iter_ts = end_ts
        symlinks = []
        while spans_multiple_directories(begin_ts, iter_ts):
            path = self._path_for_ts(iter_ts)
            link_fname = self._get_filename(begin_ts, end_ts, cam_id,
                                            extension, path)
            symlinks.append(link_fname)
            link_dir = os.path.dirname(link_fname)
            os.makedirs(link_dir, exist_ok=True)
            rel_goal = os.path.relpath(fname, start=link_dir)
            os.symlink(rel_goal, link_fname)
            iter_path = self._one_directory_earlier(iter_ts)
            iter_ts = self._get_timestamp_from_path(iter_path)
        return fname, symlinks

    def _get_filename(self, begin_ts, end_ts, cam_id, extension='', path=None):
        assert type(cam_id) is int
        assert type(path) is str or path is None

        if path is None:
            path = self._path_for_ts(begin_ts)
        basename = get_video_fname(cam_id, begin_ts, end_ts)
        if extension != '':
            basename += '.' + extension
        return self._join_with_repo_dir(path, basename)

    def _repo_json_fname(self):
        return os.path.join(self.root_dir, 'bbb_repo.json')

    def _to_config(self):
        return {
            'root_dir': self.root_dir,
            'directory_breadths': self.directory_breadths,
        }
