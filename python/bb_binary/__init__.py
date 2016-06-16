# -*- coding: utf-8 -*-
import capnp
import os
import errno
import numpy as np
import numpy.lib.recfunctions as rf
import json
import math
from datetime import datetime, timedelta
import iso8601
import pytz
import six


capnp.remove_import_hook()
_dirname = os.path.dirname(os.path.realpath(__file__))

bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))
Frame = bbb.Frame
FrameContainer = bbb.FrameContainer
DataSource = bbb.DataSource
DetectionCVP = bbb.DetectionCVP
DetectionDP = bbb.DetectionDP

_TIMEZONE = pytz.timezone('Europe/Berlin')


def get_timezone():
    return _TIMEZONE


def to_timestamp(dt):
    try:
        return dt.timestamp()
    except AttributeError:  # python 2
        utc_naive = dt.replace(tzinfo=None) - dt.utcoffset()
        timestamp = (utc_naive - datetime(1970, 1, 1)).total_seconds()
        return timestamp


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_image_fname_beesbook(fname):
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

    dt = datetime(year, month, day, hour, minute, second, us)
    ts = to_timestamp(get_timezone().localize(dt))
    return camIdx, ts


def parse_image_fname_iso(fname):
    basename = os.path.basename(fname)
    name = basename.split('.')[0]
    _, camIdxStr, iso_str = name.split('_')
    dt = iso8601.parse_date(iso_str)
    return int(camIdxStr), dt


def parse_image_fname(fname, format='iso'):
    assert format in ['beesbook', 'iso']
    if format == 'beesbook':
        return parse_image_fname_beesbook(fname)
    else:
        return parse_image_fname_iso(fname)


def parse_video_fname(fname, format='beesbook'):
    fname = os.path.basename(fname)
    if format == 'beesbook':
        begin_name, end_name = fname.split('_TO_')
        (camIdx, begin) = parse_image_fname(begin_name, format)
        (_, end) = parse_image_fname(end_name, format)
        return camIdx, begin, end
    elif format == 'iso':
        fname, _ = os.path.splitext(fname)
        _, camIdx, isotimespan = fname.split('_')
        start, end = isotimespan.split('--')
        end = end.rstrip(".bbb")
        return int(camIdx), iso8601.parse_date(start), iso8601.parse_date(end)
    else:
        raise ValueError("Unknown format {}.".format(format))


def parse_cam_id(fname):
    fname = os.path.basename(fname)
    camIdx = fname.split("_")[1]
    return int(camIdx)


def parse_fname(fname):
    fname = os.path.basename(fname)
    try:
        return parse_video_fname(fname, 'iso')
    except:
        try:
            return parse_video_fname(fname, 'beesbook')
        except:
            camIdx, dtime = parse_image_fname(fname)
            return camIdx, dtime, dtime


def to_datetime(t):
    if type(t) in (int, float):
        dt = datetime.fromtimestamp(t, tz=pytz.utc)
        return dt
    elif type(t) == datetime:
        return t
    else:
        TypeError("Cannot convert {} to datetime".format(t))


def dt_to_str(dt):
    dt = to_datetime(dt)
    isoformat = "%Y%m%dT%H%M%S"

    dt_str = dt.strftime(isoformat)
    if dt.microsecond != 0:
        dt_str += ".{:03d}".format(dt.microsecond // 10**3)
    if dt.utcoffset().total_seconds() == 0:
        return dt_str + "Z"
    else:
        return dt_str + dt.strftime("%z")


def get_fname(camIdx, dt):
    dt = to_datetime(dt)
    return ("Cam_{cam}_{ts}").format(
                cam=camIdx,
                ts=dt_to_str(dt)
            )


def get_video_fname(camIdx, begin, end):
    return get_fname(camIdx, begin) + "--" + dt_to_str(end)


def convert_frame_to_numpy(frame, keys=None):
    """Returns the frame data and detections as a numpy array from the frame.

    Note: the frame id is identified in the array as frameId instead of id!

    Args:
        frame (Frame): datastructure with frame data from capnp.
        keys (Optional tuple): only these keys are converted to the np array.
    """
    frame_arr = _convert_frame_to_numpy(frame, keys)
    detection_arr = None

    if keys is None or 'detectionsUnion' in keys:
        union_type = frame.detectionsUnion.which()
        if union_type == 'detectionsDP':
            detections = frame.detectionsUnion.detectionsDP
        elif union_type == 'detectionsCVP':
            detections = frame.detectionsUnion.detectionsCVP
        else:
            raise KeyError("Type {0} not supported.".format(union_type))
        detection_arr = _convert_detections_to_numpy(detections, keys)

    if frame_arr is None:
        return detection_arr

    if detection_arr is None:
        return frame_arr

    frame_arr = np.repeat(frame_arr, len(detections), axis=0)
    return rf.merge_arrays((frame_arr, detection_arr),
                           flatten=True, usemask=False)


def _convert_frame_to_numpy(frame, keys=None):
    """Helper function for `convert_frame_to_numpy(frame, keys)`.

    Converts the frame data to a numpy array.
    """
    # automatically deduce keys and types from frame
    frame_keys = set(frame.to_dict().keys())
    frame_keys.discard('detectionsUnion')
    if keys is None:
        keys = list(frame_keys)
    else:
        keys = set(keys)
        if 'frameId' in keys:
            keys.add('id')
        keys = list(keys & frame_keys)

    # abort if no frame information should be extracted
    if len(keys) == 0:
        return None

    fields = [getattr(frame, key) for key in keys]
    formats = [type(field) for field in fields]

    # create frame
    frame_arr = np.array(tuple(fields),
                         dtype={'names': keys, 'formats': formats})

    # replace id with frameId for better readability!
    frame_arr.dtype.names = ["frameId" if x == "id" else x
                             for x in frame_arr.dtype.names]

    return frame_arr


def _convert_detections_to_numpy(detections, keys=None):
    """Helper function for `convert_frame_to_numpy(frame, keys)`.

    Converts the detections data to a numpy array.
    """
    nrows = len(detections)

    # automatically deduce keys and types except for decodedId
    detection0 = detections[0].to_dict()
    detection_keys = set(detection0.keys())
    if keys is None:
        keys = list(detection_keys)
    else:
        keys = list(set(keys) & detection0.keys())

    # abort if no frame information should be extracted
    if len(keys) == 0:
        return None

    formats = [type(detection0[key]) for key in keys]

    decoded_id_key = "decodedId"
    decoded_id_index = None
    # special handling of decodedId as float array in CP pipeline data
    if decoded_id_key in keys and isinstance(detection0[decoded_id_key], list):
        decoded_id_index = keys.index(decoded_id_key)
        formats[decoded_id_index] = str(len(detection0[decoded_id_key])) + 'f8'

    detection_arr = np.empty(nrows, dtype={'names': keys, 'formats': formats})
    for i, detection in enumerate(detections):
        # make sure we have the same order as in keys
        val = [getattr(detection, key) for key in keys]
        if decoded_id_index is not None:
            val[decoded_id_index] = np.array(val[decoded_id_index]) / 255.
        # structured np arrays only accept tuples
        detection_arr[i] = tuple(val)

    return detection_arr


def get_detections(frame):
    """Extracts detections of CP or CVP from frame."""
    union_type = frame.detectionsUnion.which()
    if union_type == 'detectionsDP':
        detections = frame.detectionsUnion.detectionsDP
    elif union_type == 'detectionsCVP':
        detections = frame.detectionsUnion.detectionsCVP
    else:
        raise KeyError("Type {0} not supported.".format(union_type))
    return detections


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
    The structure of the numpy array:
        idx
        xpos
        ypos
        xposHive
        yposHive
        zRotation
        yRotation
        xRotation
        radius
        bit_0
        bit_1
        ...
        bit_n

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
    frame.dataSourceIdx = int(data_source)
    detec_builder = frame.detectionsUnion.init('detectionsDP',
                                               len(detections))
    for i, detection in enumerate(detections):
        detec_builder[i].idx = int(detection[0])
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
                          hive_id=None,
                          transformation_matrix=None,
                          data_source_fname=None,
                          video_preview_fname=None,
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
    fc.camId = cam_id
    if hive_id is not None:
        fc.hiveId = hive_id
    if transformation_matrix is not None:
        fc.transformationMatrix = transformation_matrix
    if data_source_fname is not None:
        data_source.filename = data_source_fname
    if video_preview_fname is not None:
        data_source.videoPreviewFilename = video_preview_fname
    return fc


def load_frame_container(fname):
    """Loads frame container from this filename"""
    with open(fname, 'rb') as f:
        return FrameContainer.read(f)


class Repository(object):
    """
    The Repository class manages multiple bb_binary files. It creates a
    directory layout that enables fast access by the timestamp.
    """

    _CONFIG_FNAME = '.bbb_repo_config.json'

    def __init__(self, root_dir, minute_step=20):
        """
        Creates a new repository at `root_dir`.

        Args:
            root_dir (str):  Path where the repository is created
        """
        self.root_dir = os.path.abspath(root_dir)
        mkdir_p(self.root_dir)
        self.minute_step = minute_step
        if not os.path.exists(self._repo_json_fname()):
            self._save_json()

    def add(self, frame_container):
        """
        Adds the `frame_container` to the repository.
        """
        begin = frame_container.fromTimestamp
        end = frame_container.toTimestamp
        cam_id = frame_container.camId
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
            fname_cam_id = parse_cam_id(fname)
            if cam_id == fname_cam_id:
                return load_frame_container(fname)

    def find(self, ts, cam=None):
        """
        Returns all files that includes detections to the given timestamp `ts`.
        """
        dt = to_datetime(ts)
        path = self._path_for_dt(dt)
        if not os.path.exists(self._join_with_repo_dir(path)):
            return []
        fnames = self._all_files_in(path)
        parts = [self._parse_repo_fname(f) for f in fnames]
        if cam is not None:
            parts = list(filter(lambda p: p[0] == cam, parts))
        found_files = []
        for (camId, begin, end), fname in zip(parts, fnames):
            if begin <= dt < end:
                found_files.append(self._join_with_repo_dir(path, fname))
        return found_files

    def iter_fnames(self, begin=None, end=None, cam=None):
        """
        Returns a generator that yields filenames in sorted order.
        From `begin` to `end`.

        Args:
            begin (Optional timestamp): The first filename contains at least one frame with a
                timestamp greater or equal to `begin`. If `begin` is not set, it will
                start with the earliest file.
            end (Optional timestamp): The last filename contains at least one
                frame with a timestamp smaller then `end`.
                If not set, it will continue until the last file.
            cam (Optional int): Only yield filenames with this cam id.

        Example:

            Files:        A     B     C        D     E
            Frames:    |-----|-----|-----|  |-----|-----|
                            ⬆          ⬆
                          begin       end

            This should return the files A, B and C.
            If `begin` and `end` are `None`, then all will be yield.
        """

        def remove_links(directory, fnames):
            return list(filter(
                lambda f: not os.path.islink(os.path.join(directory, f)),
                fnames))

        if begin is None:
            current_path = self._get_earliest_path()
            begin = pytz.utc.localize(datetime.min)
        else:
            begin = to_datetime(begin)
            current_path = self._path_for_dt(begin, abs=True)

        if current_path == self.root_dir:
            return

        if end is None:
            end_dir = self._get_latest_path()
            end = pytz.utc.localize(datetime.max)
        else:
            end = to_datetime(end)
            end_dir = self._path_for_dt(end, abs=True)

        first_directory = True
        while True:
            fnames = self._all_files_in(current_path)
            if not first_directory:
                fnames = remove_links(current_path, fnames)
            else:
                first_directory = False

            parsed_fname = [self._parse_repo_fname(f) for f in fnames]
            cam_id_begin_end_fnames = [(c, b, e, f) for (c, b, e), f in zip(parsed_fname, fnames)]
            if cam is not None:
                cam_id_begin_end_fnames = list(filter(lambda p: p[0] == cam,
                                                      cam_id_begin_end_fnames))

            cam_id_begin_end_fnames.sort(key=lambda p: p[1])
            for cam_idx, begin_ts, end_ts, fname in cam_id_begin_end_fnames:
                if begin <= end_ts and begin_ts < end:
                    yield self._join_with_repo_dir(current_path, fname)

            if end_dir == current_path:
                break
            else:
                current_path = self._step_to_next_directory(
                    current_path, direction='forward')
                current_path = self._join_with_repo_dir(current_path)

    @staticmethod
    def load(directory):
        """Load the repository from this directory."""
        config_fname = os.path.join(directory, Repository._CONFIG_FNAME)
        assert config_fname, \
            "Tries to load directory: {}, but file {} is missing".\
            format(directory, config_fname)
        with open(config_fname) as f:
            config = json.load(f)
        config['root_dir'] = directory
        return Repository(**config)

    def __eq__(self, other):
        return self._to_config() == other._to_config()

    def _save_json(self):
        with open(self._repo_json_fname(), 'w+') as f:
            json.dump(self._to_config(), f)

    def _path_for_dt(self, time, abs=False):
        dt = to_datetime(time)
        minutes = int(math.floor(dt.minute / self.minute_step) * self.minute_step)
        path = "{:04d}/{:02d}/{:02d}/{:02d}/{:02d}".format(
            dt.year, dt.month, dt.day, dt.hour, minutes)
        if abs:
            return self._join_with_repo_dir(path)
        else:
            return path

    def _get_directory(self, selection_fn):
        def directories(dirname):
            return [os.path.join(dirname, d) for d in os.listdir(dirname)
                    if os.path.isdir(os.path.join(dirname, d))]

        current_dir = self.root_dir
        while True:
            dirs = directories(current_dir)

            if dirs:
                current_dir = selection_fn(dirs)
            else:
                return current_dir

    def _get_earliest_path(self):
        return self._get_directory(min)

    def _get_latest_path(self):
        return self._get_directory(max)

    def _join_with_repo_dir(self, *paths):
        return os.path.join(self.root_dir, *paths)

    def _all_files_in(self, *paths):
        def isfile_or_link(fname):
            return os.path.isfile(fname) or os.path.islink(fname)
        dirname = self._join_with_repo_dir(*paths)
        return [f for f in os.listdir(dirname)
                if isfile_or_link(os.path.join(dirname, f))]

    def _get_time_from_path(self, path):
        path = path.rstrip('/\\')
        time_parts_str = path.split('/')[-5:]
        time_parts = list(map(int, time_parts_str))
        return datetime(*time_parts, tzinfo=pytz.utc)

    def _step_one_directory(self, path, direction='forward'):
        if type(path) == str:
            dt = self._get_time_from_path(path)
        else:
            dt = to_datetime(path)

        offset = timedelta(minutes=self.minute_step)
        if direction == 'forward':
            dt += offset
        elif direction == 'backward':
            dt -= offset
        else:
            raise ValueError("Unknown direction {}.".format(direction))
        return self._path_for_dt(dt)

    def _step_to_next_directory(self, path, direction='forward'):
        if direction == 'forward':
            end = self._get_latest_path()
            path_dt = self._get_time_from_path(path)
            end_dt = self._get_time_from_path(end)
            assert path_dt < end_dt

        elif direction == 'backward':
            begin = self._get_earliest_path()
            assert self._get_time_from_path(begin) < \
                self._get_time_from_path(path)
        else:
            raise ValueError("Unknown direction {}.".format(direction))
        current_path = path
        while True:
            current_path = self._step_one_directory(current_path, direction)
            if os.path.exists(self._join_with_repo_dir(current_path)):
                return current_path

    def _create_file_and_symlinks(self, begin, end, cam_id,
                                  extension=''):
        begin_dt = to_datetime(begin)
        end_dt = to_datetime(end)

        def spans_multiple_directories(first_ts, end_dt):
            return self._path_for_dt(first_ts) != \
                self._path_for_dt(end_dt)
        fname = self._get_filename(begin_dt, end_dt, cam_id, extension)
        mkdir_p(os.path.dirname(fname))
        if not os.path.exists(fname):
            open(fname, 'a').close()
        iter_ts = end
        symlinks = []
        while spans_multiple_directories(begin_dt, iter_ts):
            path = self._path_for_dt(iter_ts)
            link_fname = self._get_filename(begin_dt, end, cam_id,
                                            extension, path)
            symlinks.append(link_fname)
            link_dir = os.path.dirname(link_fname)
            mkdir_p(link_dir)
            rel_goal = os.path.relpath(fname, start=link_dir)
            os.symlink(rel_goal, link_fname)
            iter_path = self._step_one_directory(iter_ts, 'backward')
            iter_ts = self._get_time_from_path(iter_path)
        return fname, symlinks

    def _get_filename(self, begin_dt, end_dt, cam_id, extension='', path=None):
        assert type(cam_id) in six.integer_types
        assert type(path) is str or path is None

        if path is None:
            path = self._path_for_dt(begin_dt)
        basename = get_video_fname(cam_id, begin_dt, end_dt)
        if extension != '':
            basename += '.' + extension
        return self._join_with_repo_dir(path, basename)

    def _repo_json_fname(self):
        return os.path.join(self.root_dir, self._CONFIG_FNAME)

    def _to_config(self):
        return {
            'root_dir': self.root_dir,
            'minute_step': self.minute_step,
        }

    @staticmethod
    def _parse_repo_fname(fname):
        try:
            return parse_video_fname(fname, format='iso')
        except:
            return parse_image_fname_iso(fname)
