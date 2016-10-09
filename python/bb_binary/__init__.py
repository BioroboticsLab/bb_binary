# -*- coding: utf-8 -*-
import os
import errno
import json
import math
from datetime import datetime, timedelta
import numpy as np
import numpy.lib.recfunctions as rf
import iso8601
import pytz
import six
import capnp

# TODO: Add warning about windows symlinks


capnp.remove_import_hook()
_dirname = os.path.dirname(os.path.realpath(__file__))

bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))
Frame = bbb.Frame
FrameContainer = bbb.FrameContainer
DataSource = bbb.DataSource
DetectionCVP = bbb.DetectionCVP
DetectionDP = bbb.DetectionDP
DetectionTruth = bbb.DetectionTruth

_TIMEZONE = pytz.timezone('Europe/Berlin')

CAM_IDX = 0
BEGIN_IDX = 1
TIME_IDX = 1
END_IDX = 2


def get_timezone():
    return _TIMEZONE


def to_timestamp(dt):
    try:
        return dt.timestamp()
    except AttributeError:  # python 2
        utc_naive = dt.replace(tzinfo=None) - dt.utcoffset()
        timestamp = (utc_naive - datetime(1970, 1, 1)).total_seconds()
        return timestamp


def _mkdir_p(path):
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
    name = name.replace('__', '_')
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
    dt = get_timezone().localize(dt)
    return camIdx, dt.astimezone(pytz.utc)


def parse_image_fname_iso(fname):
    basename = os.path.basename(fname)
    name = basename.split('.')[0]
    _, camIdxStr, iso_str = name.split('_')
    dt = iso8601.parse_date(iso_str)
    return int(camIdxStr), dt


def parse_image_fname(fname, format='auto'):
    if format == 'beesbook':
        return parse_image_fname_beesbook(fname)
    elif format == 'iso':
        return parse_image_fname_iso(fname)
    elif format == 'auto':
        basename = os.path.basename(fname)
        if basename.count('_') >= 3:
            return parse_image_fname_beesbook(fname)
        else:
            return parse_image_fname_iso(fname)
    else:
        raise Exception("Unknown format {}.".format(format))


def parse_video_fname(fname, format='auto'):
    def beesbook_parse():
        begin_name, end_name = basename.split('_TO_')
        (camIdx, begin) = parse_image_fname(begin_name, 'beesbook')
        (_, end) = parse_image_fname(end_name, 'beesbook')
        return camIdx, begin, end

    def iso_parse():
        name, _ = os.path.splitext(basename)
        _, camIdx, isotimespan = name.split('_')
        start, end = isotimespan.split('--')
        end = end.rstrip(".bbb")
        return int(camIdx), iso8601.parse_date(start), iso8601.parse_date(end)

    basename = os.path.basename(fname)
    if format == 'beesbook':
        return beesbook_parse()
    elif format == 'iso':
        return iso_parse()
    elif format == 'auto':
        try:
            return beesbook_parse()
        except ValueError:
            return iso_parse()
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
        raise TypeError("Cannot convert {} to datetime".format(t))


def dt_to_str(dt):
    dt = to_datetime(dt)
    isoformat = "%Y-%m-%dT%H:%M:%S"

    dt_str = dt.strftime(isoformat)
    if dt.microsecond != 0:
        dt_str += ".{:06d}".format(dt.microsecond)
    if dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0:
        return dt_str + "Z"
    else:
        raise Exception("Got a datetime object not in UTC. Allways use UTC.")


def get_fname(camIdx, dt):
    dt = to_datetime(dt)
    return ("Cam_{cam}_{ts}").format(cam=camIdx, ts=dt_to_str(dt))


def get_video_fname(camIdx, begin, end):
    return get_fname(camIdx, begin) + "--" + dt_to_str(end)


def int_id_to_binary(id, nb_bits=12):
    result = np.zeros(nb_bits, dtype=np.uint8)
    if id >= 2**nb_bits:
        raise Exception("Id overflows {} bits".format(nb_bits))
    a = nb_bits - 1
    while a >= 0:
        result[a] = id & 1
        id >>= 1
        a -= 1
    return result


def convert_frame_to_numpy(frame, keys=None, add_cols=None):
    """Returns the frame data and detections as a numpy array from the frame.

    Note: the frame id is identified in the array as frameId instead of id!

    Args:
        frame (Frame): datastructure with frame data from capnp.
        keys (Optional iterable): only these keys are converted to the np array.
        add_cols (Optional dictionary): additional columns for the np array,
            use either a single value or a sequence of correct length.
    """
    ret_arr = None

    if keys is None or 'detectionsUnion' in keys:
        detections = get_detections(frame)
        ret_arr = _convert_detections_to_numpy(detections, keys)

    frame_arr = _convert_frame_to_numpy(frame, keys)
    if ret_arr is not None and frame_arr is not None:
        frame_arr = np.repeat(frame_arr, ret_arr.shape[0], axis=0)
        ret_arr = rf.merge_arrays((frame_arr, ret_arr),
                                  flatten=True, usemask=False)
    elif frame_arr is not None:
        ret_arr = frame_arr

    if ret_arr is not None and add_cols is not None:
        if keys is None:
            keys = ret_arr.dtype.names
        for key, val in add_cols.items():
            assert key not in keys, "{} not allowed in add_cols".format(key)
            if hasattr(val, '__len__') and not isinstance(val, six.string_types):
                msg = "{} has not length {}".format(key, ret_arr.shape[0])
                assert len(val) == ret_arr.shape[0], msg
            else:
                val = np.repeat(val, ret_arr.shape[0], axis=0)
            ret_arr = rf.append_fields(ret_arr, key, val, usemask=False)

    return ret_arr


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
        keys = list(set(keys) & detection_keys)

    # abort if no information should be extracted
    if len(keys) == 0:
        return None

    formats = [type(detection0[key]) for key in keys]

    readability_key = 'readability'
    decoded_id_key = 'decodedId'
    decoded_id_index = None
    descriptor_key = 'descriptor'
    descriptor_index = None
    if decoded_id_key in keys and isinstance(detection0[decoded_id_key], list):
        # special handling of decodedId as float array in DP pipeline data
        decoded_id_index = keys.index(decoded_id_key)
        formats[decoded_id_index] = str(len(detection0[decoded_id_key])) + 'f8'
    elif readability_key in keys:
        # special handling of enum because numpy does not determine str length
        readbility_index = keys.index(readability_key)
        formats[readbility_index] = 'S10'
    if descriptor_key in keys and isinstance(detection0[descriptor_key], list):
        # special handling of descriptor as uint8 array in DP pipeline data
        descriptor_index = keys.index(descriptor_key)
        formats[descriptor_index] = str(len(detection0[descriptor_key])) + 'u8'
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
    """Extracts detections of DP, CVP or truth data from frame."""
    union_type = frame.detectionsUnion.which()
    if union_type == 'detectionsDP':
        detections = frame.detectionsUnion.detectionsDP
    elif union_type == 'detectionsCVP':
        detections = frame.detectionsUnion.detectionsCVP
    elif union_type == 'detectionsTruth':
        detections = frame.detectionsUnion.detectionsTruth
    else:
        raise KeyError("Type {0} not supported.".format(union_type))  # pragma: no cover
    return detections


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


def build_frame_container_from_df(df, union_type, cam_id, frame_offset=0):
    """Builds a frame container from a Pandas DataFrame.

    Operates differently from `build_frame_container` because it will be used
    in a different context where we have access to more data.

    Column names are matched to `Frame` and `Detection*` attributes.
    Set additional `FrameContainer` attributes like `hiveId` in the return value.

    Args:
        df (dataframe): dataframe with detection data
        union_type (String): the type of detections e.g. detectionsTruth
        cam_id (int): id of camera, also used as `FrameContainer` id
        frame_offset (Optional int): offset for unique frame ids

     Returns:
         FrameContainer: converted data from `df`
         int: number of frames that could be used as `frame_offset`
     """
    def set_attr_from(obj, src, key):
        """Get attr `key` from `src` and set val to `obj` on same `key`"""
        val = getattr(src, key)
        # special handling for list type fields
        if key in list_keys:
            set_list_attr(obj, val, key)
            return
        if type(val).__module__ == np.__name__:
            val = np.asscalar(val)
        setattr(obj, key, val)

    def set_list_attr(obj, list_src, key):
        """Initialize list `key` on `object` and set all values from `list_src`."""
        new_list = obj.init(key, len(list_src))
        for i, val in enumerate(list_src):
            if type(val).__module__ == np.__name__:
                val = np.asscalar(val)
            new_list[i] = val

    detection = {
        'detectionsCVP': DetectionCVP.new_message(),
        'detectionsDP': DetectionDP.new_message(),
        'detectionsTruth': DetectionTruth.new_message()
    }[union_type]

    # check that we have all the information we need
    skip_keys = frozenset(['readability', 'xposHive', 'yposHive', 'frameIdx', 'idx'])
    minimal_keys = set(detection.to_dict().keys()) - skip_keys
    list_keys = set()
    # for some reasons lists are not considered when using to_dict()!
    if union_type == 'detectionsDP':
        minimal_keys = minimal_keys | set(['decodedId'])
        list_keys = list_keys | set(['decodedId', 'descriptor'])
    available_keys = set(df.keys())
    assert minimal_keys <= available_keys,\
        "Missing keys {} in DataFrame.".format(minimal_keys - available_keys)

    # select only entries for cam
    if 'camId' in available_keys:
        df = df[df.camId == cam_id].copy()

    # convert timestamp to unixtimestamp
    if 'datetime' in df.dtypes.timestamp.name:
        df.loc[:, 'timestamp'] = df.loc[:, 'timestamp'].apply(
            lambda t: to_timestamp(datetime(
                t.year, t.month, t.day, t.hour, t.minute, t.second,
                t.microsecond, tzinfo=pytz.utc)))

    # convert decodedId from float to integers (if necessary)
    if 'decodedId' in available_keys and union_type == 'detectionsDP' and\
       np.all(np.array(df.loc[df.index[0], 'decodedId']) < 1.1):
        df.loc[:, 'decodedId'] = df.loc[:, 'decodedId'].apply(
            lambda l: [int(round(fid * 255.)) for fid in l])

    # create frame container
    tstamps = df.timestamp.unique()
    start = np.asscalar(min(tstamps))
    end = np.asscalar(max(tstamps))
    new_frame_offset = frame_offset + len(tstamps)

    fc = build_frame_container(start, end, cam_id)
    fc.id = cam_id  # overwrite if necessary!
    fc.init('frames', len(tstamps))

    # determine which fields we could assign from dataframe to cap'n proto
    frame = Frame.new_message()
    frame_fields = [field for field in available_keys if hasattr(frame, field)]

    detection_fields = [field for field in available_keys if hasattr(detection, field)]

    # create frames (each timestamp maps to a frame)
    for frameIdx, (_, detections) in enumerate(df.groupby(by='timestamp')):
        frame = fc.frames[frameIdx]
        frame.id = frame_offset + frameIdx
        frame.frameIdx = frameIdx

        # take first row, assumes that cols `frame_fields` have unique values!
        for key in frame_fields:
            set_attr_from(frame, detections.iloc[0], key)

        # create detections
        frame.detectionsUnion.init(union_type, detections.shape[0])
        for detectionIdx, row in enumerate(detections.itertuples(index=False)):
            detection = getattr(frame.detectionsUnion, union_type)[detectionIdx]
            detection.idx = detectionIdx
            for key in detection_fields:
                set_attr_from(detection, row, key)

    return fc, new_frame_offset


def load_frame_container(fname):
    """Loads frame container from this filename"""
    with open(fname, 'rb') as f:
        return FrameContainer.read(f)


class TimeInterval(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def in_interval(self, dt):
        return self.begin <= dt < self.end


class Repository(object):
    """
    The Repository class manages multiple bb_binary files. It creates a
    directory layout that enables fast access by the timestamp.
    """

    _CONFIG_FNAME = '.bbb_repo_config.json'

    def __init__(self, root_dir, minute_step=None):
        """
        Creates a new repository at `root_dir`.

        Args:
            root_dir (str):  Path where the repository is created
            minute_step (int): Number of minutes that spans a directory. Default: 20
        """
        self.root_dir = os.path.abspath(root_dir)
        config_fname = os.path.join(self.root_dir, Repository._CONFIG_FNAME)
        if os.path.exists(config_fname):
            if minute_step is not None:
                raise Exception(
                    "Got repo at {} with existing config file {}, but also "
                    "got minute_step {}."
                    .format(self.root_dir, self._CONFIG_FNAME, minute_step))

            with open(config_fname) as f:
                config = json.load(f)
            self.minute_step = config['minute_step']
        else:
            _mkdir_p(self.root_dir)
            if minute_step is None:
                minute_step = 20

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
        TODO: UTC timestamps! Generall
        """
        dt = to_datetime(ts)
        path = self._path_for_dt(dt)
        if not os.path.exists(self._join_with_repo_dir(path)):
            return []
        fnames = self._all_files_in(path)
        fnames_parts = [(f, self._parse_repo_fname(f)) for f in fnames]
        if cam is not None:
            fnames_parts = list(filter(lambda f, p: p[CAM_IDX] == cam, fnames_parts))
        found_files = []
        for fname, (camId, begin, end) in fnames_parts:
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

        .. code::

            Files:        A     B     C        D     E
            Frames:    |-----|-----|-----|  |-----|-----|
                            ⬆          ⬆
                          begin       end

        This should return the files A, B and C.
        If `begin` and `end` are `None`, then all will be yield.
        """

        def remove_links(directory, fnames):
            assert os.path.isdir(directory)
            assert os.path.isabs(directory)
            return list(filter(
                lambda f: not os.path.islink(os.path.join(directory, f)),
                fnames))

        if begin is None:
            current_path = self._get_earliest_path()
            begin = pytz.utc.localize(datetime.min)

        else:
            begin = to_datetime(begin)
            current_path = self._path_for_dt(begin, abs=True)

        # if the repository is empty, current_path is the root_dir
        if current_path == self.root_dir:
            return

        if end is None:
            end_dir = self._get_latest_path()
            end = pytz.utc.localize(datetime.max)
        else:
            end = to_datetime(end)
            end_dir = self._path_for_dt(end, abs=True)

        iter_range = TimeInterval(begin, end)
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
                cam_id_begin_end_fnames = list(filter(lambda p: p[CAM_IDX] == cam,
                                                      cam_id_begin_end_fnames))

            cam_id_begin_end_fnames.sort(key=lambda p: p[BEGIN_IDX])
            for cam_idx, begin_ts, end_ts, fname in cam_id_begin_end_fnames:
                if iter_range.in_interval(begin_ts) or iter_range.in_interval(end_ts):
                    yield self._join_with_repo_dir(current_path, fname)

            if end_dir == current_path:
                break
            else:
                current_path = self._step_to_next_directory(
                    current_path, direction='forward')
                current_path = self._join_with_repo_dir(current_path)

    def iter_frames(self, begin=None, end=None, cam=None):
        """
        Yields frames in sorted order.
        From `begin` to `end`.

        Args:
            begin (Optional timestamp): select frames with begin <= timestamp.
            Starts with smallest timestamp in repository if not set.
            end (Optional timestamp): select frames with timestamp < end.
            Ends with biggest timestamp in repository if not set.
            cam (Optional int): only yield filenames with this cam id.

        Returns:
            iterator: iterator with Frames
            FrameContainer: the corresponding FrameContainer for each frame.
        """
        for f in self.iter_fnames(begin=begin, end=end, cam=cam):
            fc = load_frame_container(f)
            for frame in fc.frames:
                if ((begin is None or begin <= frame.timestamp) and
                   (end is None or frame.timestamp < end)):
                    # it seems to be more efficient to yield a FrameContainer
                    # with frames, instead of extracting all the information
                    # into another data structure.
                    yield frame, fc

    def _save_json(self):
        with open(self._repo_json_fname(), 'w+') as f:
            json.dump(self._to_config(), f)

    _DIR_FORMAT = "{:04d}/{:02d}/{:02d}/{:02d}/{:02d}"
    _DIR_FORMAT_PARTS = _DIR_FORMAT.split('/')

    def _path_for_dt(self, time, abs=False):
        dt = to_datetime(time)
        minutes = int(math.floor(dt.minute / self.minute_step) * self.minute_step)
        values = (dt.year, dt.month, dt.day, dt.hour, minutes)
        format_parts = list(map(lambda t: t[0].format(t[1]),
                                zip(self._DIR_FORMAT_PARTS, values)))
        if abs:
            return self._join_with_repo_dir(*format_parts)
        else:
            return os.path.join(*format_parts)

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

    def _all_files_in(self, path):
        assert type(path) == str

        def isfile_or_link(fname):
            return os.path.isfile(fname) or os.path.islink(fname)
        dirname = self._join_with_repo_dir(path)
        if not os.path.isdir(dirname):
            return []
        return [f for f in os.listdir(dirname)
                if isfile_or_link(os.path.join(dirname, f))]

    def _get_time_from_path(self, path):
        path = os.path.normpath(path)
        if path.startswith(self.root_dir):
            path = path[len(self.root_dir)+1:]
        time_parts_str = path.split(os.path.sep)
        if len(time_parts_str) > 5:
            raise Exception(
                "A path must be of format {}. But got {} with {} parts.".
                format(self._DIR_FORMAT, path, len(time_parts_str)))
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
            # TODO: loop forever?

    def _create_file_and_symlinks(self, begin, end, cam_id,
                                  extension=''):
        begin_dt = to_datetime(begin)
        end_dt = to_datetime(end)

        def spans_multiple_directories(first_ts, end_dt):
            return self._path_for_dt(first_ts) != \
                self._path_for_dt(end_dt)
        fname = self._get_filename(begin_dt, end_dt, cam_id, extension)
        _mkdir_p(os.path.dirname(fname))
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
            _mkdir_p(link_dir)
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
            'minute_step': self.minute_step,
        }

    @staticmethod
    def _parse_repo_fname(fname):
        try:
            return parse_video_fname(fname, format='iso')
        except:
            return parse_image_fname_iso(fname)
