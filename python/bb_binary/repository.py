# -*- coding: utf-8 -*-
"""
The Repository class allows to read and write to *bb_binary* data stores.

The *bb_binary* data is splitted into several files that are organized in subfolders via date and
time. A :class:`.Repository` manages these *bb_binary* datafiles and provides methods to
**add** new :obj:`.FrameContainer`, or iterate through :obj:`.Frame`.

This class provides performant access to a *bb_binary* data store but you will have to parse
the data by yourself. You might use some helpers from :doc:`parsing`.
"""
import os
import errno
import json
import math
from datetime import datetime, timedelta
import pytz
import six

from .common import FrameContainer, CAM_IDX, BEGIN_IDX
from .parsing import get_video_fname, to_datetime, parse_cam_id, parse_video_fname, \
    parse_image_fname_iso


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def load_frame_container(fname):
    """Loads :obj:`.FrameContainer` from this filename."""
    with open(fname, 'rb') as f:
        return FrameContainer.read(f, traversal_limit_in_words=2**63)


class Repository(object):
    """The Repository class manages multiple *bb_binary* files. It creates a
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
        """Adds the `frame_container` of type :obj:`.FrameContainer` to the repository.
        """
        begin = frame_container.fromTimestamp
        end = frame_container.toTimestamp
        cam_id = frame_container.camId
        fname, _ = self._create_file_and_symlinks(begin, end, cam_id, 'bbb')
        with open(fname, 'w') as f:
            frame_container.write(f)

    def open(self, timestamp, cam_id):
        """Finds and load the :obj:`.FrameContainer` that matches the `timestamp` and `cam_id`.
        """
        fnames = self.find(timestamp)
        for fname in fnames:
            fname_cam_id = parse_cam_id(fname)
            if cam_id == fname_cam_id:
                return load_frame_container(fname)

    def find(self, ts, cam=None):
        """Returns all files that includes detections to the given timestamp `ts`.

        TODO:
            UTC timestamps! Generall
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

    def iter_fnames(self, begin=None, end=None, cam=None, fname_filter=None):
        """ Returns a generator that yields filenames in sorted order.

        From `begin` to `end`.

        Args:
            begin (Optional timestamp): The first filename contains at least one frame with a
                timestamp greater or equal to `begin`. If `begin` is not set, it will
                start with the earliest file.
            end (Optional timestamp): The last filename contains at least one
                frame with a timestamp smaller then `end`.
                If not set, it will continue until the last file.
            cam (Optional int): Only yield filenames with this cam id.
            fname_filter (Optional function): only yield fnames for which the function
                returns true

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

        last_timestamp = self._get_time_from_path(self._get_latest_path())
        if end is None:
            end = pytz.utc.localize(datetime.max)
        else:
            end = to_datetime(end)
            if end < last_timestamp:
                last_timestamp = end

        iter_range = TimeInterval(begin, end)
        first_directory = True
        while True:
            fnames = self._all_files_in(current_path)
            if not first_directory:
                fnames = remove_links(current_path, fnames)
            else:
                first_directory = False

            parsed_fname = [self._parse_repo_fname(f) for f in fnames]
            # c = camId, b = begin, e = end, f = filename
            cam_id_begin_end_fnames = [(c, b, e, f) for (c, b, e), f in zip(parsed_fname, fnames)]
            if cam is not None:
                cam_id_begin_end_fnames = list(filter(lambda p: p[CAM_IDX] == cam,
                                                      cam_id_begin_end_fnames))

            cam_id_begin_end_fnames.sort(key=lambda p: p[BEGIN_IDX])
            for cam_idx, begin_ts, end_ts, fname in cam_id_begin_end_fnames:
                if iter_range.in_interval(begin_ts) or iter_range.in_interval(end_ts) or \
                   (begin > begin_ts and end < end_ts):
                    if fname_filter is not None and fname_filter(fname) is False:
                        continue
                    yield self._join_with_repo_dir(current_path, fname)

            current_timestamp = self._get_time_from_path(current_path)
            if current_timestamp >= last_timestamp:
                break
            else:
                current_path = self._step_to_next_directory(
                    current_path, direction='forward')
                current_path = self._join_with_repo_dir(current_path)

    def iter_frames(self, begin=None, end=None, cam=None, frame_filter=None):
        """
        Yields frames with their corresponding FrameContainers. The FrameContainers are ordered
        in time. Beware that individual frames may not be in order if cam is not set.

        From `begin` to `end`.

        Args:
            begin (Optional timestamp): select frames with begin <= timestamp.
            Starts with smallest timestamp in repository if not set.
            end (Optional timestamp): select frames with timestamp < end.
            Ends with biggest timestamp in repository if not set.
            cam (Optional int): only yield filenames with this cam id.
            frame_filter (Optional function): only yield frames for which the function
                returns true

        Returns:
            (tuple): tuple containing:

                iterator (iterable): iterator with Frames
                FrameContainer (FrameContainer): the corresponding FrameContainer for each frame.
        """
        for f in self.iter_fnames(begin=begin, end=end, cam=cam):
            fc = load_frame_container(f)
            for frame in fc.frames:
                if frame_filter is not None and frame_filter(frame) is False:
                    continue
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
            if not os.path.islink(link_fname):
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
            return parse_video_fname(fname, format='bbb')
        except:
            return parse_image_fname_iso(fname)


class TimeInterval(object):
    """Helper class to represent time intervals."""
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    def in_interval(self, dt):
        return self.begin <= dt < self.end
