
import capnp
import os
import numpy as np
import json
import math


capnp.remove_import_hook()


_dirname = os.path.dirname(os.path.realpath(__file__))
bbb = capnp.load(os.path.join(_dirname, 'bb_binary_schema.capnp'))
Frame = bbb.Frame
FrameContainer = bbb.FrameContainer
DataSource = bbb.DataSource
Cam = bbb.Cam
DetectionCVP = bbb.DetectionCVP
DetectionDP = bbb.DetectionDP


def convert_detections_to_numpy(frame):
    """
    Returns the detections as a numpy array from the frame.
    """

    union_type = frame.detectionsUnion.which()
    assert union_type == 'detectionsDP', \
        "Currently only the new pipeline format is supported."
    detections = frame.detectionsUnion.detectionsDP

    shape = (len(detections), nb_parameters(detections[0]))
    arr = np.zeros(shape, dtype=np.float32)
    for i, detection in enumerate(detections):
        arr[i, 0] = detection.tagIdx
        arr[i, 1] = detection.xpos
        arr[i, 2] = detection.ypos
        arr[i, 3] = detection.zRotation
        arr[i, 4] = detection.yRotation
        arr[i, 5] = detection.xRotation
        arr[i, 6] = detection.radius
        arr[i, 7:] = np.array(list(detection.decodedId)) / 255

    return arr


def nb_parameters(detection):
    """Returns the number of parameter of the detections."""
    return 7 + len(detection.decodedId)


def build_frame(
        frame_builder,
        timestamp,
        detections,
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
        build_frame(frame_builder[i], timestamp, detections)
    ```
    """
    assert detection_format == 'deeppipeline'
    detec_builder = frame_builder.detectionsUnion.init('detectionsDP',
                                                       len(detections))
    for i, detection in enumerate(detections):
        detec_builder[i].tagIdx = int(detection[0])
        detec_builder[i].xpos = int(detection[1])
        detec_builder[i].ypos = int(detection[2])
        detec_builder[i].zRotation = float(detection[3])
        detec_builder[i].yRotation = float(detection[4])
        detec_builder[i].xRotation = float(detection[5])
        detec_builder[i].radius = float(detection[6])
        decodedId = detec_builder[i].init('decodedId', len(detection) - 7)
        for j, bit in enumerate(detections[i, 7:]):
            decodedId[j] = int(round(255*bit))


class Repository:
    """
    The Repository class manages multiple bb_binary files. It creates a
    directory layout that enables fast access by the timestamp.
    """
    def __init__(self, root_dir, ts_begin=0, directory_depths=None):
        """
        Opens the repository at `root_dir`
        """
        self.root_dir = root_dir
        self.ts_begin = ts_begin
        if directory_depths is None:
            directory_depths = [3, 3, 3, 5]
        self.directory_depths = directory_depths
        if not os.path.exists(self.repo_json_fname()):
            self._save_json()

    def repo_json_fname(self):
        return os.path.join(self.root_dir, 'bbb_repo.json')

    @property
    def max_ts(self):
        return 10**sum(self.directory_depths) - 1

    def can_contain_ts(self, timestamp):
        return self.ts_begin <= timestamp < self.max_ts

    def to_config(self):
        return {
            'root_dir': self.root_dir,
            'ts_begin': self.ts_begin,
            'directory_depths': self.directory_depths,
        }

    def _save_json(self):
        with open(self.repo_json_fname(), 'w+') as f:
            json.dump(self.to_config(), f)

    def add(frame_container: bbb.FrameContainer):
        """
        Adds the `frame_container` to the repository.
        """
        pass

    def _create_file_and_symlinks(self, begin_ts, end_ts, cam_ids,
                                  extension=''):
        fname = self.get_file_name(begin_ts, end_ts, cam_ids, extension)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if not os.path.exists(fname):
            open(fname, 'a').close()

        iter_ts = end_ts
        symlinks = []
        while self.spans_multiple_directories(begin_ts, iter_ts):
            dir_slices = self.directory_slices_for_ts(iter_ts)
            link_fname = self.get_file_name(begin_ts, iter_ts, cam_ids,
                                            extension, dir_slices)
            symlinks.append(link_fname)
            link_dir = os.path.dirname(link_fname)
            os.makedirs(link_dir, exist_ok=True)
            rel_goal = os.path.relpath(fname, start=link_dir)
            os.symlink(rel_goal, link_fname)

            iter_dir_slices = self.one_directory_earlier(iter_ts)
            iter_ts = self.get_timestamp_for_directory_slice(iter_dir_slices)

        return symlinks

    def spans_multiple_directories(self, first_ts, end_ts):
        return self.directory_slices_for_ts(first_ts) != \
            self.directory_slices_for_ts(end_ts)

    def open(timestamp, cam) -> bbb.FrameContainer:
        """
        Finds and load the FrameContainer that matches the timestamp and the
        cam.
        """
        pass

    def get_timestamp_for_directory_slice(self, dir_slices):
        dir_slices = list(map(int, dir_slices))
        ts = 0
        for d, dir_number in zip(self._cumsum_directory_depths(), dir_slices):
            ts += dir_number * 10 ** d
        return ts

    def one_directory_earlier(self, dir_slices):
        if type(dir_slices) == int:
            dir_slices = self.directory_slices_for_ts(dir_slices)
        ts = self.get_timestamp_for_directory_slice(dir_slices)
        ts -= 10**self._cumsum_directory_depths()[-1]
        return self.directory_slices_for_ts(ts)

    def find(self, ts) -> 'list(str)':
        """
        Returns all files that includes detections to the given timestamp `ts`.
        """
        dir_slices = self.directory_slices_for_ts(ts)
        try:
            fnames = self.all_files_in(*dir_slices)
        except FileNotFoundError:
            return []

        parts = [f.split('_') for f in fnames]
        begin_end_fnames = [(int(p[0]), int(p[1]), f)
                            for p, f in zip(parts, fnames)]

        found_files = []
        for begin, end, fname in begin_end_fnames:
            if begin <= ts < end:
                full_slices = dir_slices + [fname]
                found_files.append(self.join_with_repo_dir(*full_slices))
        return found_files

    def get_file_name(self, begin_ts, end_ts, cam_ids, extension='',
                      dir_slices=None):
        if dir_slices is None:
            dir_slices = self.directory_slices_for_ts(begin_ts)
        cam_id_as_str = list(map(str, sorted(cam_ids)))
        basename = "{begin}_{end}_cam_{cam_ids}".format(
            cam_ids="".join(cam_id_as_str),
            begin=begin_ts,
            end=end_ts
        )
        if extension != '':
            basename += '.' + extension
        full_slices = dir_slices + [basename]
        return self.join_with_repo_dir(*full_slices)

    def all_files_in(self, *paths):
        dirname = self.join_with_repo_dir(*paths)
        return [f for f in os.listdir(dirname)
                if os.path.isfile(os.path.join(dirname, f))]

    def join_with_repo_dir(self, *paths):
        return os.path.join(self.root_dir, *paths)

    def _cumsum_directory_depths(self):
        return [sum(self.directory_depths[i:-1])
                for i in range(len(self.directory_depths))]

    def directory_slices_for_ts(self, timestamp):
        assert self.can_contain_ts(timestamp)

        def convert_timestamp_to_path(ts, max_digets):
            format_str = "{{:0{}d}}".format(max_digets)
            return format_str.format(ts)

        def split_number(n):
            d = 0
            depths = [sum(self.directory_depths[i:-1])
                      for i in range(len(self.directory_depths))]
            for d in depths:
                n_d = int(math.floor(n / 10 ** d))
                yield n_d
                n -= n_d * 10 ** (d)

        slices = []
        for t, d in zip(split_number(timestamp), self.directory_depths):
            slices.append(convert_timestamp_to_path(t, d))
        return slices[:-1]

    def common_directory_slices(self, first_ts, second_ts):
        slices = []
        for f, s in zip(self.directory_slices_for_ts(first_ts),
                        self.directory_slices_for_ts(second_ts)):
            if f == s:
                slices.append(f)
            else:
                return slices

    @staticmethod
    def load(root_dir):
        config_fname = os.path.join(root_dir, 'bbb_repo.json')
        assert config_fname, \
            "Tries to load directory: {}, but file {} is missing".\
            format(root_dir, config_fname)
        with open(config_fname) as f:
            config = json.load(f)
        return Repository(**config)

    def __eq__(self, other):
        return self.to_config() == other.to_config()
