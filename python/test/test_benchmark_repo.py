# -*- coding: utf-8 -*-
#
import os
import random
import shutil
import tempfile
import time

import numpy as np
import pytest

from bb_binary import build_frame_container, Repository
from conftest import fill_repository


@pytest.fixture
def example_experiment_repo(request):
    tmpdir = tempfile.mkdtemp(prefix=os.path.dirname(__file__) + "_tmpdir_")
    repo = Repository(tmpdir)
    experiment_duration = 6*7*24*3600
    one_video = int(1024 / 3)
    begin = int(time.time())
    end = begin + experiment_duration

    begin_end_cam_id = []
    for cam_id in range(4):
        begin_end_cam_id.extend([(ts, ts + one_video, cam_id)
                                 for ts in range(begin, end, one_video)])
    fill_repository(repo, begin_end_cam_id)

    def fin():
        shutil.rmtree(tmpdir)
    request.addfinalizer(fin)
    return repo, begin, end


@pytest.mark.slow
def test_benchmark_find(benchmark, example_experiment_repo):
    repo, begin, end = example_experiment_repo

    def find():
        ts = random.randint(begin, end)
        repo.find(ts)
    benchmark(find)


_detection_dp_fields_before_ids = 9


def build_frame(
        frame,
        timestamp,
        detections,
        frame_idx,
        data_source=0,
        detection_format='deeppipeline'
):
    """
    Builds a frame from a numpy array.
    The columns of the ``detections`` array must be in this order:

       * ``idx``
       * ``xpos``
       * ``ypos``
       * ``zRotation``
       * ``yRotation``
       * ``xRotation``
       * ``radius``
       * ``bit_0``
       * ``bit_1``
       * ``...``
       * ``bit_n``


    Usage (not tested):

    .. code::

        frames = [(timestamp, pipeline(image_to_timestamp))]
        nb_frames = len(frames)
        fc = FrameContainer.new_message()
        frames_builder = fc.init('frames', len(frames))
        for i, (timestamp, detections) in enumerate(frames):
            build_frame(frame[i], timestamp, detections)
    """
    # TODO: Use a structed numpy array
    assert detection_format == 'deeppipeline'
    frame.dataSourceIdx = int(data_source)
    frame.frameIdx = int(frame_idx)
    detec_builder = frame.detectionsUnion.init('detectionsDP',
                                               len(detections))
    for i, detection in enumerate(detections):
        detec_builder[i].idx = int(detection[0])
        detec_builder[i].xpos = int(detection[1])
        detec_builder[i].ypos = int(detection[2])
        detec_builder[i].zRotation = float(detection[3])
        detec_builder[i].yRotation = float(detection[4])
        detec_builder[i].xRotation = float(detection[5])
        detec_builder[i].radius = float(detection[6])

        nb_ids = len(detection) - _detection_dp_fields_before_ids
        decodedId = detec_builder[i].init('decodedId', nb_ids)
        for j, bit in enumerate(
                detections[i, _detection_dp_fields_before_ids:]):
            decodedId[j] = int(round(255*bit))


def nb_parameters(nb_bits):
    """Returns the number of parameter of the detections."""
    return _detection_dp_fields_before_ids + nb_bits


@pytest.mark.slow
def test_benchmark_add(benchmark, example_experiment_repo):
    repo, begin, end = example_experiment_repo

    ts = random.randint(begin, end)
    duration = 500 + random.randint(0, 250)
    cam_id = 0
    nb_bits = 12
    frame_container = build_frame_container(ts, ts + duration, cam_id)
    frames = frame_container.init('frames', 1024)
    frame_ts = ts
    for i, frame in enumerate(frames):
        nb_detections = random.randint(75, 150)
        detections = np.random.uniform(
            0, 1, (nb_detections, nb_parameters(nb_bits)))
        build_frame(frame, frame_ts, detections, frame_idx=i)

    def add():
        ts = random.randint(begin, end)
        duration = 500 + random.randint(0, 250)
        frame_container.fromTimestamp = ts
        frame_container.toTimestamp = ts + duration
        repo.add(frame_container)
    benchmark(add)
