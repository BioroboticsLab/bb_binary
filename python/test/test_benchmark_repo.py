
from conftest import fill_repository
from bb_binary import build_frame_container, parse_video_fname, Frame, \
    Repository, convert_detections_to_numpy, build_frame, nb_parameters

import time
import numpy as np
import pytest
import os
import random
import tempfile
import shutil


@pytest.fixture
def example_experiment_repo(request):
    tmpdir = tempfile.mkdtemp(prefix=os.path.dirname(__file__) + "_tmpdir_")
    repo = Repository(tmpdir, breadth_exponents=[2]*4 + [3])
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
    for frame in frames:
        nb_detections = random.randint(75, 150)
        detections = np.random.uniform(
            0, 1, (nb_detections, nb_parameters(nb_bits)))
        build_frame(frame, frame_ts, detections)

    def add():
        ts = random.randint(begin, end)
        duration = 500 + random.randint(0, 250)
        frame_container.fromTimestamp = ts
        frame_container.toTimestamp = ts + duration
        repo.add(frame_container)
    benchmark(add)
