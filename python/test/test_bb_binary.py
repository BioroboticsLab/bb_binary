# -*- coding: utf-8 -*-
from conftest import fill_repository
from bb_binary import build_frame_container, parse_video_fname, Frame, \
    Repository,  build_frame, dt_to_str, convert_frame_to_numpy, \
    _convert_detections_to_numpy, _convert_frame_to_numpy, get_detections

import time
from datetime import datetime
import pytz
import numpy as np
import os
import pytest
import math


def test_bbb_is_loaded():
    frame = Frame.new_message()
    assert hasattr(frame, 'timestamp')


def test_bbb_relative_path():
    repo = Repository("test_repo")
    assert os.path.isabs(repo.root_dir)


def test_dt_to_str():
    dt = datetime(2015, 8, 15, 12, 0, 40, 333967, tzinfo=pytz.utc)
    assert dt_to_str(dt) == "20150815T120040.333Z"


def test_bbb_frame_from_detections():
    frame = Frame.new_message()
    timestamp = time.time()
    frame_idx = 10
    detections = np.array([
        [0, 24, 43, 243, 234, 1, 0.1, 0.4, 0.1, 0.3] + [0.9] * 12,
        [1, 324, 543, 243, 234, 1,  0.1, 0.4, 0.1, 0.3] + [0.2] * 12,
    ])

    build_frame(frame, timestamp, detections, frame_idx)
    capnp_detections = frame.detectionsUnion.detectionsDP

    assert frame.frameIdx == frame_idx
    assert frame.dataSourceIdx == 0
    for i in range(len(detections)):
        assert capnp_detections[i].xpos == detections[i, 1]
        assert capnp_detections[i].ypos == detections[i, 2]
        assert capnp_detections[i].xposHive == detections[i, 3]
        assert capnp_detections[i].yposHive == detections[i, 4]
        assert np.allclose(capnp_detections[i].zRotation, detections[i, 5])
        assert np.allclose(capnp_detections[i].yRotation, detections[i, 6])
        assert np.allclose(capnp_detections[i].xRotation, detections[i, 7])
        assert np.allclose(capnp_detections[i].radius, detections[i, 8])
        assert np.allclose(
            np.array(list(capnp_detections[i].decodedId)) / 255.,
            detections[i, 9:],
            atol=0.5/255.,
        )


@pytest.fixture
def frame_data():
    """Frame without detections."""
    frame = Frame.new_message()
    frame.id = 1
    frame.dataSourceIdx = 1
    frame.timedelta = 150
    frame.timestamp = 1465290180

    return frame


@pytest.fixture
def frame_dp_data(frame_data):
    """Frame with detections in new pipeline format."""
    frame = frame_data.copy()
    frame.detectionsUnion.init('detectionsDP', 3)
    for i in range(0, 3):
        detection = frame.detectionsUnion.detectionsDP[i]
        detection.idx = i
        detection.xpos = 344 + 10 * i
        detection.ypos = 5498 + 10 * i
        detection.zRotation = 0.24 + 0.1 * i
        detection.yRotation = 0.1 + 0.1 * i
        detection.xRotation = -0.14 - 0.1 * i
        detection.radius = 22 + i
        nb_bits = 12
        bits = detection.init('decodedId', nb_bits)
        bit_value = nb_bits * (1+i)
        for i in range(nb_bits):
            bits[i] = bit_value

    return frame


@pytest.fixture
def frame_cvp_data(frame_data):
    """Frame with detections in old pipeline format."""
    frame = frame_data.copy()
    frame.detectionsUnion.init('detectionsCVP', 3)
    for i in range(0, 3):
        detection = frame.detectionsUnion.detectionsCVP[i]
        detection.idx = 0
        detection.candidateIdx = 0
        detection.gridIdx = i
        detection.xpos = 344
        detection.ypos = 5498
        detection.zRotation = 0.24
        detection.yRotation = 0.1
        detection.xRotation = -0.14
        detection.decodedId = 2015 + i

    return frame


@pytest.fixture
def frame_truth_data(frame_data):
    """Frame with truth data."""
    frame = frame_data.copy()
    frame.detectionsUnion.init('detectionsTruth', 4)
    readability = ('completely', 'partially', 'none', 'unknown')
    for i in range(0, len(readability)):
        detection = frame.detectionsUnion.detectionsTruth[i]
        detection.idx = i
        detection.xpos = 344
        detection.ypos = 5498
        detection.decodedId = 2015 + i
        detection.readability = readability[i]

    return frame


def test_bbb_convert_detections_to_numpy(frame_dp_data):
    """Detections are correctly converted to np array and frame is ignored."""
    frame = frame_dp_data
    expected_keys = ('idx', 'xpos', 'ypos', 'xRotation', 'yRotation',
                     'zRotation', 'radius', 'decodedId')

    detections = frame.detectionsUnion.detectionsDP
    arr = _convert_detections_to_numpy(detections, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_to_numpy(frame_data):
    """Frame is correctly converted to np array and detections are ignored."""
    frame = frame_data

    expected_keys = ('frameId', 'timedelta', 'timestamp')

    arr = _convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_only_frame_to_numpy(frame_dp_data):
    """Frame is correctly converted to np array and detections are ignored."""
    frame = frame_dp_data

    expected_keys = ('frameId', 'timedelta', 'timestamp')

    arr = _convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_and_detections_dp_to_numpy(frame_dp_data):
    """Frame and detections (dp) are correctly converted to np array."""
    frame = frame_dp_data

    expected_keys = ('frameId', 'timedelta', 'timestamp',
                     'idx', 'xpos', 'ypos', 'xRotation', 'yRotation',
                     'zRotation', 'radius', 'decodedId', 'detectionsUnion')

    arr = convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_and_detections_cvp_to_numpy(frame_cvp_data):
    """Frame and detections (cvp) are correctly converted to np array."""
    frame = frame_cvp_data

    expected_keys = ('frameId', 'timedelta', 'timestamp',
                     'idx', 'xpos', 'ypos', 'xRotation', 'yRotation',
                     'zRotation', 'candidateIdx', 'gridIdx', 'decodedId',
                     'detectionsUnion')

    arr = convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_and_detections_truth_to_numpy(frame_truth_data):
    """Frame and detections (cvp) are correctly converted to np array."""
    frame = frame_truth_data

    expected_keys = ('frameId', 'timedelta', 'timestamp',
                     'idx', 'xpos', 'ypos', 'decodedId', 'readability',
                     'detectionsUnion')

    arr = convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def bbb_check_frame_data(frame, arr, expected_keys):
    """Helper to compare frame data to numpy array."""
    # check if we have all the expected keys in the array (and only these)
    expected_keys = set(expected_keys)
    expected_keys.discard('detectionsUnion')
    assert expected_keys == set(arr.dtype.names)
    assert len(expected_keys) == len(arr.dtype.names)

    detection_string_fields = ('readability')
    detections = get_detections(frame)
    for i, detection in enumerate(detections):
        # check if the values are as expected
        for key in expected_keys:
            if key == 'decodedId' and \
               frame.detectionsUnion.which() == 'detectionsDP':
                assert np.allclose(arr[key][i],
                                   np.array([detection.decodedId[0] / 255.] *
                                            len(detection.decodedId)),
                                   atol=0.5/255.)
            elif key == 'frameId':
                assert np.all(arr[key] == getattr(frame, 'id'))
            elif hasattr(frame, key):
                # all detections are from the same frame
                # so we expect the whole column to have the same value.
                assert np.all(arr[key] == getattr(frame, key))
            elif key in detection_string_fields:
                assert arr[key][i].decode('UTF-8') == getattr(detection, key)
            else:
                assert arr[key][i] == getattr(detection, key)


def test_bbb_get_detections(frame_data, frame_dp_data, frame_cvp_data):
    """Extracts correct detections from old and new pipeline data."""
    diffKey = 'candidateIdx'
    detections = get_detections(frame_dp_data)
    assert not hasattr(detections[0], diffKey)

    detections = get_detections(frame_cvp_data)
    assert hasattr(detections[0], diffKey)

    # Important note: the default value for detectionsUnion is detectionsCVP
    assert frame_data.detectionsUnion.which() == 'detectionsCVP'


def test_bbb_repo_save_json(tmpdir):
    repo = Repository(str(tmpdir), 0)
    assert tmpdir.join(Repository._CONFIG_FNAME).exists()

    loaded_repo = Repository.load(str(tmpdir))
    assert repo == loaded_repo


def test_bbb_repo_path_for_ts(tmpdir):
    repo = Repository(str(tmpdir))
    path = repo._path_for_dt(datetime(1970, 1, 20, 8, 25))
    assert path == '1970/01/20/08/20'

    repo = Repository(str(tmpdir))

    path = repo._path_for_dt(datetime(2012, 2, 29, 8, 55))
    assert path == '2012/02/29/08/40'

    now = datetime.now(pytz.utc)
    print(now.utcoffset())
    path = repo._path_for_dt(now)
    expected_minutes = int(math.floor(now.minute / repo.minute_step) * repo.minute_step)
    expected_dt = now.replace(minute=expected_minutes, second=0, microsecond=0)
    print(expected_dt.utcoffset())
    assert repo._get_time_from_path(path) == expected_dt


def test_bbb_repo_get_ts_from_path(tmpdir):
    repo = Repository(str(tmpdir))

    path = '1800/10/01/00/00'
    assert repo._get_time_from_path(path) == datetime(1800, 10, 1, 0, 0, tzinfo=pytz.utc)

    path = '2017/10/15/23/40'
    assert repo._get_time_from_path(path) == datetime(2017, 10, 15, 23, 40, tzinfo=pytz.utc)

    # test inverse to path_for_ts
    dt = datetime(2017, 10, 15, 23, repo.minute_step, tzinfo=pytz.utc)
    path = repo._path_for_dt(dt)
    assert path == '2017/10/15/23/{:02d}'.format(repo.minute_step)
    assert repo._get_time_from_path(path) == dt


def find_and_assert_begin(repo, timestamp, expect_begin, nb_files_found=1):
    fnames = repo.find(timestamp)
    if type(expect_begin) == int:
        expect_begin = [expect_begin]
    assert type(expect_begin) in (tuple, list)
    expect_begin = list(map(str, expect_begin))

    assert len(fnames) == nb_files_found
    for fname in fnames:
        with open(fname) as f:
            assert f.read() in expect_begin


def test_bbb_repo_find_single_file_per_timestamp(tmpdir):
    repo = Repository(str(tmpdir))
    span = 60*10
    begin_end_cam_id = [(ts, ts + span, 0) for ts in range(0, 100000, span)]

    fill_repository(repo, begin_end_cam_id)

    assert repo.find(0)[0] == repo._get_filename(0, span, 0, 'bbb')
    assert repo.find(60*10)[0] == repo._get_filename(60*10, 60*10+span, 0, 'bbb')
    assert repo.find(1000000) == []


def test_bbb_repo_iter_fnames_empty(tmpdir):
    repo = Repository(str(tmpdir.join('empty')))
    assert list(repo.iter_fnames()) == []


def test_bbb_repo_iter_fnames_2_files_and_1_symlink_per_directory(tmpdir):
    repo = Repository(str(tmpdir.join('2_files_and_1_symlink_per_directory')))
    span = 500
    begin_end_cam_id = [(ts, ts + span + 100, 0) for ts in range(0, 10000, span)]

    fill_repository(repo, begin_end_cam_id)

    fnames = [os.path.basename(f) for f in repo.iter_fnames()]
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb')) for p in begin_end_cam_id]
    assert fnames == expected_fnames


def test_bbb_repo_iter_fnames_missing_directories(tmpdir):
    repo = Repository(str(tmpdir.join('missing_directories')))
    span = 1500
    begin_end_cam_id = [(ts, ts + span, 0)
                        for ts in range(0, 10000, span)]

    fill_repository(repo, begin_end_cam_id)
    fnames = list(repo.iter_fnames())
    for fname in fnames:
        assert os.path.isabs(fname)
    fbasenames = [os.path.basename(f) for f in fnames]
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb')) for p in begin_end_cam_id]
    assert fbasenames == expected_fnames


def test_bbb_repo_iter_fnames_from_to(tmpdir):
    repo = Repository(str(tmpdir.join('complex_from_to')))
    span = 1500
    begin_end_cam_id = [(ts, ts + span, 0)
                        for ts in range(0, 10000, span)]

    fill_repository(repo, begin_end_cam_id)
    begin = 2500
    end = 5000
    fnames = list(repo.iter_fnames(begin, end))
    for fname in fnames:
        assert os.path.isabs(fname)
    fbasenames = [os.path.basename(f) for f in fnames]
    slice_begin_end_cam_id = list(filter(lambda p: begin <= p[1] and p[0] < end,
                                         begin_end_cam_id))
    print(slice_begin_end_cam_id)
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb'))
                       for p in slice_begin_end_cam_id]
    print(expected_fnames)
    print(fbasenames)
    assert fbasenames == expected_fnames


def test_bbb_repo_iter_fnames_from_to_and_cam(tmpdir):
    repo = Repository(str(tmpdir.join('complex_from_to_and_cam')))
    span = 200
    begin_end_cam_id0 = [(ts, ts + span, 0) for ts in range(0, 10000, span)]
    begin_end_cam_id1 = [(ts, ts + span, 1) for ts in range(0, 10000, span)]

    begin_end_cam_id = begin_end_cam_id0 + begin_end_cam_id1

    fill_repository(repo, begin_end_cam_id)
    begin = 2500
    end = 5000
    cam = 0
    fnames = list(repo.iter_fnames(begin, end, cam))
    for fname in fnames:
        assert os.path.isabs(fname)
    fbasenames = [os.path.basename(f) for f in fnames]
    print(begin_end_cam_id)
    slice_begin_end_cam_id = list(filter(
        lambda p: begin <= p[1] and p[0] < end and p[2] == cam,
        begin_end_cam_id))
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb'))
                       for p in slice_begin_end_cam_id]
    assert fbasenames == expected_fnames


def test_bbb_repo_find_multiple_file_per_timestamp(tmpdir):
    repo = Repository(str(tmpdir))
    span = 500
    begin = 1000
    end = 100000
    begin_end_cam_id = [(ts, ts + span, 0)
                        for ts in range(begin, end, span)]
    begin_end_cam_id += [(ts, ts + span, 1)
                         for ts in range(begin, end, span)]

    fill_repository(repo, begin_end_cam_id)

    find_and_assert_begin(repo, 0, expect_begin=0, nb_files_found=0)
    find_and_assert_begin(repo, 1050, expect_begin=1000, nb_files_found=2)
    find_and_assert_begin(repo, 1499, expect_begin=1000, nb_files_found=2)
    find_and_assert_begin(repo, 1500, expect_begin=1500, nb_files_found=2)


def test_bbb_create_symlinks(tmpdir):
    repo = Repository(str(tmpdir))
    fname, symlinks = repo._create_file_and_symlinks(0, 60*repo.minute_step*2 + 10, 0, 'bbb')
    with open(fname, 'w') as f:
        f.write("hello world!")
    assert len(symlinks) == 2
    assert os.path.exists(symlinks[0])
    for symlink in symlinks:
        with open(symlink) as f:
            assert f.read() == "hello world!"

    _, symlinks = repo._create_file_and_symlinks(1045, 1045 + 60*repo.minute_step*3 + 5, 0, 'bbb')
    assert len(symlinks) == 3

    _, symlinks = repo._create_file_and_symlinks(1045, 1045 + repo.minute_step // 2, 0, 'bbb')
    assert len(symlinks) == 0


def test_bbb_repo_add_frame_container(tmpdir):
    repo = Repository(str(tmpdir))
    cam_id = 1
    fc = build_frame_container(1000, 5000, 1)

    repo.add(fc)
    fnames = repo.find(1000)
    expected_fname = repo._get_filename(fc.fromTimestamp,
                                        fc.toTimestamp, cam_id, 'bbb')
    expected_fname = os.path.basename(expected_fname)
    assert os.path.basename(fnames[0]) == expected_fname

    fnames = repo.find(1500)
    assert os.path.basename(fnames[0]) == expected_fname

    fnames = repo.find(2500)
    assert os.path.basename(fnames[0]) == expected_fname


def test_bbb_repo_open_frame_container(tmpdir):
    repo = Repository(str(tmpdir))
    cam_id = 1
    fc = build_frame_container(1000, 5000, cam_id)

    repo.add(fc)
    open_fc = repo.open(2000, 1)
    assert fc.fromTimestamp == open_fc.fromTimestamp
    assert fc.toTimestamp == open_fc.toTimestamp


def test_parse_video_fname():
    fname = "Cam_1_20160501160208_958365_TO_Cam_1_20160501160748_811495.avi"
    camIdx, begin, end = parse_video_fname(fname, format='beesbook')
    begin_dt = datetime.fromtimestamp(begin)
    assert camIdx == 1
    assert begin_dt.year == 2016

    fname = "Cam_1_20160501160208_0_TO_Cam_1_20160501160748_0.bbb"
    camIdx, begin, end = parse_video_fname(fname, format='beesbook')
    begin_dt = datetime.fromtimestamp(begin)
    assert camIdx == 1
    assert begin_dt.year == 2016
    assert begin_dt.month == 5

    fname = "Cam_0_19700101T001000.000000Z--19700101T002000.000000Z.bbb"
    camIdx, begin, end = parse_video_fname(fname, format='iso')
    assert begin == datetime.fromtimestamp(10*60, tz=pytz.utc)
    assert end == datetime.fromtimestamp(20*60, tz=pytz.utc)
