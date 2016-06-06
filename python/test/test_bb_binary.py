
from conftest import fill_repository
from bb_binary import build_frame_container, parse_video_fname, Frame, \
    Repository, convert_detections_to_numpy, build_frame

import time
import datetime
import numpy as np
import pytest
import os


def test_bbb_is_loaded():
    frame = Frame.new_message()
    assert hasattr(frame, 'timestamp')


def test_bbb_frame_from_detections():
    frame = Frame.new_message()
    timestamp = time.time()
    detections = np.array([
        [0, 24, 43, 243, 234, 1, 0.1, 0.4, 0.1, 0.3] + [0.9] * 12,
        [1, 324, 543, 243, 234, 1,  0.1, 0.4, 0.1, 0.3] + [0.2] * 12,
    ])

    build_frame(frame, timestamp, detections)
    capnp_detections = frame.detectionsUnion.detectionsDP

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
            np.array(list(capnp_detections[i].decodedId)) / 255,
            detections[i, 9:],
            atol=0.5/255,
        )


def test_bbb_convert_detections_to_numpy():
    frame = Frame.new_message()
    frame.detectionsUnion.init('detectionsDP', 1)
    detection = frame.detectionsUnion.detectionsDP[0]
    detection.idx = 0
    detection.xpos = 344
    detection.ypos = 5498
    detection.zRotation = 0.24
    detection.yRotation = 0.1
    detection.xRotation = -0.14
    detection.radius = 23
    nb_bits = 12
    bits = detection.init('decodedId', nb_bits)
    bit_value = 24
    for i in range(nb_bits):
        bits[i] = bit_value

    arr = convert_detections_to_numpy(frame)
    assert arr[0, 0] == detection.idx
    assert arr[0, 1] == detection.xpos
    assert arr[0, 2] == detection.ypos
    assert arr[0, 3] == detection.yposHive
    assert arr[0, 4] == detection.yposHive
    assert arr[0, 5] == detection.zRotation
    assert arr[0, 6] == detection.yRotation
    assert arr[0, 7] == detection.xRotation
    assert arr[0, 8] == detection.radius
    assert np.allclose(arr[0, 9:], np.array([bit_value / 255] * nb_bits),
                       atol=0.5/255)


def test_bbb_repo_save_json(tmpdir):
    repo = Repository(str(tmpdir), 0)
    assert tmpdir.join(Repository._CONFIG_FNAME).exists()

    loaded_repo = Repository.load(str(tmpdir))
    assert repo == loaded_repo


def test_bbb_repo_path_for_ts(tmpdir):
    repo = Repository(str(tmpdir), breadth_exponents=[2]*4)
    path = repo._path_for_ts(3000)
    assert path == '00/00/30'

    repo = Repository(str(tmpdir), breadth_exponents=[3]*4)

    path = repo._path_for_ts(58000)
    assert path == '000/000/058'

    path = repo._path_for_ts(99014358000)
    assert path == '099/014/358'

    path = repo._path_for_ts(14358000)
    assert path == '000/014/358'

    now = int(time.time())
    path = repo._path_for_ts(now)
    now = str(now)
    path = path.split(os.path.sep)
    assert path[-1] == now[-6:-3]
    assert path[-2] == now[-9:-6]

    path = repo._path_for_ts(repo.max_ts - 1)
    assert path == '999/999/999'

    with pytest.raises(AssertionError):
        repo._path_for_ts(repo.max_ts)


def test_bbb_repo_get_ts_from_path(tmpdir):
    repo = Repository(str(tmpdir), breadth_exponents=[2]*4)

    path = '00/10/01'
    assert repo._get_timestamp_from_path(path) == 100100

    path = '50/10/99'
    assert repo._get_timestamp_from_path(path) == 50109900

    # test inverse to path_for_ts
    ts = 3000
    path = repo._path_for_ts(ts)
    assert path == '00/00/30'
    assert repo._get_timestamp_from_path(path) == ts


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
    repo = Repository(str(tmpdir), breadth_exponents=[3]*3 + [3])
    span = 500
    begin_end_cam_id = [(ts, ts + span, 0) for ts in range(0, 100000, span)]

    fill_repository(repo, begin_end_cam_id)

    find_and_assert_begin(repo, 0, expect_begin=0)
    find_and_assert_begin(repo, 50, expect_begin=0)
    find_and_assert_begin(repo, 500, expect_begin=500)
    find_and_assert_begin(repo, 650, expect_begin=500)
    find_and_assert_begin(repo, 999, expect_begin=500)
    find_and_assert_begin(repo, 1000, expect_begin=1000)

    with pytest.raises(AssertionError):
        find_and_assert_begin(repo, 50000, 50)


def test_bbb_repo_iter_fnames(tmpdir):
    repo = Repository(str(tmpdir.join('empty')),
                      breadth_exponents=[3]*3 + [3])
    assert list(repo.iter_fnames()) == []

    repo = Repository(str(tmpdir.join('2_files_and_1_symlink_per_directory')),
                      breadth_exponents=[3]*3 + [3])
    span = 500
    begin_end_cam_id = [(ts, ts + span + 100, 0) for ts in range(0, 10000, span)]

    fill_repository(repo, begin_end_cam_id)

    fnames = [os.path.basename(f) for f in repo.iter_fnames()]
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb')) for p in begin_end_cam_id]
    assert fnames == expected_fnames

    repo = Repository(str(tmpdir.join('missing_directories')),
                      breadth_exponents=[3]*3 + [3])
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

    repo = Repository(str(tmpdir.join('complex_from_to')),
                      breadth_exponents=[3]*3 + [3])
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
    slice_begin_end_cam_id = list(filter(lambda p: begin <= p[0] <= end,
                                         begin_end_cam_id))
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb'))
                       for p in slice_begin_end_cam_id]
    assert fbasenames == expected_fnames

    repo = Repository(str(tmpdir.join('complex_from_to_and_cam')),
                      breadth_exponents=[3]*3 + [3])
    span = 1500
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
    slice_begin_end_cam_id = list(filter(
        lambda p: begin <= p[0] <= end and p[2] == cam,
        begin_end_cam_id))
    expected_fnames = [os.path.basename(
        repo._get_filename(*p, extension='bbb'))
                       for p in slice_begin_end_cam_id]
    assert fbasenames == expected_fnames


def test_bbb_repo_find_multiple_file_per_timestamp(tmpdir):
    repo = Repository(str(tmpdir), breadth_exponents=[3]*3 + [3])
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
    repo = Repository(str(tmpdir), breadth_exponents=[3]*3 + [3])
    fname, symlinks = repo._create_file_and_symlinks(0, 2000, 0, 'bbb')
    with open(fname, 'w') as f:
        f.write("hello world!")
    assert len(symlinks) == 2
    assert os.path.exists(symlinks[0])
    for symlink in symlinks:
        with open(symlink) as f:
            assert f.read() == "hello world!"

    _, symlinks = repo._create_file_and_symlinks(1045, 4567, 0, 'bbb')
    assert len(symlinks) == 3

    _, symlinks = repo._create_file_and_symlinks(1045, 1999, 0, 'bbb')
    assert len(symlinks) == 0


def test_bbb_repo_add_frame_container(tmpdir):
    repo = Repository(str(tmpdir), breadth_exponents=[3]*3 + [3])
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
    repo = Repository(str(tmpdir), breadth_exponents=[3]*3 + [3])
    cam_id = 1
    fc = build_frame_container(1000, 5000, cam_id)

    repo.add(fc)
    open_fc = repo.open(2000, 1)
    assert fc.fromTimestamp == open_fc.fromTimestamp
    assert fc.toTimestamp == open_fc.toTimestamp


def test_parse_video_fname():
    fname = "Cam_1_20160501160208_958365_TO_Cam_1_20160501160748_811495.avi"
    camIdx, begin, end = parse_video_fname(fname, format='readable')
    begin_dt = datetime.datetime.fromtimestamp(begin)
    assert camIdx == 1
    assert begin_dt.year == 2016

    fname = "Cam_1_20160501160208_0_TO_Cam_1_20160501160748_0.bb"
    camIdx, begin, end = parse_video_fname(fname, format='readable')
    begin_dt = datetime.datetime.fromtimestamp(begin)
    assert camIdx == 1
    assert begin_dt.year == 2016
    assert begin_dt.month == 5

    fname = "Cam_1_20160501160208_0_TO_Cam_1_20160501160748_0.bb"
    camIdx, begin, end = parse_video_fname(fname, format='timestamp')
    assert begin == 20160501160208
