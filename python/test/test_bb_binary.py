
import bb_binary as bbb
import time
import numpy as np
import pytest
import os

def test_bbb_is_loaded():
    frame = bbb.Frame.new_message()
    assert hasattr(frame, 'timestamp')


def test_bbb_frame_from_detections():
    frame = bbb.Frame.new_message()
    timestamp = time.time()
    detections = np.array([
        [0, 24, 43, 0.1, 0.4, 0.1, 0.3] + [0.9] * 12,
        [1, 324, 543, 0.1, 0.4, 0.1, 0.3] + [0.2] * 12,
    ])

    bbb.build_frame(frame, timestamp, detections)
    capnp_detections = frame.detectionsUnion.detectionsDP

    for i in range(len(detections)):
        assert capnp_detections[i].xpos == detections[i, 1]
        assert capnp_detections[i].ypos == detections[i, 2]
        assert np.allclose(capnp_detections[i].zRotation, detections[i, 3])
        assert np.allclose(capnp_detections[i].yRotation, detections[i, 4])
        assert np.allclose(capnp_detections[i].xRotation, detections[i, 5])
        assert np.allclose(capnp_detections[i].radius, detections[i, 6])

        assert np.allclose(
            np.array(list(capnp_detections[i].decodedId)) / 255,
            detections[i, 7:],
            atol=0.5/255,
        )


def test_bbb_convert_detections_to_numpy():
    frame = bbb.Frame.new_message()
    frame.detectionsUnion.init('detectionsDP', 1)
    detection = frame.detectionsUnion.detectionsDP[0]
    detection.tagIdx = 0
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

    arr = bbb.convert_detections_to_numpy(frame)
    assert arr[0, 0] == detection.tagIdx
    assert arr[0, 1] == detection.xpos
    assert arr[0, 2] == detection.ypos
    assert arr[0, 3] == detection.zRotation
    assert arr[0, 4] == detection.yRotation
    assert arr[0, 5] == detection.xRotation
    assert arr[0, 6] == detection.radius
    assert np.allclose(arr[0, 7:], np.array([bit_value / 255] * nb_bits),
                       atol=0.5/255)


def test_bbb_repo_save_json(tmpdir):
    repo = bbb.Repository(str(tmpdir), 0)
    assert tmpdir.join('bbb_repo.json').exists()

    loaded_repo = bbb.Repository.load(str(tmpdir))
    assert repo == loaded_repo



def test_bbb_repo_directory_slices_for_ts(tmpdir):
    repo = bbb.Repository(str(tmpdir), ts_begin=0, directory_depths=[2]*4,
                          )
    dirs = list(repo.directory_slices_for_ts(3000))
    assert dirs == ['00', '00', '30']

    repo = bbb.Repository(str(tmpdir), ts_begin=1101, directory_depths=[3]*4)

    dirs = list(repo.directory_slices_for_ts(58000))
    assert dirs == ['000', '000', '058']

    dirs = list(repo.directory_slices_for_ts(14358000))
    assert dirs == ['000', '014', '358']

    now = int(time.time())
    dirs = list(repo.directory_slices_for_ts(now))
    now = str(now)
    assert dirs[-1] == now[-6:-3]
    assert dirs[-2] == now[-9:-6]

    dirs = list(repo.directory_slices_for_ts(repo.max_ts - 1))
    assert dirs == ['999', '999', '999']

    with pytest.raises(AssertionError):
        repo.directory_slices_for_ts(repo.max_ts)


def test_bbb_repo_get_ts_for_directory_slices(tmpdir):
    repo = bbb.Repository(str(tmpdir), ts_begin=0, directory_depths=[2]*4)

    dir_slices = ['00', '10', '01']
    assert repo.get_timestamp_for_directory_slice(dir_slices) == 100100

    # test inverse to directory_slices_for_ts
    ts = 3000
    dir_slices = list(repo.directory_slices_for_ts(ts))
    assert repo.get_timestamp_for_directory_slice(dir_slices)


def fill_repository(repo, begin_end_cam_ids):
    for begin, end, cam_ids in begin_end_cam_ids:
        params = begin, end, cam_ids, 'bbb'
        fname = repo.get_file_name(*params)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w+') as f:
            f.write(str(begin))
        repo._create_file_and_symlinks(*params)


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
    repo = bbb.Repository(str(tmpdir), directory_depths=[3]*3 + [3])
    span = 500
    begin_end_cam_ids = [(ts, ts + span, [0, 1])
                         for ts in range(0, 100000, span)]

    fill_repository(repo, begin_end_cam_ids)

    find_and_assert_begin(repo, 0, expect_begin=0)
    find_and_assert_begin(repo, 50, expect_begin=0)
    find_and_assert_begin(repo, 500, expect_begin=500)
    find_and_assert_begin(repo, 650, expect_begin=500)
    find_and_assert_begin(repo, 999, expect_begin=500)
    find_and_assert_begin(repo, 1000, expect_begin=1000)

    with pytest.raises(AssertionError):
        find_and_assert_begin(repo, 50000, 50)


def test_bbb_repo_find_multiple_file_per_timestamp(tmpdir):
    repo = bbb.Repository(str(tmpdir), directory_depths=[3]*3 + [3])
    span = 500
    begin = 1000
    end = 100000
    begin_end_cam_ids = [(ts, ts + span, [0, 1])
                         for ts in range(begin, end, span)]
    begin_end_cam_ids += [(ts, ts + span, [1, 2])
                          for ts in range(begin, end, span)]

    fill_repository(repo, begin_end_cam_ids)

    find_and_assert_begin(repo, 0, expect_begin=0, nb_files_found=0)
    find_and_assert_begin(repo, 1050, expect_begin=1000, nb_files_found=2)
    find_and_assert_begin(repo, 1499, expect_begin=1000, nb_files_found=2)
    find_and_assert_begin(repo, 1500, expect_begin=1500, nb_files_found=2)


def test_bbb_create_symlinks(tmpdir):
    repo = bbb.Repository(str(tmpdir), directory_depths=[3]*3 + [3])
    file_params = (0, 2000, [0, 1], 'bbb')
    symlinks = repo._create_file_and_symlinks(*file_params)
    fname = repo.get_file_name(*file_params)
    with open(fname, 'w') as f:
        f.write("hello world!")
    assert len(symlinks) == 2
    assert os.path.exists(symlinks[0])
    for symlink in symlinks:
        with open(symlink) as f:
            assert f.read() == "hello world!"

    file_params = (1045, 4567, [0, 1], 'bbb')
    symlinks = repo._create_file_and_symlinks(*file_params)
    assert len(symlinks) == 3

    file_params = (1045, 1999, [0, 1], 'bbb')
    symlinks = repo._create_file_and_symlinks(*file_params)
    assert len(symlinks) == 0

