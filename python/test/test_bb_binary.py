# -*- coding: utf-8 -*-
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytz
import pytest
from conftest import fill_repository
from bb_binary import build_frame_container, parse_fname, parse_video_fname, parse_image_fname, \
    Frame, Repository, dt_to_str, convert_frame_to_numpy, \
    _convert_detections_to_numpy, _convert_frame_to_numpy, get_detections, \
    build_frame_container_from_df, to_datetime, int_id_to_binary


def test_bbb_is_loaded():
    frame = Frame.new_message()
    assert hasattr(frame, 'timestamp')


def test_bbb_relative_path():
    repo = Repository("test_repo")
    assert os.path.isabs(repo.root_dir)


def test_dt_to_str():
    """Test conversion of datetime objects to string representation."""
    dt = datetime(2015, 8, 15, 12, 0, 40, 333967, tzinfo=pytz.utc)
    assert dt_to_str(dt) == "2015-08-15T12:00:40.333967Z"

    with pytest.raises(Exception, message="Got a datetime object not in UTC. Allways use UTC."):
        dt_to_str(dt.replace(tzinfo=None))


def test_to_datetime():
    """Test conversion of timestamps to datetime."""
    expected_dt = datetime(2015, 8, 15, 12, 0, 40, tzinfo=pytz.utc)

    # test with int
    dt = to_datetime((expected_dt - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds())
    assert dt == expected_dt

    # test with float
    expected_dt_float = expected_dt.replace(microsecond=333967)
    dt = to_datetime((expected_dt_float - datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds())
    assert dt == expected_dt_float

    # test with datetime object
    dt = to_datetime(expected_dt)
    assert dt == expected_dt

    # test with string
    with pytest.raises(TypeError):
        dt = to_datetime("2015-08-15T12:00:40.333967Z")


def test_int_id_to_binary():
    """Test conversion of integer id representation to binary array representation."""
    bit_arr = int_id_to_binary(8)
    assert np.all(bit_arr == np.array([0, 0, 0, 0, 0, 0,
                                       0, 0, 1, 0, 0, 0], dtype=np.uint8))

    bit_arr = int_id_to_binary(4095)
    assert np.all(bit_arr == np.array([1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1], dtype=np.uint8))

    with pytest.raises(Exception) as exception_information:  # to big value
        bit_arr = int_id_to_binary(8096)

    assert 'overflows' in str(exception_information.value)


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
    frame.detectionsUnion.init('detectionsDP', 4)
    for i in range(0, 4):
        detection = frame.detectionsUnion.detectionsDP[i]
        detection.idx = i
        detection.xpos = 344 + 10 * i
        detection.ypos = 5498 + 10 * i
        detection.zRotation = 0.24 + 0.1 * i
        detection.yRotation = 0.1 + 0.1 * i
        detection.xRotation = -0.14 - 0.1 * i
        detection.radius = 22 + i
        detection.localizerSaliency = 0.8765
        nb_bits = 12
        bits = detection.init('decodedId', nb_bits)
        bit_value = nb_bits * (1+i)
        for i in range(nb_bits):
            bits[i] = bit_value
        descriptor = detection.init('descriptor', nb_bits)
        for i in range(nb_bits):
            descriptor[i] = bit_value
    return frame


@pytest.fixture
def frame_cvp_data(frame_data):
    """Frame with detections in old pipeline format."""
    frame = frame_data.copy()
    frame.detectionsUnion.init('detectionsCVP', 4)
    for i in range(0, 4):
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


@pytest.fixture(params=['cvp', 'dp', 'truth'])
def frame_data_all(request, frame_cvp_data, frame_dp_data, frame_truth_data):
    """Fixture to test different union types in Frames."""
    return {'cvp': frame_cvp_data, 'dp': frame_dp_data, 'truth': frame_truth_data}[request.param]


def get_detection_keys(union_type):
    """Function holding union type specific detection keys."""
    keys = ['idx', 'xpos', 'ypos', 'xposHive', 'yposHive', 'decodedId', 'detectionsUnion']

    additional_keys = {
        'detectionsDP': ['xRotation', 'yRotation', 'zRotation',
                         'radius', 'localizerSaliency', 'descriptor'],
        'detectionsCVP': ['xRotation', 'yRotation', 'zRotation',
                          'gridIdx', 'candidateIdx', 'eScore', 'gScore', 'lScore'],
        'detectionsTruth': ['readability']
    }[union_type]

    keys.extend(additional_keys)
    return keys


def convert_readability(df):
    """Helper to convert readability column from binary to string."""
    df.readability = df.readability.apply(lambda x: x.decode('UTF-8'))
    return df


def make_df_from_np(np_arr, union_type):
    """Helper to convert a numpy array with detection data to a pandas dataframe."""
    if union_type == 'detectionsDP':
        detections = pd.DataFrame(np_arr[list(set(np_arr.dtype.fields.keys()) -
                                              set(['decodedId', 'descriptor']))])
        detections['decodedId'] = pd.Series([list(idList) for idList in np_arr['decodedId']])
        detections['descriptor'] = pd.Series([list(descList) for descList in np_arr['descriptor']])
    else:
        detections = pd.DataFrame(np_arr)
    if union_type == 'detectionsTruth':
        detections = convert_readability(detections)
    return detections


def test_bbb_frame_container_errors(frame_data_all):
    """Test minimal keys when building `FrameContainer`"""
    frame = frame_data_all
    expected_keys = ('timestamp', 'xpos', 'ypos', 'detectionsUnion')

    arr = convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)
    detections = pd.DataFrame(arr)
    with pytest.raises(AssertionError) as error_information:
        fc, offset = build_frame_container_from_df(detections, frame.detectionsUnion.which(), 0)
    assert 'decodedId' in str(error_information.value)


def test_bbb_frame_container_from_df(frame_data_all):
    """Data in DataFrame correctly converted to `FrameContainer`."""
    frame = frame_data_all
    union_type = frame.detectionsUnion.which()
    expected_keys = ['timestamp', ]
    expected_keys.extend(get_detection_keys(union_type))

    arr = convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)
    detections = make_df_from_np(arr, union_type)
    expected_detections = detections.copy()
    nDetections = len(getattr(frame.detectionsUnion, union_type))

    offset = 0
    # test minimal set
    df = detections.copy()
    fc, offset = build_frame_container_from_df(df, union_type, offset)
    assert offset == 1
    assert fc.id == 0
    assert fc.camId == 0
    assert fc.fromTimestamp == expected_detections.timestamp[0]
    assert fc.toTimestamp == expected_detections.timestamp[3]
    assert len(fc.frames) == 1

    frame0 = fc.frames[0]
    assert frame0.id == offset - 1
    assert frame0.frameIdx == 0
    assert frame0.timestamp == expected_detections.timestamp[0]
    assert len(getattr(frame0.detectionsUnion, union_type)) == 4

    arr = convert_frame_to_numpy(frame0, expected_keys)
    converted_detections = make_df_from_np(arr, union_type)
    assert_frame_equal(expected_detections, converted_detections)

    # test without readability
    df = detections.copy()
    if union_type == 'detectionsTruth':
        df.drop('readability', axis=1, inplace=True)
    fc, offset = build_frame_container_from_df(df, union_type, offset, frame_offset=offset)
    assert offset == 2  # test offset and id to test for fixed assignments
    assert fc.id == 1
    assert fc.camId == 1
    assert len(fc.frames) == 1

    frame0 = fc.frames[0]
    assert frame0.id == offset - 1  # test if offset is considered
    assert frame0.frameIdx == 0

    if union_type == 'detectionsTruth':
        for i in range(0, nDetections):  # test for default readability value
            frame0.detectionsUnion.detectionsTruth[i].readability == b'undefined'

    # test with datetime instead of unixtimestamp
    df = detections.copy()
    df.timestamp = df.timestamp.apply(lambda x: to_datetime(x))
    fc, offset = build_frame_container_from_df(df, union_type, offset, frame_offset=offset)
    assert offset == 3
    fc.fromTimestamp == expected_detections.timestamp[0]
    assert len(fc.frames) == 1

    # test with additional column for frames
    df = detections.copy()
    df['dataSourceIdx'] = 99
    fc, offset = build_frame_container_from_df(df, union_type, offset, frame_offset=offset)
    assert offset == 4
    assert len(fc.frames) == 1

    frame0 = fc.frames[0]
    assert frame0.dataSourceIdx == df.dataSourceIdx[0]

    # test with additional column for detections
    df = detections.copy()
    df['xposHive'] = range(0, nDetections)
    fc, offset = build_frame_container_from_df(df, union_type, offset, frame_offset=offset)
    assert offset == 5
    assert len(fc.frames) == 1

    converted_detections = getattr(fc.frames[0].detectionsUnion, union_type)
    for i in range(0, nDetections):
        assert converted_detections[i].xposHive == df.xposHive[i]

    # test with camId column
    df = detections.copy()
    df['camId'] = [offset] * df.shape[0]
    df.loc[0, 'camId'] = offset - 1
    fc, offset = build_frame_container_from_df(df, union_type, offset, frame_offset=offset)
    assert offset == 6
    assert len(fc.frames) == 1

    converted_detections = getattr(fc.frames[0].detectionsUnion, union_type)
    assert len(converted_detections) == nDetections - 1  # one detection removed!


def test_bbb_convert_detections_to_numpy(frame_data_all):
    """Detections are correctly converted to np array and frame is ignored."""
    frame = frame_data_all
    expected_keys = get_detection_keys(frame.detectionsUnion.which())

    detections = get_detections(frame)
    arr = _convert_detections_to_numpy(detections, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)

    # now compare behaviour of helper and parent function
    arr_frame = convert_frame_to_numpy(frame, keys=expected_keys)
    assert np.all(arr == arr_frame)


def test_bbb_convert_only_frame_to_numpy(frame_data_all):
    """Frame is correctly converted to np array and detections are ignored."""
    frame = frame_data_all

    expected_keys = ('frameId', 'timedelta', 'timestamp', 'detectionsUnion')

    arr = _convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)

    # now compare behaviour of helper and parent function
    arr_frame = convert_frame_to_numpy(frame, keys=expected_keys)
    assert np.all(arr == arr_frame)


def test_bbb_convert_frame_and_detections_to_numpy(frame_data_all):
    """Frame and detections are correctly converted to np array."""
    frame = frame_data_all

    expected_keys = ['frameId', 'timedelta', 'timestamp']
    expected_keys.extend(get_detection_keys(frame.detectionsUnion.which()))

    arr = convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_with_additional_cols_to_numpy(frame_data_all):
    """Frame with additional columns is correctly converted to np array."""
    frame = frame_data_all

    expected_keys = ('frameId', 'timedelta', 'timestamp', 'decodedId', 'detectionsUnion')

    # one col, single value for whole columns
    arr = convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': 2})
    assert 'camId' in arr.dtype.names
    assert np.all(arr['camId'] == 2)

    # two cols, single value
    arr = convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': 2, 'second': 3})
    assert 'camId' in arr.dtype.names
    assert 'second' in arr.dtype.names
    assert np.all(arr['camId'] == 2)
    assert np.all(arr['second'] == 3)

    # list for whole column
    arr = convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': range(0, 4)})
    assert 'camId' in arr.dtype.names
    assert np.all(arr['camId'] == np.array(range(0, 4)))

    # existing column
    with pytest.raises(AssertionError):
        arr = convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'frameId': 9})


def test_bbb_convert_frame_default_keys_to_numpy(frame_data_all):
    """Convert frame to np arrays without explicitly requesting certain keys."""
    frame = frame_data_all

    frame_keys = set(['frameId', 'dataSourceIdx', 'frameIdx', 'timestamp', 'timedelta'])
    detection_keys = set(get_detection_keys(frame.detectionsUnion.which()))
    expected_keys = frame_keys | detection_keys | set(['camId'])

    # extract frame without explicitly asking for keys
    arr = convert_frame_to_numpy(frame, add_cols={'camId': 0})
    bbb_check_frame_data(frame, arr, expected_keys)


def bbb_check_frame_data(frame, arr, expected_keys):
    """Helper to compare frame data to numpy array."""
    # check if we have all the expected keys in the array (and only these)
    expected_keys = set(expected_keys) - set(['detectionsUnion'])
    assert expected_keys == set(arr.dtype.names)
    assert len(expected_keys) == len(arr.dtype.names)

    detection_string_fields = ('readability')
    detections = get_detections(frame)
    for i, detection in enumerate(detections):
        # check if the values are as expected
        for key in expected_keys:
            if key == 'camId':
                continue
            elif key == 'decodedId' and frame.detectionsUnion.which() == 'detectionsDP':
                assert np.allclose(arr[key][i],
                                   np.array([detection.decodedId[0] / 255.] *
                                            len(detection.decodedId)),
                                   atol=0.5/255.)
            elif key == 'descriptor':
                assert np.all(arr[key][i] == getattr(detection, key))
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


def test_bbb_get_detections(frame_data, frame_dp_data, frame_cvp_data, frame_truth_data):
    """Extracts correct detections from old and new pipeline data."""
    diffKey = 'candidateIdx'
    detections = get_detections(frame_dp_data)
    assert not hasattr(detections[0], diffKey)

    detections = get_detections(frame_cvp_data)
    assert hasattr(detections[0], diffKey)

    detections = get_detections(frame_truth_data)
    assert hasattr(detections[0], 'readability')

    # Important note: the default value for detectionsUnion is detectionsCVP
    assert frame_data.detectionsUnion.which() == 'detectionsCVP'


def test_bbb_repo_save_json(tmpdir):
    repo = Repository(str(tmpdir), 24)
    assert tmpdir.join(Repository._CONFIG_FNAME).exists()

    loaded_repo = Repository(str(tmpdir))
    assert loaded_repo.minute_step == repo.minute_step


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


def test_bbb_iter_frames_from_to(tmpdir):
    """Tests that only frames in given range are iterated."""
    repo = Repository(str(tmpdir.join('frames_from_to')))
    repo_start = 0
    nFC = 10
    span = 1000
    nFrames = nFC * span
    repo_end = repo_start + nFrames
    begin_end_cam_id = [(ts, ts + span, 0)
                        for ts in range(repo_start, repo_end, span)]
    for begin, end, cam_id in begin_end_cam_id:
        fc = build_frame_container(begin, end, cam_id)
        fc.init('frames', span)
        for i, tstamp in enumerate(range(begin, end)):
            frame = fc.frames[i]
            frame.id = tstamp
            frame.timestamp = tstamp
        repo.add(fc)

    def check_tstamp_invariant(begin, end):
        """Helper to check if begin <= tstamp < end is true for all frames."""
        for frame, fc in repo.iter_frames(begin, end):
            # frame container invariant
            assert begin < fc.toTimestamp
            assert fc.fromTimestamp < end
            # frame invariant
            assert begin <= frame.timestamp < end

    # repo_start < start < end < repo_end
    check_tstamp_invariant(repo_start + 10, repo_end - 10)
    # start < repo_start < end < repo_end
    check_tstamp_invariant(repo_start - 10, repo_end - 10)
    # start < end < repo_start < repo_end
    check_tstamp_invariant(repo_start - 20, repo_start - 10)
    # repo_start < start < repo_end < end
    check_tstamp_invariant(repo_start + 10, repo_end + 10)
    # repo_start < repo_end < start < end
    check_tstamp_invariant(repo_end + 10, repo_end + 20)

    # check whole length
    all_frames = [f for f, _ in repo.iter_frames()]
    assert len(all_frames) == nFrames
    # check with begin = None
    skip_end = [f for f, _ in repo.iter_frames(end=repo_end - span)]
    assert len(skip_end) == nFrames - span
    # check with end = None
    skip_start = [f for f, _ in repo.iter_frames(begin=span)]
    assert len(skip_start) == nFrames - span


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
    expected_fnames = [
        os.path.basename(repo._get_filename(*p, extension='bbb'))
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
    expected_fnames = [
        os.path.basename(repo._get_filename(*p, extension='bbb'))
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


@pytest.fixture(params=['iso', 'beesbook', 'auto_iso', 'auto_bb', 'arbitrary'])
def image(request):
    """Fixture to test extraction of information on different image filenames."""
    name_beesbook = 'Cam_0_20140805151756_200.jpeg'
    name_iso = 'Cam_0_2014-08-05T13:17:56,000200Z.jpeg'
    expected_dt = datetime(2014, 8, 5, 13, 17, 56, 200, tzinfo=pytz.utc)
    expected_cam = 0
    data = {'dt': expected_dt, 'cam': expected_cam, 'format': 'beesbook', 'name': name_beesbook}
    if 'iso' in request.param:
        data['format'] = 'iso'
        data['name'] = name_iso
    elif 'beesbook' not in request.param:
        data['format'] = 'arbitrary'

    if 'auto' in request.param:
        data['format'] = 'auto'
    return data


def test_parse_fname_images(image):
    """Tests the extraction of camera, date and time information from filenames."""
    camIdx, begin, end = parse_fname(image['name'])
    assert camIdx == image['cam']
    assert begin == image['dt']
    assert begin == end


def test_parse_image_fname(image):
    """Tests the extraction of camera, date and time information from image filenames."""
    if image['format'] == 'arbitrary':
        with pytest.raises(Exception):
            camIdx, ts = parse_image_fname(image['name'], format=image['format'])
        return

    camIdx, ts = parse_image_fname(image['name'], format=image['format'])
    assert camIdx == image['cam']
    assert ts == image['dt']


@pytest.fixture(params=['arbitrary', 'beesbook', 'beesbook_one_underscore',
                        'beesbook_two_underscores', 'auto_beesbook',
                        'iso', 'iso_hyphen', 'auto_iso'])
def video(request):
    """Fixture to test extraction of information on different video filenames."""
    name = "Cam_1_20160501160208_0_TO_Cam_1_20160501160748_0.bbb"
    dt_begin = datetime(2016, 5, 1, 14, 2, 8, tzinfo=pytz.utc)
    dt_end = datetime(2016, 5, 1, 14, 7, 48, tzinfo=pytz.utc)
    cam = 1
    vformat = 'beesbook'
    if 'beesbook_one_underscore' in request.param:
        name = "Cam_1_20160501160208_958365_TO_Cam_1_20160501160748_811495.avi"
        dt_begin = dt_begin.replace(microsecond=958365)
        dt_end = dt_end.replace(microsecond=811495)
    elif 'beesbook_two_underscores' in request.param:
        name = "Cam_1_20160501160208__5_TO_Cam_1_20160501160748__1.avi"
        dt_begin = dt_begin.replace(microsecond=5)
        dt_end = dt_end.replace(microsecond=1)
    elif 'iso' in request.param:
        if 'hyphen' in request.param:
            name = "Cam_0_1970-01-01T00:10:00.000000Z--1970-01-01T00:20:00.000000Z.bbb"
        else:
            name = "Cam_0_19700101T001000.000000Z--19700101T002000.000000Z.bbb"
        dt_begin = datetime.fromtimestamp(10*60, tz=pytz.utc)
        dt_end = datetime.fromtimestamp(20*60, tz=pytz.utc)
        cam = 0
        vformat = 'iso'
    elif 'arbitrary' in request.param:
        vformat = 'arbitrary'

    if 'auto' in request.param:
        vformat = 'auto'
    return {'dt_begin': dt_begin, 'dt_end': dt_end, 'cam': cam, 'format': vformat, 'name': name}


def test_parse_fname_videos(video):
    """Tests the extraction of camera, date and time information from filenames."""
    camIdx, begin, end = parse_video_fname(video['name'])
    assert camIdx == video['cam']
    assert begin == video['dt_begin']
    assert end == video['dt_end']


def test_parse_video_fname(video):
    """Tests the extraction of camera and date information from video filenames."""
    if video['format'] == 'arbitrary':
        with pytest.raises(Exception):
            camIdx, begin, end = parse_video_fname(video['name'], format=video['format'])
        return

    camIdx, begin, end = parse_video_fname(video['name'], format=video['format'])
    assert camIdx == video['cam']
    assert begin == video['dt_begin']
    assert end == video['dt_end']
