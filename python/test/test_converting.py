# -*- coding: utf-8 -*-
"""Tests convenience functions to convert bb_binary data to structured NumPy Arrays."""
# pylint:disable=redefined-outer-name
from datetime import datetime

import numpy as np
import pandas as pd
import pandas.util.testing as pdut
import pytest
import pytz

import bb_binary as bbb
import bb_binary.converting as bbb_c


@pytest.fixture
def frame_data():
    """Frame without detections."""
    frame = bbb.Frame.new_message()
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
    keys = ['idx', 'xpos', 'ypos', 'decodedId', 'detectionsUnion']

    additional_keys = {
        'detectionsDP': ['xRotation', 'yRotation', 'zRotation',
                         'radius', 'localizerSaliency', 'descriptor'],
        'detectionsCVP': ['xRotation', 'yRotation', 'zRotation',
                          'gridIdx', 'candidateIdx', 'eScore', 'gScore', 'lScore'],
        'detectionsTruth': ['readability']
    }[union_type]

    keys.extend(additional_keys)
    return keys


def convert_readability(dfr):
    """Helper to convert readability column from binary to string."""
    dfr.readability = dfr.readability.apply(lambda x: x.decode('UTF-8'))
    return dfr


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


def test_bbb_build_frame_container():
    """Tests the generation of a ``FrameContainer`` via function."""
    # very simple case
    fco = bbb.build_frame_container(0, 1, 0)
    assert fco.camId == 0
    assert fco.fromTimestamp == 0
    assert fco.toTimestamp == 1

    ts1 = bbb.to_timestamp(datetime(1970, 1, 1, tzinfo=pytz.utc))
    ts2 = bbb.to_timestamp(datetime(2015, 8, 15, 12, 0, 40, tzinfo=pytz.utc))

    # more advanced
    fco = bbb.build_frame_container(ts1, ts2, 1)
    assert fco.camId == 1
    assert fco.fromTimestamp == ts1
    assert fco.toTimestamp == ts2

    # try to set every parameter
    fco = bbb.build_frame_container(ts1, ts2, 1, hive_id=5, data_source_fname="testname")
    assert fco.hiveId == 5
    assert len(fco.dataSources) == 1
    assert fco.dataSources[0].filename == "testname"
    assert fco.dataSources[0].videoPreviewFilename == ""

    # filename and video are both strings
    fco = bbb.build_frame_container(ts1, ts2, 1, data_source_fname="testname",
                                    video_preview_fname="test_video_name")
    assert len(fco.dataSources) == 1
    assert fco.dataSources[0].filename == "testname"
    assert fco.dataSources[0].videoPreviewFilename == "test_video_name"

    # try combination of filenames and preview names
    fnames = ["testname", "testname 1"]
    vnames = ["test_video_name", "test_video_name 1"]
    fco = bbb.build_frame_container(ts1, ts2, 1, data_source_fname=fnames,
                                    video_preview_fname=vnames)
    assert len(fco.dataSources) == 2
    assert fco.dataSources[0].filename == "testname"
    assert fco.dataSources[0].videoPreviewFilename == "test_video_name"
    assert fco.dataSources[1].filename == "testname 1"
    assert fco.dataSources[1].videoPreviewFilename == "test_video_name 1"


def test_bbb_frame_container_errors(frame_data_all):
    """Test minimal keys when building `FrameContainer`"""
    frame = frame_data_all
    expected_keys = ('timestamp', 'xpos', 'ypos', 'detectionsUnion')

    arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)
    detections = pd.DataFrame(arr)
    with pytest.raises(AssertionError) as error_information:
        bbb.build_frame_container_from_df(detections, frame.detectionsUnion.which(), 0)
    assert 'decodedId' in str(error_information.value)


def test_bbb_fc_from_df(frame_data_all):
    """Data in DataFrame correctly converted to `FrameContainer`."""
    frame = frame_data_all
    union_type = frame.detectionsUnion.which()
    expected_keys = ['timestamp', ]
    expected_keys.extend(get_detection_keys(union_type))

    arr = bbb.convert_frame_to_numpy(frame, expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)
    detections = make_df_from_np(arr, union_type)
    expected_detections = detections.copy()
    n_detections = len(getattr(frame.detectionsUnion, union_type))

    offset = 0
    # test minimal set
    dfr = detections.copy()
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset)
    assert offset == 1
    assert fco.id == 0
    assert fco.camId == 0
    assert fco.fromTimestamp == expected_detections.timestamp[0]
    assert fco.toTimestamp == expected_detections.timestamp[3]
    assert len(fco.frames) == 1

    frame0 = fco.frames[0]
    assert frame0.id == offset - 1
    assert frame0.frameIdx == 0
    assert frame0.timestamp == expected_detections.timestamp[0]
    assert len(getattr(frame0.detectionsUnion, union_type)) == 4

    arr = bbb.convert_frame_to_numpy(frame0, expected_keys)
    converted_detections = make_df_from_np(arr, union_type)
    pdut.assert_frame_equal(expected_detections, converted_detections)

    # test without readability
    dfr = detections.copy()
    if union_type == 'detectionsTruth':
        dfr.drop('readability', axis=1, inplace=True)
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset, frame_offset=offset)
    assert offset == 2  # test offset and id to test for fixed assignments
    assert fco.id == 1
    assert fco.camId == 1
    assert len(fco.frames) == 1

    frame0 = fco.frames[0]
    assert frame0.id == offset - 1  # test if offset is considered
    assert frame0.frameIdx == 0

    if union_type == 'detectionsTruth':
        for i in range(0, n_detections):  # test for default readability value
            assert str(frame0.detectionsUnion.detectionsTruth[i].readability) == 'unknown'

    # test with datetime instead of unixtimestamp
    dfr = detections.copy()
    dfr.timestamp = dfr.timestamp.apply(bbb.to_datetime)
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset, frame_offset=offset)
    assert offset == 3
    assert fco.fromTimestamp == expected_detections.timestamp[0]
    assert len(fco.frames) == 1

    # test with additional column for frames
    dfr = detections.copy()
    dfr['dataSourceIdx'] = 99
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset, frame_offset=offset)
    assert offset == 4
    assert len(fco.frames) == 1

    frame0 = fco.frames[0]
    assert frame0.dataSourceIdx == dfr.dataSourceIdx[0]

    # test with additional column for detections
    dfr = detections.copy()
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset, frame_offset=offset)
    assert offset == 5
    assert len(fco.frames) == 1

    # test with camId column
    dfr = detections.copy()
    dfr['camId'] = [offset] * dfr.shape[0]
    dfr.loc[0, 'camId'] = offset - 1
    fco, offset = bbb.build_frame_container_from_df(dfr, union_type, offset, frame_offset=offset)
    assert offset == 6
    assert len(fco.frames) == 1

    converted_detections = getattr(fco.frames[0].detectionsUnion, union_type)
    assert len(converted_detections) == n_detections - 1  # one detection removed!


def test_bbb_convert_detections(frame_data_all):
    """Detections are correctly converted to np array and frame is ignored."""
    frame = frame_data_all
    expected_keys = get_detection_keys(frame.detectionsUnion.which())

    detections = bbb_c._get_detections(frame)
    arr = bbb_c._convert_detections_to_numpy(detections, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)

    # now compare behaviour of helper and parent function
    arr_frame = bbb.convert_frame_to_numpy(frame, keys=expected_keys)
    assert np.all(arr == arr_frame)


def test_bbb_convert_only_frame(frame_data_all):
    """Frame is correctly converted to np array and detections are ignored."""
    frame = frame_data_all

    expected_keys = ('frameId', 'timedelta', 'timestamp', 'detectionsUnion')

    arr = bbb_c._convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)

    # now compare behaviour of helper and parent function
    arr_frame = bbb.convert_frame_to_numpy(frame, keys=expected_keys)
    assert np.all(arr == arr_frame)


def test_bbb_convert_frame_det(frame_data_all):
    """Frame and detections are correctly converted to np array."""
    frame = frame_data_all

    expected_keys = ['frameId', 'timedelta', 'timestamp']
    expected_keys.extend(get_detection_keys(frame.detectionsUnion.which()))

    arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys)
    bbb_check_frame_data(frame, arr, expected_keys)


def test_bbb_convert_frame_add_cols(frame_data_all):
    """Frame with additional columns is correctly converted to np array."""
    frame = frame_data_all

    expected_keys = ('frameId', 'timedelta', 'timestamp', 'decodedId', 'detectionsUnion')

    # one col, single value for whole columns
    arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': 2})
    assert 'camId' in arr.dtype.names
    assert np.all(arr['camId'] == 2)

    # two cols, single value
    arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': 2, 'second': 3})
    assert 'camId' in arr.dtype.names
    assert 'second' in arr.dtype.names
    assert np.all(arr['camId'] == 2)
    assert np.all(arr['second'] == 3)

    # list for whole column
    arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'camId': range(0, 4)})
    assert 'camId' in arr.dtype.names
    assert np.all(arr['camId'] == np.array(range(0, 4)))

    # existing column
    with pytest.raises(AssertionError):
        arr = bbb.convert_frame_to_numpy(frame, keys=expected_keys, add_cols={'frameId': 9})


def test_bbb_convert_frame_default(frame_data_all):
    """Convert frame to np arrays without explicitly requesting certain keys."""
    frame = frame_data_all

    frame_keys = set(['frameId', 'dataSourceIdx', 'frameIdx', 'timestamp', 'timedelta'])
    detection_keys = set(get_detection_keys(frame.detectionsUnion.which()))
    expected_keys = frame_keys | detection_keys | set(['camId'])

    # extract frame without explicitly asking for keys
    arr = bbb.convert_frame_to_numpy(frame, add_cols={'camId': 0})
    bbb_check_frame_data(frame, arr, expected_keys)


def bbb_check_frame_data(frame, arr, expected_keys):
    """Helper to compare frame data to numpy array."""
    # check if we have all the expected keys in the array (and only these)
    expected_keys = set(expected_keys) - set(['detectionsUnion'])
    assert expected_keys == set(arr.dtype.names)
    assert len(expected_keys) == len(arr.dtype.names)

    detection_string_fields = ('readability')
    detections = bbb_c._get_detections(frame)
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
    detections = bbb_c._get_detections(frame_dp_data)
    assert not hasattr(detections[0], diffKey)

    detections = bbb_c._get_detections(frame_cvp_data)
    assert hasattr(detections[0], diffKey)

    detections = bbb_c._get_detections(frame_truth_data)
    assert hasattr(detections[0], 'readability')

    # Important note: the default value for detectionsUnion is detectionsCVP
    assert frame_data.detectionsUnion.which() == 'detectionsCVP'
