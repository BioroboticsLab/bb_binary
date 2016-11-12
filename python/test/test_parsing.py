# -*- coding: utf-8 -*-
"""Tests parsing functions to parse information from filenames and to python objects."""
# pylint:disable=redefined-outer-name
from __future__ import division
from datetime import datetime
import numpy as np
import pytz
import pytest
from bb_binary import parse_cam_id, parse_fname, parse_video_fname, parse_image_fname, \
    dt_to_str, to_datetime, to_timestamp, int_id_to_binary, binary_id_to_int, \
    get_fname, get_video_fname


def test_dt_to_str():
    """Test conversion of datetime objects to string representation."""
    dt = datetime(2015, 8, 15, 12, 0, 40, 333967, tzinfo=pytz.utc)
    assert dt_to_str(dt) == "2015-08-15T12:00:40.333967Z"

    with pytest.raises(Exception) as excinfo:
        dt_to_str(dt.replace(tzinfo=None))
    assert str(excinfo.value) == "Got a datetime object not in UTC. Allways use UTC."


def test_get_fname():
    """Tests the generation of filenames from metadata."""
    dt = datetime(1970, 1, 1, tzinfo=pytz.utc)
    assert get_fname(0, dt) == "Cam_0_1970-01-01T00:00:00Z"

    dt = datetime(2015, 8, 15, 12, 0, 40, tzinfo=pytz.utc)
    assert get_fname(1, dt) == "Cam_1_2015-08-15T12:00:40Z"


def test_get_video_fname():
    """Tests the generation of filenames from metadata."""
    dt1 = datetime(1970, 1, 1, tzinfo=pytz.utc)
    dt2 = datetime(2015, 8, 15, 12, 0, 40, tzinfo=pytz.utc)
    assert get_video_fname(0, dt1, dt2) == "Cam_0_1970-01-01T00:00:00Z--2015-08-15T12:00:40Z"


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


def test_to_timestamp():
    """Tests the conversion of several data types to timestamps."""
    dt = datetime(1970, 1, 1, tzinfo=pytz.utc)
    assert to_timestamp(dt) == 0
    dt = datetime(2015, 8, 15, 12, 0, 40, tzinfo=pytz.utc)
    assert to_timestamp(dt) == 1439640040


@pytest.fixture
def int_bin_mapping():
    """Mapping from integer to binary array represenations."""
    mapping = []
    mapping.append((8, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)))
    mapping.append((4095, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)))

    beeid_digits = 12
    # test setting of single bits
    expected_result = np.zeros((beeid_digits, beeid_digits), int)
    np.fill_diagonal(expected_result, 1)
    expected_result = np.fliplr(expected_result)

    for i in range(0, beeid_digits):
        mapping.append((2**i, expected_result[:, i]))

    # correct conversion of all integers in range
    for i in range(0, 2**beeid_digits):
        number_bin = bin(i)[2:]
        number_bin = '0' * (beeid_digits - len(number_bin)) + number_bin
        bit_array = np.array([1 if bit == '1' else 0 for bit in number_bin], dtype=np.uint8)
        mapping.append((i, bit_array))
    return mapping


def test_int_id_to_binary(int_bin_mapping):
    """Test conversion of integer id representation to binary array representation."""
    for int_repr, bin_repr in int_bin_mapping:
        assert np.all(int_id_to_binary(int_repr) == bin_repr)

    with pytest.raises(Exception) as exception_information:  # to big value
        int_id_to_binary(8096)
    assert 'overflows' in str(exception_information.value)


def test_binary_id_to_int(int_bin_mapping):
    """Test conversion of binary array representation to interger id."""
    for int_repr, bin_repr in int_bin_mapping:
        assert binary_id_to_int(bin_repr) == int_repr

    # test with floats
    for int_repr, bin_repr in int_bin_mapping:
        assert binary_id_to_int(bin_repr / 2) == int_repr

    # change endianess
    for int_repr, bin_repr in int_bin_mapping:
        bin_repr = bin_repr[::-1]
        assert binary_id_to_int(bin_repr / 2, endian='little') == int_repr

    # different threshold
    bit_array = np.arange(0, 1.2, 0.1)
    assert len(bit_array) == 12
    assert np.sum(int_id_to_binary(binary_id_to_int(bit_array))) == 12 - 5
    assert np.sum(int_id_to_binary(binary_id_to_int(bit_array, threshold=0))) == 12
    assert np.sum(int_id_to_binary(binary_id_to_int(bit_array, threshold=1))) == 2


@pytest.fixture(params=['iso', 'beesbook', 'auto_iso', 'auto_bb', 'arbitrary', 'iso_nomilis'])
def image(request):
    """Fixture to test extraction of information on different image filenames."""
    name_beesbook = 'Cam_0_20140805151756_200.jpeg'
    name_iso = 'Cam_0_2014-08-05T13:17:56.000200Z.jpeg'
    name_iso_nomilis = 'Cam_0_2014-08-05T13:17:56.jpeg'
    expected_dt = datetime(2014, 8, 5, 13, 17, 56, 200, tzinfo=pytz.utc)
    expected_dt_nomilis = datetime(2014, 8, 5, 13, 17, 56, tzinfo=pytz.utc)
    expected_cam = 0
    data = {'cam': expected_cam, 'format': 'beesbook', 'name': name_beesbook}
    if request.param == 'iso_nomilis':
        data['dt'] = expected_dt_nomilis
    else:
        data['dt'] = expected_dt
    if 'iso' in request.param:
        data['format'] = 'iso'
        if 'nomilis' in request.param:
            data['name'] = name_iso_nomilis
        else:
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


def test_parse_cam_id_images(image):
    """Tests the extraction of the camera id from video filenames."""
    assert parse_cam_id(image['name']) == image['cam']


def test_parse_cam_id_videos(video):
    """Tests the extraction of the camera id from image filenames."""
    assert parse_cam_id(video['name']) == video['cam']
