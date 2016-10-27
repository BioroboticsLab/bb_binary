# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import iso8601
import pytz

_TIMEZONE = pytz.timezone('Europe/Berlin')


def dt_to_str(dt):
    dt = to_datetime(dt)
    isoformat = "%Y-%m-%dT%H:%M:%S"

    dt_str = dt.strftime(isoformat)
    if dt.microsecond != 0:
        dt_str += ".{:06d}".format(dt.microsecond)
    if dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0:
        return dt_str + "Z"
    else:
        raise Exception("Got a datetime object not in UTC. Allways use UTC.")


def get_fname(camIdx, dt):
    dt = to_datetime(dt)
    return ("Cam_{cam}_{ts}").format(cam=camIdx, ts=dt_to_str(dt))


def get_video_fname(camIdx, begin, end):
    return get_fname(camIdx, begin) + "--" + dt_to_str(end)


def get_timezone():
    return _TIMEZONE


def int_id_to_binary(int_id, nb_bits=12):
    """Helper to convert an id represented as integer to a bit array.

    Arguments:
        int_id (int): the integer id to convert to a bit array

    Keyword Arguments:
        nb_bits (int, optional): number of bits in the bit array

    Returns:
        :obj:`np.array`: the bit array in big-endian notation
    """
    result = np.zeros(nb_bits, dtype=np.uint8)
    if int_id >= 2**nb_bits:
        raise Exception("Id {} overflows {} bits".format(int_id, nb_bits))
    pos = nb_bits - 1
    while pos >= 0:
        result[pos] = int_id & 1
        int_id >>= 1
        pos -= 1
    return result


def parse_cam_id(fname):
    fname = os.path.basename(fname)
    camIdx = fname.split("_")[1]
    return int(camIdx)


def parse_fname(fname):
    fname = os.path.basename(fname)
    try:
        return parse_video_fname(fname, 'iso')
    except:
        try:
            return parse_video_fname(fname, 'beesbook')
        except:
            camIdx, dtime = parse_image_fname(fname)
            return camIdx, dtime, dtime


def parse_image_fname_beesbook(fname):
    basename = os.path.basename(fname)
    name = basename.split('.')[0]
    name = name.replace('__', '_')
    _, camIdxStr, datetimeStr, usStr = name.split('_')

    camIdx = int(camIdxStr)
    year = int(datetimeStr[:4])
    month = int(datetimeStr[4:6])
    day = int(datetimeStr[6:8])

    hour = int(datetimeStr[8:10])
    minute = int(datetimeStr[10:12])
    second = int(datetimeStr[12:14])
    us = int(usStr)

    dt = datetime(year, month, day, hour, minute, second, us)
    dt = get_timezone().localize(dt)
    return camIdx, dt.astimezone(pytz.utc)


def parse_image_fname_iso(fname):
    basename = os.path.basename(fname)
    name_splitted = basename.split('.')

    def parse_name(name):
        _, camIdxStr, iso_str = name.split('_')
        dt = iso8601.parse_date(iso_str)
        return int(camIdxStr), dt

    try:
        name = '.'.join(name_splitted[:2])
        camIdx, dt = parse_name(name)
    except iso8601.ParseError:
        name = name_splitted[0]
        camIdx, dt = parse_name(name)

    return camIdx, dt


def parse_image_fname(fname, format='auto'):
    if format == 'beesbook':
        return parse_image_fname_beesbook(fname)
    elif format == 'iso':
        return parse_image_fname_iso(fname)
    elif format == 'auto':
        basename = os.path.basename(fname)
        if basename.count('_') >= 3:
            return parse_image_fname_beesbook(fname)
        else:
            return parse_image_fname_iso(fname)
    else:
        raise Exception("Unknown format {}.".format(format))


def parse_video_fname(fname, format='auto'):
    def beesbook_parse():
        begin_name, end_name = basename.split('_TO_')
        (camIdx, begin) = parse_image_fname(begin_name, 'beesbook')
        (_, end) = parse_image_fname(end_name, 'beesbook')
        return camIdx, begin, end

    def iso_parse():
        name, _ = os.path.splitext(basename)
        _, camIdx, isotimespan = name.split('_')
        start, end = isotimespan.split('--')
        end = end.rstrip(".bbb")
        return int(camIdx), iso8601.parse_date(start), iso8601.parse_date(end)

    basename = os.path.basename(fname)
    if format == 'beesbook':
        return beesbook_parse()
    elif format == 'iso':
        return iso_parse()
    elif format == 'auto':
        try:
            return beesbook_parse()
        except ValueError:
            return iso_parse()
    else:
        raise ValueError("Unknown format {}.".format(format))


def to_datetime(t):
    if type(t) in (int, float):
        dt = datetime.fromtimestamp(t, tz=pytz.utc)
        return dt
    elif type(t) == datetime:
        return t
    else:
        raise TypeError("Cannot convert {} to datetime".format(t))


def to_timestamp(dt):
    try:
        return dt.timestamp()
    except AttributeError:  # python 2
        utc_naive = dt.replace(tzinfo=None) - dt.utcoffset()
        timestamp = (utc_naive - datetime(1970, 1, 1)).total_seconds()
        return timestamp
