# -*- coding: utf-8 -*-
"""
This submodule contains a collection of helper to parse filenames and some fields in
*bb_binary* repositories to Python datastructures.

There are also some helper to convert some data representations like timeformats or ids.
"""
import os
from datetime import datetime
from bitarray import bitarray
import numpy as np
import iso8601
import pytz
import re

_TIMEZONE = pytz.timezone('Europe/Berlin')
def get_timezone():
    return _TIMEZONE

## Generating names from timestamps
def dt_to_str(dt):
    """Converts a datetime object to a formatted string."""
    dt = to_datetime(dt)
    isoformat = "%Y-%m-%dT%H_%M_%S"

    dt_str = dt.strftime(isoformat)
    if dt.microsecond != 0:
        dt_str += ".{:06d}".format(dt.microsecond)
    if dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0:
        return dt_str + "Z"
    else:
        raise Exception("Got a datetime object not in UTC. Allways use UTC.")

def get_fname(camIdx, dt):
    """Generates a filename based on the camera ID and datetime."""
    dt = to_datetime(dt)
    return ("Cam_{cam}_{ts}").format(cam=camIdx, ts=dt_to_str(dt))

def get_video_fname(camIdx, begin, end):
    """Generates a video filename based on the camera ID, begin and end timestamps."""
    return get_fname(camIdx, begin) + "--" + dt_to_str(end)

## Datetime to/from Unix timestamps
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

## Helper functions for ID's
def int_id_to_binary(int_id, nb_bits=12):
    """Helper to convert an id represented as integer to a bit array.

    Warning:
        This function uses big-endian notation whereas :obj:`.DetectionDP` uses
        little-endian notation for `decodedId`.

    Arguments:
        int_id (int): the integer id to convert to a bit array

    Keyword Arguments:
        nb_bits (Optional int): number of bits in the bit array

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

def binary_id_to_int(decoded_id, threshold=0.5, endian='big'):
    """Helper to convert an id represented as bit array to an integer.

    Warning:
        :obj:`.DetectionDP` uses little-endian notation for `decodedId`.

    Note:
        This is a generic solution to the problem. If have to decode a lot of ids take a look on
        `numpy.packbits()
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.packbits.html>`_ and implement a
        vectorized version.

    Arguments:
        decoded_id (:obj:`list` of int or float): id as bit array

    Keyword Arguments:
        threshold (Optional float): `decoded_id` values >= threshold are interpreted as 1
        endian (Optional str): use either 'big' or 'little', default is 'big'

    Returns:
        int: the decoded id represented as integer
    """
    if endian == 'little':
        decoded_id = decoded_id[::-1]
    return int(bitarray([bit >= threshold for bit in decoded_id], endian=endian).to01(), 2)


## image - frame parsing functions, with basler helper function.
# Frames have a single camera id and a single timestamp,  e.g. cam-0_20240621T100554.249052.472Z
def parse_image_fname(fname, format='auto'):
    """Parses a filename to extract the camera ID and timestamp."""
    parsers = {
        'beesbook': parse_image_fname_beesbook,
        'iso': parse_image_fname_iso,
        'basler': parse_image_fname_basler
    }
    basename = os.path.basename(fname)
    if format == 'auto':
        if basename.startswith('cam-'):  # Detect the 'basler' format: it starts with 'cam-'
            return parse_image_fname_basler(fname)
        elif basename.count('_') >= 3: # If the filename contains 3 or more underscores, assume 'beesbook' format
            return parse_image_fname_beesbook(fname)
        else:  # Try ISO format as a fallback
            try:
                return parse_image_fname_iso(fname)
            except ValueError:
                raise ValueError(f"Filename '{fname}' does not match any known format (basler, iso, beesbook).")
    else:  # use the format passed in
        if format not in parsers:
            raise ValueError(f"Unknown format '{format}' provided.")
        return parsers[format](fname)

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

def parse_image_fname_basler(fname):
    basename = os.path.basename(fname)
    def parse_name(name):
        camIdxStr, date_str = name.split('_')
        dt = iso8601.parse_date(basler_to_iso_format(date_str))
        return int(camIdxStr.replace('cam-', '')), dt
    try:
        camIdx, dt = parse_name(basename)
    except iso8601.ParseError:  # don't save the seconds, just the minute, if there is a problem
        name_splitted = basename.split('.')
        camIdx, dt = parse_name(name_splitted[0])

    return camIdx, dt

def basler_to_iso_format(date_str):
    # Add colons and dashes to match ISO 8601 format and handle fractional seconds correctly
    # fractional seconds should have 6 digits in ISO format.  pad to the left with a zero if there are only 5 digits
    date_str = re.sub(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})\.(\d+)\.(\d+)', 
                      lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}.{m.group(7).zfill(6)}Z", 
                      date_str)
    # Ensure there is only one 'Z' at the end
    date_str = re.sub(r'Z+', 'Z', date_str)
    return date_str



## Video filename parsing
# Videos have a single camera id and two timestamps for start and end,  e.g. cam-0_20240621T100553.581867.682Z--20240621T100653.461292.962Z.txt
def parse_video_fname(fname, format='auto'):
    def beesbook_parse():
        begin_name, end_name = basename.split('_TO_')
        (camIdx, begin) = parse_image_fname(begin_name, 'beesbook')
        (_, end) = parse_image_fname(end_name, 'beesbook')
        return camIdx, begin, end

    def iso_parse(): # was previously used for .bbb files
        name, _ = os.path.splitext(basename)
        _, camIdx, isotimespan = name.split('_')
        start, end = isotimespan.split('--')
        end = end.rstrip(".bbb")
        return int(camIdx), iso8601.parse_date(start), iso8601.parse_date(end)
    
    def bbb_parse():
        name, _ = os.path.splitext(basename)        
        _, camIdx, isotimespan = name.split('_', 2)
        start, end = isotimespan.split('--')
        start = start.replace('_', ':')  # to convert to iso format
        end = end.rstrip(".bbb").replace('_', ':')
        return int(camIdx), iso8601.parse_date(start), iso8601.parse_date(end)    
    
    def basler_parse():
        name, _ = os.path.splitext(basename)
        camIdx, isotimespan = name.split('_')
        start, end = isotimespan.split('--')
        # Correct the date format
        start = basler_to_iso_format(start)
        end = basler_to_iso_format(end)
        return int(camIdx.replace('cam-', '')), iso8601.parse_date(start), iso8601.parse_date(end)

    basename = os.path.basename(fname)
    parsers = {
        'beesbook': beesbook_parse,
        'iso': iso_parse,
        'bbb': bbb_parse,
        'basler': basler_parse
    }    
    # If a specific format is provided, use the corresponding parser
    if format in parsers:
        return parsers[format]()
    # If format is 'auto', try each parser in order
    for parser in parsers.values():
        try:
            return parser()
        except (ValueError, KeyError):
            continue
    # If no parser succeeded, raise an error
    raise ValueError(f"Filename '{fname}' does not match any known format (beesbook, iso, bbb, basler).")


## General parsing functions
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