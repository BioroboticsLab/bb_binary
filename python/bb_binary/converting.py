# -*- coding: utf-8 -*-
"""Converter"""
# pylint:disable=no-member
from datetime import datetime
import numpy as np
import numpy.lib.recfunctions as rf
import pytz
import six
from bb_binary.constants import Frame, DetectionCVP, DetectionDP, DetectionTruth
from bb_binary.parsing import to_timestamp
from bb_binary.repository import build_frame_container


def convert_frame_to_numpy(frame, keys=None, add_cols=None):
    """Returns the frame data and detections as a numpy array from the frame.

    Note:
        The frame id is identified in the array as ``frameId`` instead of ``id``!

    Args:
        frame (Frame): datastructure with frame data from capnp.

    Keyword Args:
        keys (Optional iterable): only these keys are converted to the np array.
        add_cols (Optional dict): additional columns for the np array,
            use either a single value or a sequence of correct length.

    Returns:
        numpy array (np.array): a structured numpy array with frame and detection data.
    """
    ret_arr = None

    if keys is None or 'detectionsUnion' in keys:
        detections = _get_detections(frame)
        ret_arr = _convert_detections_to_numpy(detections, keys)

    frame_arr = _convert_frame_to_numpy(frame, keys)
    if ret_arr is not None and frame_arr is not None:
        frame_arr = np.repeat(frame_arr, ret_arr.shape[0], axis=0)
        ret_arr = rf.merge_arrays((frame_arr, ret_arr),
                                  flatten=True, usemask=False)
    elif frame_arr is not None:
        ret_arr = frame_arr

    if ret_arr is not None and add_cols is not None:
        if keys is None:
            keys = ret_arr.dtype.names
        for key, val in add_cols.items():
            assert key not in keys, "{} not allowed in add_cols".format(key)
            if hasattr(val, '__len__') and not isinstance(val, six.string_types):
                msg = "{} has not length {}".format(key, ret_arr.shape[0])
                assert len(val) == ret_arr.shape[0], msg
            else:
                val = np.repeat(val, ret_arr.shape[0], axis=0)
            ret_arr = rf.append_fields(ret_arr, key, val, usemask=False)

    return ret_arr


def _convert_frame_to_numpy(frame, keys=None):
    """Helper function for :func:`convert_frame_to_numpy`.

    Converts the frame data to a numpy array.
    """
    # automatically deduce keys and types from frame
    frame_keys = set(frame.to_dict().keys())
    frame_keys.discard('detectionsUnion')
    if keys is None:
        keys = list(frame_keys)
    else:
        keys_set = set(keys)
        if 'frameId' in keys:
            keys_set.add('id')
        keys = list(keys_set & frame_keys)

    # abort if no frame information should be extracted
    if len(keys) == 0:
        return None

    fields = [getattr(frame, key) for key in keys]
    formats = [type(field) for field in fields]

    # create frame
    frame_arr = np.array(tuple(fields),
                         dtype={'names': keys, 'formats': formats})

    # replace id with frameId for better readability!
    frame_arr.dtype.names = ["frameId" if x == "id" else x
                             for x in frame_arr.dtype.names]

    return frame_arr


def _convert_detections_to_numpy(detections, keys=None):
    """Helper function for :func:`convert_frame_to_numpy`.

    Converts the detections data to a numpy array.
    """
    nrows = len(detections)

    # automatically deduce keys and types except for decodedId
    detection0 = detections[0].to_dict()
    detection_keys = set(detection0.keys())
    if keys is None:
        keys = list(detection_keys)
    else:
        keys = list(set(keys) & detection_keys)

    # abort if no information should be extracted
    if len(keys) == 0:
        return None

    formats = [type(detection0[key]) for key in keys]

    readability_key = 'readability'
    decoded_id_key = 'decodedId'
    decoded_id_index = None
    descriptor_key = 'descriptor'
    descriptor_index = None
    if decoded_id_key in keys and isinstance(detection0[decoded_id_key], list):
        # special handling of decodedId as float array in DP pipeline data
        decoded_id_index = keys.index(decoded_id_key)
        formats[decoded_id_index] = str(len(detection0[decoded_id_key])) + 'f8'
    elif readability_key in keys:
        # special handling of enum because numpy does not determine str length
        readbility_index = keys.index(readability_key)
        formats[readbility_index] = 'S10'
    if descriptor_key in keys and isinstance(detection0[descriptor_key], list):
        # special handling of descriptor as uint8 array in DP pipeline data
        descriptor_index = keys.index(descriptor_key)
        formats[descriptor_index] = str(len(detection0[descriptor_key])) + 'u8'
    detection_arr = np.empty(nrows, dtype={'names': keys, 'formats': formats})
    for i, detection in enumerate(detections):
        # make sure we have the same order as in keys
        val = [getattr(detection, key) for key in keys]
        if decoded_id_index is not None:
            val[decoded_id_index] = np.array(val[decoded_id_index]) / 255.
        # structured np arrays only accept tuples
        detection_arr[i] = tuple(val)

    return detection_arr


def _get_detections(frame):
    """Extracts detections of DP, CVP or truth data from frame."""
    union_type = frame.detectionsUnion.which()
    if union_type == 'detectionsDP':
        detections = frame.detectionsUnion.detectionsDP
    elif union_type == 'detectionsCVP':
        detections = frame.detectionsUnion.detectionsCVP
    elif union_type == 'detectionsTruth':
        detections = frame.detectionsUnion.detectionsTruth
    else:
        raise KeyError("Type {0} not supported.".format(union_type))  # pragma: no cover
    return detections


def build_frame_container_from_df(dfr, union_type, cam_id, frame_offset=0):
    """Builds a frame container from a Pandas DataFrame.

    Operates differently from :func:`build_frame_container` because it will be used
    in a different context where we have access to more data.

    Column names are matched to ``Frame`` and ``Detection*`` attributes.
    Set additional ``FrameContainer`` attributes like ``hiveId`` in the return value.

    Args:
        dfr (dataframe): dataframe with detection data
        union_type (str): the type of detections e.g. ``detectionsTruth``
        cam_id (int): id of camera, also used as ``FrameContainer`` id

    Keyword Args:
        frame offset (Optional int): offset for unique frame ids

    Returns:
        (tuple): tuple containing:

            - **frame container** (*FrameContainer*): converted data from :attr:`dfr`
            - **new offset** (*int*): number of frames that could be used as :attr:`frame_offset`
     """
    def set_attr_from(obj, src, key):
        """Get attr :attr:`key` from :attr:`src` and set val to :attr:`obj` on same :attr:`key`"""
        val = getattr(src, key)
        # special handling for list type fields
        if key in list_keys:
            set_list_attr(obj, val, key)
            return
        if type(val).__module__ == np.__name__:
            val = np.asscalar(val)
        setattr(obj, key, val)

    def set_list_attr(obj, list_src, key):
        """Initialize list :attr:`key` on :attr:`obj` and set all values from :attr:`list_src`."""
        new_list = obj.init(key, len(list_src))
        for i, val in enumerate(list_src):
            if type(val).__module__ == np.__name__:
                val = np.asscalar(val)
            new_list[i] = val

    detection = {
        'detectionsCVP': DetectionCVP.new_message(),
        'detectionsDP': DetectionDP.new_message(),
        'detectionsTruth': DetectionTruth.new_message()
    }[union_type]

    # check that we have all the information we need
    skip_keys = frozenset(['readability', 'xposHive', 'yposHive', 'frameIdx', 'idx'])
    minimal_keys = set(detection.to_dict().keys()) - skip_keys
    list_keys = set()
    # for some reasons lists are not considered when using to_dict()!
    if union_type == 'detectionsDP':
        minimal_keys = minimal_keys | set(['decodedId'])
        list_keys = list_keys | set(['decodedId', 'descriptor'])
    available_keys = set(dfr.keys())
    assert minimal_keys <= available_keys,\
        "Missing keys {} in DataFrame.".format(minimal_keys - available_keys)

    # select only entries for cam
    if 'camId' in available_keys:
        dfr = dfr[dfr.camId == cam_id].copy()

    # convert timestamp to unixtimestamp
    if 'datetime' in dfr.dtypes.timestamp.name:
        dfr.loc[:, 'timestamp'] = dfr.loc[:, 'timestamp'].apply(
            lambda t: to_timestamp(datetime(
                t.year, t.month, t.day, t.hour, t.minute, t.second,
                t.microsecond, tzinfo=pytz.utc)))

    # convert decodedId from float to integers (if necessary)
    if 'decodedId' in available_keys and union_type == 'detectionsDP' and\
       np.all(np.array(dfr.loc[dfr.index[0], 'decodedId']) < 1.1):
        dfr.loc[:, 'decodedId'] = dfr.loc[:, 'decodedId'].apply(
            lambda l: [int(round(fid * 255.)) for fid in l])

    # create frame container
    tstamps = dfr.timestamp.unique()
    start = np.asscalar(min(tstamps))
    end = np.asscalar(max(tstamps))
    new_frame_offset = frame_offset + len(tstamps)

    fco = build_frame_container(start, end, cam_id)
    fco.id = cam_id  # overwrite if necessary!
    fco.init('frames', len(tstamps))

    # determine which fields we could assign from dataframe to cap'n proto
    frame = Frame.new_message()
    frame_fields = [field for field in available_keys if hasattr(frame, field)]

    detection_fields = [field for field in available_keys if hasattr(detection, field)]

    # create frames (each timestamp maps to a frame)
    for frame_idx, (_, detections) in enumerate(dfr.groupby(by='timestamp')):
        frame = fco.frames[frame_idx]
        frame.id = frame_offset + frame_idx
        frame.frameIdx = frame_idx

        # take first row, assumes that cols `frame_fields` have unique values!
        for key in frame_fields:
            set_attr_from(frame, detections.iloc[0], key)

        # create detections
        frame.detectionsUnion.init(union_type, detections.shape[0])
        for detection_idx, row in enumerate(detections.itertuples(index=False)):
            detection = getattr(frame.detectionsUnion, union_type)[detection_idx]
            detection.idx = detection_idx
            for key in detection_fields:
                set_attr_from(detection, row, key)

    return fco, new_frame_offset
