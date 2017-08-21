# -*- coding: utf-8 -*-
"""
We provide convenience functions to read a *bb_binary* :class:`.Repository` to NumPy Arrays
and DataFrames. Or to create a :class:`.Repository` from existing data via Pandas Dataframes.

You might need them if you want to

    - create *bb_binary* Repositories from existing data (like ground truth data)
    - take a quick glance at a subset of the data (one that fits into memory)
    - experiment with new features

.. warning::
    The convenience functions are **not** designed for performance. When working with
    huge datasets it is recommended to use the :class:`.Repository`.

Convert *bb_binary* to NumPy Array
----------------------------------

To generate NumPy Arrays or Pandas DataFrames we provide a convenience function. Here is
an example how to read all the frames and detection data to NumPy::

    import numpy as np
    import pandas as pd
    from bb_binary import Repository, convert_frame_to_numpy

    repo = Repository("some/path/to/a/repo")

    arr = None
    for frame, fc in repo.iter_frames():
        tmp = convert_frame_to_numpy(frame)
        arr = tmp if arr is None else np.hstack((arr, tmp))

Sometimes we also need fields from the :obj:`.FrameContainer`. You can add those
fields using the `add_cols` argument. This works for every other singular values or lists::

    arr = None
    for frame, fc in repo.iter_frames():
        tmp = convert_frame_to_numpy(frame, add_cols={'camId': fc.camId})
        arr = tmp if arr is None else np.hstack((arr, tmp))

It is also possible to restrict the output to a set of fields that should be extracted.
When using the `keys` argument you need to specify `detectionsUnion` as :obj:`.Frame`
key when you want to extract detections::

    arr = None
    frame_keys = ('frameId', 'frameIdx', 'timedelta', 'timestamp', 'dataSourceIdx')
    detection_keys = ('idx', 'xpos', 'ypos')
    keys = frame_keys + detection_keys
    for frame, fc in repo.iter_frames():
        tmp = convert_frame_to_numpy(frame, keys=keys + ('detectionsUnion',))
        arr = tmp if arr is None else np.hstack((arr, tmp))

Same for hiveMappedDetection. You need to specify `detectionsUnion` and `hiveMappedDetection`
as :obj:`.Frame` key when you want to extract hiveMappedDetections ::

    arr = None
    frame_keys = ('frameId', 'frameIdx', 'timedelta', 'timestamp', 'dataSourceIdx')
    hive_mapped_detection_keys = ('xposHive', 'yposHive')
    keys = frame_keys + hive_mapped_detection_keys
    for frame, fc in repo.iter_frames():
        tmp = convert_frame_to_numpy(frame, keys=keys + ('detectionsUnion', 'hiveMappedDetection'))
        arr = tmp if arr is None else np.hstack((arr, tmp))

Convert *bb_binary* to Pandas DataFrame
----------------------------------------

Usually you could directly create a Pandas DataFrame from a NumPy Array::

    data = pd.DataFrame(arr)

Assuming that we have standard pipeline output with :obj:`.DetectionDP` you have to
convert list like fields separately (because Pandas has problems with lists in fields)::

    list_like_fields = set(['decodedId', 'descriptor'])
    data = pd.DataFrame(arr[list(set(arr.dtype.fields.keys()) - list_like_fields)])
    for field in list_like_fields:
        data[field] = pd.Series([list(list_field) for list_field in arr[field]])

Convert a Pandas DataFrame to *bb_binary*
-----------------------------------------

When you have data from other sources like ground truth data, or you need to generate a
:class:`.Repository` for testing purposes or feature evaluation you might need this
converting function. All the column names in the Pandas DataFrame are matched to field names.
You have to specify the `detectionUnion` type and also the camera id, because each
:obj:`.FrameContainer` is specific for a camera.

The `frame_offset` is used to generate unique :obj:`.Frame` ids::

    from bb_binary import Repository, build_frame_container_from_df
    cam_ids = (0, 2)
    offset = 0
    for cid in cam_ids:
        fc, offset = build_frame_container_from_df(df, 'detectionsTruth', cid, frame_offset=offset)
        repo.add(fc)

Function Documentation
----------------------
"""
# pylint:disable=no-member
from datetime import datetime

import numpy as np
import numpy.lib.recfunctions as rf
import pytz
import six

from bb_binary.common import Frame, FrameContainer, DetectionCVP, DetectionDP, DetectionTruth,\
    HiveMappedDetection
from bb_binary.parsing import to_timestamp


def build_frame_container(from_ts, to_ts, cam_id, hive_id=None, data_source_fname=None,
                          video_preview_fname=None):
    """Builds a :obj:`.FrameContainer`

    Args:
        from_ts (int or float): Timestamp of the first frame
        to_ts (int or float): Timestamp of the last frame
        cam_id (int): id of camera

    Keyword Args:
        hive_id (Optional int): id of the hive
        data_source_fname (Optional str or list of str): Filename(s) of the data source(s).
        video_preview_fname (Optional str or list of str): Filename(s) of preview videos.
            Have to allign to `data_source_fname`!
    """
    fco = FrameContainer.new_message()
    fco.fromTimestamp = from_ts
    fco.toTimestamp = to_ts
    fco.camId = cam_id
    if isinstance(video_preview_fname, six.string_types):
        video_preview_fname = [video_preview_fname]
    if isinstance(data_source_fname, six.string_types):
        data_source_fname = [data_source_fname]
    if data_source_fname is not None:
        # make empty list of strings if no video preview filename is given
        data_sources = fco.init('dataSources', len(data_source_fname))
        for i, filename in enumerate(data_source_fname):
            data_sources[i].filename = filename
            if video_preview_fname is not None:
                data_sources[i].videoPreviewFilename = video_preview_fname[i]
    if hive_id is not None:
        fco.hiveId = hive_id
    return fco


def build_frame_container_from_df(dfr, union_type, cam_id, frame_offset=0):
    """Builds a :obj:`.FrameContainer` from a Pandas DataFrame.

    Operates differently from :func:`build_frame_container` because it will be used
    in a different context where we have access to more data.

    Column names are matched to :obj:`.Frame` and `Detection*` attributes.
    Set additional :obj:`.FrameContainer` attributes like `hiveId` in the return value.

    Args:
        dfr (:obj:`pd.DataFrame`): Pandas dataframe with detection data
        union_type (str): the type of detections e.g. `detectionsTruth`
        cam_id (int): id of camera, also used as :obj:`.FrameContainer` id

    Keyword Args:
        frame offset (Optional int): offset for unique frame ids

    Returns:
        tuple: tuple containing:

            - **frame container** (:obj:`.FrameContainer`): converted data from `dfr`
            - **new offset** (:obj:`int`): number of frames (could be used as `frame_offset`)
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

    hive_detection = HiveMappedDetection.new_message()

    # check that we have all the information we need
    skip_keys = frozenset(['readability', 'frameIdx', 'idx', 'hiveMappedDetection'])
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

    hive_detection_fields = [field for field in available_keys if hasattr(hive_detection, field)]

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
            for key in hive_detection_fields:
                set_attr_from(detection.hiveMappedDetection, row, key)

    return fco, new_frame_offset


def convert_frame_to_numpy(frame, keys=None, add_cols=None):
    """Returns the frame data and detections as a numpy array from the `frame`.

    Note:
        The frame id is identified in the array as `frameId` instead of `id`!

    Args:
        frame (:obj:`.Frame`): datastructure with frame data from capnp.

    Keyword Args:
        keys (Optional iterable of str): only these keys are converted to the np array.
        add_cols (Optional dict): additional columns for the np array,
            use either a single value or a sequence of correct length.

    Returns:
        numpy array (np.array): a structured numpy array with frame and detection data.
    """
    ret_arr = None

    frame_arr = _convert_frame_to_numpy(frame, keys)

    if keys is None or 'detectionsUnion' in keys:
        detections = _get_detections(frame)
        ret_arr = _convert_detections_to_numpy(detections, keys)

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
    # if a tuple has just one element it will be interpret as string
    # this will fix it.
    if type(keys) == str:
        keys = (keys, )
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
    if 'id' in keys:
        formats[keys.index('id')] = np.uint64

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

    # Get potential keys of detectionsUnion, but without the 'level' hiveMappedDetection
    detection_dict = detections[0].to_dict()
    detection_keys = set(detection_dict.keys())
    detection_keys.discard('hiveMappedDetection')

    # extract selected keys
    keys_lvl_detection = []
    if keys is None:
        keys_lvl_detection = list(detection_keys)
    else:
        keys_lvl_detection = list(set(keys) & detection_keys)

    # Get potential keys of hiveMappedDetection
    hivedata_dict = detections[0].hiveMappedDetection.to_dict()
    hivedata_keys = set(hivedata_dict.keys())

    # extract selected keys
    keys_lvl_hivedata = []
    if keys is None or 'hiveMappedDetection' in keys:
        if keys is None:
            keys_lvl_hivedata = list(hivedata_keys)
        else:
            keys_lvl_hivedata = list(set(keys) & hivedata_keys)

    # abort if no information should be extracted
    if len(keys_lvl_hivedata) == 0 and len(keys_lvl_detection) == 0:
        return None

    # get formats of the numpy fields
    formats_lvl_det = [type(detection_dict[key]) for key in keys_lvl_detection]

    readability_key = 'readability'
    decoded_id_key = 'decodedId'
    decoded_id_index = None
    descriptor_key = 'descriptor'
    descriptor_index = None

    if decoded_id_key in keys_lvl_detection and isinstance(detection_dict[decoded_id_key], list):
        # special handling of decodedId as float array in DP pipeline data
        decoded_id_index = keys_lvl_detection.index(decoded_id_key)
        formats_lvl_det[decoded_id_index] = str(len(detection_dict[decoded_id_key])) + 'f8'
    elif readability_key in keys_lvl_detection:
        # special handling of enum because numpy does not determine str length
        readbility_index = keys_lvl_detection.index(readability_key)
        formats_lvl_det[readbility_index] = 'S10'
    if descriptor_key in keys_lvl_detection and isinstance(detection_dict[descriptor_key], list):
        # special handling of descriptor as uint8 array in DP pipeline data
        descriptor_index = keys_lvl_detection.index(descriptor_key)
        formats_lvl_det[descriptor_index] = str(len(detection_dict[descriptor_key])) + 'u8'

    # concat formats of detection and hivedata
    formats = formats_lvl_det + [type(hivedata_dict[key]) for key in keys_lvl_hivedata]
    names = keys_lvl_detection + keys_lvl_hivedata

    # allocate numpy array
    detection_arr = np.empty(nrows, dtype={'names': names, 'formats': formats})

    # fill allocated numpy array
    for i, detection in enumerate(detections):
        # make sure we have the same order as in keys
        val_lvl_detection = [getattr(detection, key) for key in keys_lvl_detection]
        val_lvl_hivedata = [getattr(detection.hiveMappedDetection, key)
                            for key in keys_lvl_hivedata]
        if decoded_id_index is not None:
            val_lvl_detection[decoded_id_index] = np.array(
                val_lvl_detection[decoded_id_index]) / 255.
        # structured np arrays only accept tuples
        val = val_lvl_detection + val_lvl_hivedata
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
