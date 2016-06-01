
import bb_binary as bbb
import time
import numpy as np


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
    repo = bbb.Repository(str(tmpdir), 0, 10000)
    assert tmpdir.join('bbb_repo.json').exists()

    loaded_repo = bbb.Repository.load(str(tmpdir))
    assert repo == loaded_repo
