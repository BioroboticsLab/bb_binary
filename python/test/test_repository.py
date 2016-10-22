# -*- coding: utf-8 -*-
"""Tests the Repository class that allows reading and writing to bb_binary data stores."""
# pylint:disable=redefined-outer-name
import os
import math
from datetime import datetime
import pytz
from conftest import fill_repository
# constants
from bb_binary import Frame
# converting
from bb_binary import build_frame_container
# repository
from bb_binary import Repository


def test_bbb_is_loaded():
    frame = Frame.new_message()
    assert hasattr(frame, 'timestamp')


def test_bbb_relative_path():
    repo = Repository("test_repo")
    assert os.path.isabs(repo.root_dir)


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
