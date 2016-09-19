from __future__ import print_function

import matplotlib
matplotlib.use('agg')  # noqa

import numpy as np
from bb_binary import int_id_to_binary, parse_video_fname, \
    convert_frame_to_numpy, parse_image_fname, load_frame_container
from pipeline.io import raw_frames_generator
from pipeline.stages.visualization import ResultCrownVisualizer
from pipeline.stages.processing import Localizer
from diktya.numpy import tile

import matplotlib.pyplot as plt
from deepdecoder.data import HDF5Dataset
import click
import os
from scipy.misc import imread, imsave


class GTPeriod:
    def __init__(self, camIdx, start, end, filename, frames):
        self.camIdx = camIdx
        self.start = start
        self.end = end
        self.filename = filename
        self.frames = frames

    def pickle_filename(self):
        basename, ext = os.path.splitext(os.path.basename(self.filename))
        return basename + ".pickle"


def get_subdirs(dir):
    return listdir(dir, os.path.isdir)


def get_files(dir):
    return listdir(dir, os.path.isfile)


def listdir(dir, select_fn):
    return [os.path.join(dir, d) for d in os.listdir(dir)
            if select_fn(os.path.join(dir, d))]


def extract_gt_rois(data, image, date, roi_size=128):
    if date.year == 2014:
        pos = np.stack([data['ypos'], data['xpos']], axis=1)
    else:
        pos = np.stack([data['xpos'], data['ypos']], axis=1)

    rois, mask = Localizer.extract_rois(pos, image, roi_size)
    return rois, mask, pos


class FrameGeneratorFactory():
    def __init__(self, video_dir, image_dir):
        self.videos = get_files(video_dir)
        self.image_dirs = get_subdirs(image_dir)
        self.videos_dict = {parse_video_fname(video)[:2]: video for video in self.videos}
        self.image_dirs_dict = {}
        for dir in self.image_dirs:
            first_frame = sorted(get_files(dir))[0]
            parsed = parse_image_fname(first_frame)
            self.image_dirs_dict[parsed] = dir

    def get_generator(self, camIdx, startts):
        try:
            fname = self.videos_dict[(camIdx, startts)]
            for x in raw_frames_generator(fname):
                yield x, fname
        except:
            img_dir = self.image_dirs_dict[(camIdx, startts)]
            for img_fname in sorted(get_files(img_dir)):
                yield imread(img_fname), img_fname


def visualize_detections(gt, position, image):
    crown_vis = ResultCrownVisualizer()
    n = len(gt['tags'])
    overlay = crown_vis(image, position, np.zeros((n, 3)), gt['bits'])[0]
    img_overlay = crown_vis.add_overlay(image / 255, overlay)
    plt.imshow(img_overlay)
    plt.show()


def append_gt_to_hdf5(gt_period, dset):
    for gt_frame in gt_period.frames:
        h5_frame = {k: v for k, v in gt_frame.items() if type(v) == np.ndarray}
        dset.append(**h5_frame)


@click.command("bb_gt_to_hdf5")
@click.option('--bb-gt_files', '-g', multiple=True)
@click.option('--video-dir', '-v')
@click.option('--image-dir', '-i')
@click.option('--visualize-debug', is_flag=True)
@click.option('--force', is_flag=True)
@click.help_option()
@click.argument('output')
def run(bb_gt_files, video_dir, image_dir, visualize_debug, force, output):
    """
    Converts bb_binary ground truth Cap'n Proto files to hdf5 files and
    extracts the corresponding rois from videos or images.
    """
    gen_factory = FrameGeneratorFactory(video_dir, image_dir)
    if force and os.path.exists(output):
        os.remove(output)
    dset = HDF5Dataset(output)
    camIdxs = []
    periods = []
    for fname in bb_gt_files:
        fc = load_frame_container(fname)
        camIdx, start_dt, end_dt = parse_video_fname(fname)
        basename = os.path.basename(fname)
        gt_frames = []
        print(basename)
        gen = gen_factory.get_generator(camIdx, start_dt)
        first = True
        for frame, (video_frame, video_filename) in zip(fc.frames, gen):
            gt = {}
            np_frame = convert_frame_to_numpy(frame)
            rois, mask, positions = extract_gt_rois(np_frame, video_frame, start_dt)
            for name in np_frame.dtype.names:
                gt[name] = np_frame[name][mask]
            gt["bits"] = np.array([int_id_to_binary(id)[::-1] for id in gt["decodedId"]])
            gt["tags"] = rois
            gt['filename'] = os.path.basename(video_filename)
            gt_frames.append(gt)
            if first and visualize_debug:
                visualize_detections(gt, positions, video_frame)
                first = False

            print('.', end='')
        gt_period = GTPeriod(camIdx, start_dt, end_dt, fname, gt_frames)

        periods.append([int(gt_period.start.timestamp()), int(gt_period.end.timestamp())])
        camIdxs.append(gt_period.camIdx)
        append_gt_to_hdf5(gt_period, dset)

    dset.attrs['periods'] = np.array(periods)
    dset.attrs['camIdxs'] = np.array(camIdxs)
    dset.close()


def visualize_detection_tiles(dset, n=12**2):
    crown_vis = ResultCrownVisualizer()
    imgs = []
    indicies = np.arange(len(dset['tags']))
    np.random.shuffle(indicies)
    for i in range(n):
        p = indicies[i]
        position = np.array([dset['tags'][0, 0].shape]) / 2 + 140
        tag = dset['tags'][p, 0] / 255
        overlay = crown_vis(tag, position, np.zeros((1, 3)), dset['bits'][p:p+1])[0]
        imgs.append(crown_vis.add_overlay(tag, overlay))
    tiled = tile([img.swapaxes(0, -1) for img in imgs])
    imsave('visualize_debug_detections.png', tiled.swapaxes(0, -1))
