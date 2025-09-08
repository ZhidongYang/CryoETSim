import os
import mrc
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from skimage.morphology import closing, opening, erosion, ball, skeletonize, dilation
from skimage.filters import sobel


name = 'optimize the membrane structures with morphological algorithm'
help = 'optimize the shape of segmented membrane with morphological algorithm'


parser = argparse.ArgumentParser(description='optimize the membrane structures with morphological algorithm')
parser.add_argument('-m', '--membranes', help='directory to segmented membranes', dest='membranes')

parser.add_argument('-r', '--threshold_pre_value',
                    type=float,
                    default=0.4,
                    help='remove the redundant voxels before mophological algorithm (default: 0.4), '
                         '0 means thresholding is not needed',
                    dest='threshold_pre_value')

parser.add_argument('-b', '--ball_size',
                    type=int,
                    default=1,
                    help='ball size for morphological algorithm (default: 1)',
                    dest='ball_size')

parser.add_argument('-u', '--upsampling_size',
                    type=int,
                    default=400,
                    help='z thickness for upsampling (default: 400) '
                         '0 means thresholding is not needed',
                    dest='upsampling_size')

parser.add_argument('-p', '--threshold_post_value',
                    type=float,
                    default=0.17,
                    help='remove the redundant voxels after sobel (default: 0.17), '
                         '0 means thresholding is not needed',
                    dest='threshold_post_value')

parser.add_argument('-o', '--output', help='output directory of optimized membranes', dest='output')


def sampling(tomo, z_scale, y_scale, x_scale, mode='nearest'):
    tomo = torch.from_numpy(tomo)
    tomo = tomo.unsqueeze(0).unsqueeze(0)
    tomo = F.interpolate(tomo, size=(z_scale, y_scale, x_scale),
                         mode=mode)
    tomo = tomo.squeeze(0).squeeze(0)
    tomo = np.array(tomo)
    return tomo


if __name__ == '__main__':
    # import parameters as input arguements
    args = parser.parse_args()
    membranes = args.membranes
    threshold_pre_value = args.threshold_pre_value
    ball_size = args.ball_size
    upsampling_size = args.upsampling_size
    threshold_post_value = args.threshold_post_value
    output = args.output

    # load data as numpy array, normalize the image into 0,1
    print('load membranes...')
    img_membranes = load_mrc(membranes)
    img_membranes = np.array(img_membranes)
    img_membranes = (img_membranes - img_membranes.min()) / (img_membranes.max() - img_membranes.min())

    # upsampling for the image, if the resolution of z-axis of the image is too low (optional)
    if upsampling_size > 0:
        print('up sampling on z axis...')
        img_membranes = sampling(img_membranes,
                                      z_scale=upsampling_size,
                                      y_scale=img_membranes.shape[1],
                                      x_scale=img_membranes.shape[2])

    # pre-thresold for removing the redundant voxels (optional)
    if threshold_pre_value > 0:
        img_membranes[img_membranes <= threshold_pre_value] = 0

    # morphological analysis
    print('morphological processing for segmented structures...')
    footprint_a = ball(ball_size)
    img_membranes = (img_membranes - img_membranes.min()) / (img_membranes.max() - img_membranes.min())
    eroded = erosion(img_membranes, footprint_a)
    eroded_grad = sobel(eroded)
    if threshold_post_value > 0:
        eroded_grad[eroded_grad <= threshold_post_value] = 0
    eroded_grad = (eroded_grad - eroded_grad.min()) / (eroded_grad.max() - eroded_grad.min())

    print('save the membranes...')
    if not os.path.exists('./results_density'):
        os.makedirs('./results_density')

    out_path = os.path.join('./results_density', output)
    write_mrc(eroded_grad, out_path)


# python crSim_membranes_optimization.py -m ./example_data/example_membrane_01.mrc -r 0.4 -b 1 -u 300 -p 0.17 -o membranes.mrc