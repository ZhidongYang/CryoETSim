# -*- coding:utf-8 -*
import os
import sys
import random
import mrcfile
import operator
import argparse

import numpy as np

from tqdm import tqdm
from scipy.optimize import leastsq


name = 'noise simulator with re-weighting'
help = 'add complex noise to tilt series, according to realistic micrographs'

parser = argparse.ArgumentParser(description='add complex noise to tilt series with reweighting')
parser.add_argument('-c', '--clean_dir', help='directory to tilt series without noise degradation', dest='clean_dir')
parser.add_argument('-n', '--noise_dir', help='directory to noise micrographs', dest='noise_dir')
parser.add_argument('-r', '--real_dir', help='directory to real micrographs', dest='real_dir')
parser.add_argument('-o', '--output_dir', help='directory to reweighted simulation micrographs', dest='output_dir')


def rescale_to_fixedrange(img):
    img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0
    return img


def func(p, x, y, noise_mean):
    k1, k2, b = p
    return k1*x+k2*(y-noise_mean)+b


def error(p, x, y, noise_mean, z):
    return func(p, x, y, noise_mean) - z


def fit_noise(clean, noise, noisy):
    clean_data = np.array([i for j in clean for i in j], dtype=np.float64)
    noise_data = np.array([i for j in noise for i in j], dtype=np.float64)
    noisy_data = np.array([i for j in noisy for i in j], dtype=np.float64)
    noise_mean = np.mean(noise_data)
    p0 = [100, 100, 2]
    params = leastsq(error, p0, args=(clean_data, noise_data, noise_mean, noisy_data))
    k1, k2, b = params[0]

    return k1, k2, b


def noise_reweighting(clean_dir, noise_dir, real_dir, output_dir):

    clean_file = mrcfile.open(clean_dir, permissive=True)
    noise_file = mrcfile.open(noise_dir, permissive=True)
    real_file = mrcfile.open(real_dir, permissive=True)

    clean_stack = clean_file.data
    noise_stack = noise_file.data
    real_stack = real_file.data

    clean_stack = rescale_to_fixedrange(clean_stack)
    noise_stack = rescale_to_fixedrange(noise_stack)
    real_stack = rescale_to_fixedrange(real_stack)

    fit_result = np.zeros(clean_stack.shape)

    num_of_tilts = noise_stack.shape[0]

    for cnt in tqdm(range(0, num_of_tilts)):
        # print('Noise reweighting for tilt ' + str(cnt))
        clean_micrograph = clean_stack[cnt, :, :]
        noise_micrograph = noise_stack[cnt, :, :]
        real_micrograph = real_stack[cnt, :, :]
        k1, k2, b = fit_noise(clean_micrograph, noise_micrograph, real_micrograph)
        # print('k1, k2: {}, {}'.format(k1, k2))
        fit_result[cnt, :, :] = (k1 + 1) * clean_micrograph + (k2 + 1) * (noise_micrograph - np.mean(noise_micrograph))

    print('save the noisy tilt series...')
    if not os.path.exists('./results_tilt'):
        os.makedirs('./results_tilt')

    out_path = os.path.join('./results_tilt', output_dir)
    fit_out = mrcfile.new(out_path, overwrite=True)
    fit_result = (fit_result - np.min(fit_result)) / (np.max(fit_result) - np.min(fit_result))
    fit_out.set_data(fit_result.astype(np.float32))


if __name__ == '__main__':

    args = parser.parse_args()
    clean_dir = args.clean_dir
    noise_dir = args.noise_dir
    real_dir = args.real_dir
    output_dir = args.output_dir
    noise_reweighting(clean_dir, noise_dir, real_dir, output_dir)


# python crSim_apply_noise_to_tilts_reweighting.py -c ./results_tilt/tilts_applied_ctf.mrc \
#                                            -n ./example_noise_tilts/bin2_noise_50_angle_3_interval.mrc \
#                                            -r ./example_data/tilts_real_50_angles_3_interval.mrc \
#                                            -o tilts_applied_ctf_noise_reweighted.mrc
