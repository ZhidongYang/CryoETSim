import os
import sys
import time
import random
import mrc
import argparse

import numpy as np

from utils import *

name = 'noise simulator'
help = 'add complex noise to tilt series'

parser = argparse.ArgumentParser(description='add complex noise to tilt series')
parser.add_argument('-i', '--input_dir', help='directory to tilt series without noise degradation', dest='input_dir')
parser.add_argument('-o', '--output', help='output directory of noisy tilt series', dest='output')
parser.add_argument('-s', '--snr', help='snr of applied noise', dest='snr')
parser.add_argument('-k', '--skewness', help='skewness of detector noise', dest='skewness')
parser.add_argument('-n', '--noiseion', help='learned noise for ions', dest='noiseion')


def add_detector_noise(clean_image_path, noise_image_path, snr, skewness, output):
    """
    Add noise to the clean image.
    :param clean_image_path: Path to clean image
    :param noise_image_path: Path to noise image
    :param output: Path to output image
    :param snr: Signal-to-Noise Ratio (dB)
    :return: Noisy image
    """

    use_detector_noise = 1

    # Import clean image and noise image
    print('load tilt series...')
    clean_image = load_mrc(clean_image_path)
    noise_image = load_mrc(noise_image_path)

    clean_image = (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
    noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())

    assert(clean_image.shape == noise_image.shape)

    print('compute noise multipliers...')
    clean_energy = np.sum(clean_image ** 2)
    noise_energy = np.sum(noise_image ** 2)

    # Calculate the noise multiplier based on SNR
    noise_multiplier = np.sqrt(clean_energy / (noise_energy * 10 ** (snr / 10)))
    noise_multiplier *= np.exp(-(abs(snr)) / 2)

    if noise_multiplier >= 1.1:
        print('rescale is needed')
        noise_multiplier = 1
        # use_detector_noise = 1

    # Add noise to the clean image
    mn = clean_image.mean()
    sg_fg = mn / snr
    detector_noise = use_detector_noise * skewness * np.random.normal(0, sg_fg, clean_image.shape).astype(np.float32)
    noisy_image = clean_image + noise_image * noise_multiplier + detector_noise
    noisy_image = ((noisy_image - noisy_image.min()) / (noisy_image.max() - noisy_image.min()))

    print('save the noisy tilt series...')
    if not os.path.exists('./results_tilt'):
        os.makedirs('./results_tilt')

    out_path = os.path.join('./results_tilt', output)
    # out_path = output
    write_mrc(noisy_image, out_path)


if __name__ == '__main__':

    args = parser.parse_args()
    input_dir = args.input_dir
    output = args.output
    snr = float(args.snr)
    skewness = float(args.skewness)
    noiseion = args.noiseion
    add_detector_noise(clean_image_path=input_dir, noise_image_path=noiseion, snr=snr, skewness=skewness, output=output)


# python crSim_apply_noise_to_tilts_snr.py -i ./results_tilt/tilts_applied_ctf.mrc \
#                                 -o tilts_applied_ctf_noise_snr1.mrc \
#                                 -s 1.0 \
#                                 -k 0.15 \
#                                 -n ./example_noise_tilts/bin2_noise_50_angle_3_interval.mrc

