__author__ = 'Zhidong Yang'

import os
import argparse
import mrc
import tem
import shutil

import numpy as np

name = 'tilt_reprojection'
help = 'reproject the density into tilt series'

parser = argparse.ArgumentParser(description='reproject the density into tilt series')
parser.add_argument('-d', '--directory', help='directory to density', dest='directory')
parser.add_argument('-o', '--output', help='output directory of tilt series', dest='output')
parser.add_argument('-a', '--angle', help='tilt angles', dest='angle')
parser.add_argument('-i', '--interval', help='interval of each tilt angle', dest='interval')


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    tomo, _, _ = mrc.parse(content)
    tomo = tomo.astype(np.float32)
    tomo = np.swapaxes(tomo, 0, 2)
    return tomo


def write_mrc(x, path):
    with open(path, 'wb') as f:
        mrc.write(f, x)
    return


def main(directory, output, angle, interval):

    # Common tomogram settings
    VOI_VSIZE = 12 # 2.2 # A/vx

    # Reconstruction tomograms
    step_interval = int(interval)
    min_angle = -1*int(angle)
    max_angle = int(angle)
    TILT_ANGS = range(min_angle, max_angle, step_interval)

    # OUTPUT FILES
    OUT_DIR = output
    TEM_DIR = OUT_DIR + '/tem'
    TOMOS_DIR = os.path.join(OUT_DIR, 'tomos')
    if not os.path.exists(TOMOS_DIR):
        os.makedirs(TOMOS_DIR)

    # TEM for 3D reconstructions
    temic = tem.TEM(TEM_DIR)
    vol = load_mrc(directory)
    temic.gen_tilt_series_imod(vol, TILT_ANGS, ax='Y')
    temic.invert_mics_den()
    # temic.add_detector_noise(1.0)
    temic.set_header(data='mics', p_size=(VOI_VSIZE, VOI_VSIZE, VOI_VSIZE), origin=(0, 0, 0))
    out_mics = TOMOS_DIR + '/mics_tilt_noiseless_' + str(angle) + '_interval_' + str(step_interval) + '.mrc'
    out_tilts = TOMOS_DIR + '/tilts_' + str(angle) + '_interval_' + str(step_interval) + '.tlt'
    shutil.copyfile(TEM_DIR + '/out_micrographs.mrc', out_mics)
    shutil.copyfile(TEM_DIR + '/out_tangs.tlt', out_tilts)


if __name__ == '__main__':

    args = parser.parse_args()
    directory = args.directory
    output = args.output
    angle = args.angle
    interval = args.interval
    main(directory=directory, output=output, angle=angle, interval=interval)

# python crSim_tilt_simulator.py -d ./results_density/6z6u_with_membranes.mrc -o ./output_tilts -a 50 -i 3