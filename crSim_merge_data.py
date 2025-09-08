import os
import mrc
import argparse

import numpy as np

from utils import *

name = 'merge data'
help = 'merge the proteins to membranes'

parser = argparse.ArgumentParser(description='merge the proteins to membranes')
parser.add_argument('-m', '--membranes', help='directory to optimized membranes', dest='membranes')
parser.add_argument('-p', '--proteins', help='generated sample of proteins', dest='proteins')
parser.add_argument('-o', '--output', help='output directory of merged data', dest='output')


if __name__ == '__main__':
    args = parser.parse_args()
    den_path_membranes = args.membranes
    den_path_particles = args.proteins
    output = args.output
    img_membranes = load_mrc(den_path_membranes)
    img_membranes = np.array(img_membranes)
    img_membranes = (img_membranes - img_membranes.min()) / (img_membranes.max() - img_membranes.min())
    img_particles = load_mrc(den_path_particles)
    img_particles = np.array(img_particles)
    img_particles = (img_particles - img_particles.min()) / (img_particles.max() - img_particles.min())
    img_merged = img_particles + img_membranes
    img_merged = (img_merged - img_merged.min()) / (img_merged.max() - img_merged.min())

    if not os.path.exists('./results_density'):
        os.makedirs('./results_density')

    out_path = os.path.join('./results_density', output)
    write_mrc(img_merged, out_path)


# python crSim_merge_data.py -m ./results_density/membranes.mrc -p ./example_data/6z6u_examples.mrc -o 6z6u_with_membranes.mrc