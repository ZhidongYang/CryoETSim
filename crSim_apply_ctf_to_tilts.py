import os
import mrc
import argparse
import numpy as np
import pyfftw.interfaces.numpy_fft as fft
from ctf import *
from tqdm import tqdm
from utils import *
from numpy.fft import *


name = 'apply ctf to tilts'
help = 'apply simulated ctf to cryo-et tilts'

parser = argparse.ArgumentParser(description='apply simulated ctf to cryo-et tilts')
parser.add_argument('-t', '--tiltpath', help='directory to tilts', dest='tiltpath')
parser.add_argument('-o', '--output', help='directory to tilts applied ctf', dest='output')
parser.add_argument('-d1', '--defocus1', help='the first bound of defocus value', dest='def1')
parser.add_argument('-d2', '--defocus2', help='the second bound of defocus value', dest='def2')
parser.add_argument('-an', '--angleast', help='shift angle on x axis', dest='angast')
parser.add_argument('-kv', '--kvoltage', help='operation voltage of tem', dest='kv')
parser.add_argument('-ac', '--accum', help='operation voltage of tem', dest='ac')
parser.add_argument('-cs', '--sphereabbr', help='spherical abbrevation', dest='cs')


def ctf_modulation(tilt_path, output, def1=1500, def2=1800, angast=1, kv=300, ac=0.1, cs=2.7):

    p_tilts = load_mrc(tilt_path)
    num_tilts = p_tilts.shape[0]
    ctf_applied_tilts = []
    for i in tqdm(range(num_tilts)):
        s, a = ctf_freq(shape=p_tilts[i].shape, d=angast, full=True)
        c = eval_ctf(s, a, def1, def2, angast, kv=kv, ac=ac, cs=cs, bf=100)   # size 1024
        fft_c = fftshift(c)
        proj_ctf = np.real(ifftn(ifftshift(fft_c * fftshift(fftn(p_tilts[i])))))
        ctf_applied_tilts.append(proj_ctf)
    np_ctf_applied_tilts = np.array(ctf_applied_tilts)

    print('save the ctf modulated tilt series...')
    if not os.path.exists('./results_tilt'):
        os.makedirs('./results_tilt')

    out_path = os.path.join('./results_tilt', output)
    write_mrc(np_ctf_applied_tilts, out_path)


if __name__ == '__main__':

    args = parser.parse_args()
    tiltpath = args.tiltpath
    output = args.output
    def1 = int(args.def1)
    def2 = int(args.def2)
    angast = np.float32(args.angast)
    kv = int(args.kv)
    ac = np.float32(args.ac)
    cs = np.float32(args.cs)
    ctf_modulation(tiltpath, output, def1, def2, angast, kv, ac, cs)


# python crSim_apply_ctf_to_tilts.py -t ./example_data/tilts_clean_50_angles_3_interval.mrc -o tilts_applied_ctf.mrc -d1 1000 -d2 1300 -an 1 -kv 300 -ac 0.1 -cs 2.7