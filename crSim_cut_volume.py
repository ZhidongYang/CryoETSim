import cv2
import mrc
import os
import numpy as np
from skimage.morphology import erosion, ball, skeletonize

import torch
import torch.nn as nn
import torch.nn.functional as F


name = 'cut out data'
help = 'cut volume from full segmentations'

parser = argparse.ArgumentParser(description='cut volume from full segmentations')
parser.add_argument('-i', '--input_volume', help='directory to full segmentations', dest='input_volume')
parser.add_argument('-o', '--output', help='output directory of merged data', dest='output')


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    tomo = mrc.parse(content)
    img = np.array(tomo[0])
    img = img.astype(np.float32)
    return img


def write_mrc(x, path):
    with open(path, 'wb') as f:
        mrc.write(f, x)


def stack_out_tif(path, out_path):
    img_list = os.listdir(path)
    img_stack = []
    for i in range(len(img_list)):
        img_path = os.path.join(path, img_list[i])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_stack.append(img)
        print(i)

    img_tif_stack = np.array(img_stack)
    print(img_tif_stack.shape)
    img_tif_stack = img_tif_stack.astype(np.float32)
    img_tif_stack = (img_tif_stack - img_tif_stack.min()) / (img_tif_stack.max() - img_tif_stack.min())
    write_mrc(img_tif_stack, out_path)


def sampling(tomo, z_scale, y_scale, x_scale, mode='nearest'):
    tomo = torch.from_numpy(tomo)
    tomo = tomo.unsqueeze(0).unsqueeze(0)
    tomo = F.interpolate(tomo, size=(z_scale, y_scale, x_scale),
                         mode=mode)
    tomo = tomo.squeeze(0).squeeze(0)
    tomo = np.array(tomo)
    return tomo


if __name__ == '__main__':
    args = parser.parse_args()
    path = args.input_volume               # segmentations via CDeep3M
    out_path = args.output               # sub volume directory
    img = load_mrc(path)
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min())
    img_new = img[100:245,0:1010,20:1024]               # specify the coordinates in full segmentations to generate sub volume
    write_mrc(img_new, out_path)

