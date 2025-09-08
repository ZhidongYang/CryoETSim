import mrc
import numpy as np


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

