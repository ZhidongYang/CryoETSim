import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import rotate, affine
from torchvision.transforms import InterpolationMode
import os
import sys
import argparse
import mrc


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    tomo = mrc.parse(content)
    img = np.array(tomo[0])
    img = img.astype(float)
    return img


def write_mrc(x, path):
    with open(path, 'wb') as f:
        mrc.write(f, x)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--mrc_path', type=str)
    parser.add_argument('-m', '--membrane_path', type=str)
    parser.add_argument('-n', '--num_proteins', type=int, default=50)
    parser.add_argument('-t', '--thickness', type=int, default=300)
    parser.add_argument('-w', '--width', type=int, default=1024)
    parser.add_argument('-e', '--height', type=int, default=1024)
    parser.add_argument('-s', '--save_mrc_path', type=str)
    parser.add_argument('-f', '--save_txt_path', type=str)

    return parser.parse_args()


def generate_random_poses(num_poses, thickness, width, height, radius):
    half_thickness = int(thickness/2)
    center_z = np.full((num_poses, 1), half_thickness)
    center_y = np.random.randint(radius + 20, width - radius, size=(num_poses, 1))
    center_x = np.random.randint(radius + 80, height - radius, size=(num_poses, 1))
    centers = np.hstack((center_z, center_y, center_x))
    angles = np.random.uniform(-180, 180, size=(num_poses, 3))
    return centers, angles


def affine_3d(X, axis, theta, delta, fill=0.0):
    """
    The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
    :param X: the data that should be rotated, a torch.tensor or an ndarray, with lenx * leny * lenz shape.
    :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
    :param fill:  (sequence or number, optional) �CPixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively.
    :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
    :return: rotated tensor.
    """
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
        X = X.float()

    if axis == 0:
        X = affine(X, interpolation=InterpolationMode.NEAREST, angle=theta, translate=delta, scale=1, shear=0)
    elif axis == 1:
        X = X.permute((1, 0, 2))
        X = affine(X, interpolation=InterpolationMode.NEAREST, angle=theta, translate=delta, scale=1, shear=0)
        X = X.permute((1, 0, 2))
    elif axis == 2:
        X = X.permute((2, 1, 0))
        X = affine(X, interpolation=InterpolationMode.NEAREST, angle=theta, translate=delta, scale=1, shear=0)
        X = X.permute((2, 1, 0))
    else:
        raise Exception('Not invalid axis')
    return X.squeeze(0)


def fill_matrix_a(matrix_b, filled_positions, matrix_a, center, radius, thickness, width, height, placed_centers):
    start_x = center[0] - radius
    start_y = center[1] - radius
    start_z = center[2] - radius

    if (start_x < 0 or start_x + radius*2 > thickness or
            start_y < 0 or start_y + radius*2 > width or
            start_z < 0 or start_z + radius*2 > height):
        return False


    non_zero_indices = np.nonzero(matrix_a)

    overlap = np.any(
        filled_positions[start_x + non_zero_indices[0], start_y + non_zero_indices[1], start_z + non_zero_indices[2]])
    if overlap:
        return False

    matrix_b[start_x + non_zero_indices[0], start_y + non_zero_indices[1], start_z + non_zero_indices[2]] += matrix_a[non_zero_indices]

    filled_positions[start_x + non_zero_indices[0], start_y + non_zero_indices[1], start_z + non_zero_indices[2]] = True

    # with open(output_file, "a") as f:
    #     f.write(f"{center[0]}, {center[1]}, {center[2]}\n")
    placed_centers.append(center)

    return True


def main():
    args = get_args()
    subtomo = load_mrc(args.mrc_path)
    subtomo_np = np.array(subtomo)

    membrane = load_mrc(args.membrane_path)
    membrane_np = np.array(membrane)

    radius = int(subtomo_np.shape[0] / 2)
    num_proteins = args.num_proteins
    thickness = args.thickness
    width = args.width
    height = args.height
    half_thickness = int(thickness / 2)
    centers, angles = generate_random_poses(num_proteins, thickness, width, height, radius)

    result = np.zeros((thickness, width, height))
    membrane_non_zero_indices = np.nonzero(membrane_np)

    filled_positions = np.zeros((thickness, width, height), dtype=bool)
    filled_positions[membrane_non_zero_indices] = True
    placed_centers = []

    for center, angle in zip(centers, angles):

        subtomo = load_mrc(args.mrc_path)
        subtomo = affine_3d(subtomo, 0, float(angle[0]), [0.0, 0])
        subtomo = affine_3d(subtomo, 1, float(angle[1]), [0, 0.0])
        subtomo = affine_3d(subtomo, 2, float(angle[2]), [0, 0.0])

        subtomo_np = np.array(subtomo)

        success = False
        while not success:
            success = fill_matrix_a(matrix_b=result, filled_positions=filled_positions, matrix_a=subtomo_np, center=center,
                                    radius=radius, thickness=thickness, width=width, height=height, placed_centers=placed_centers)
            if not success:
                center = np.array([half_thickness - 50, np.random.randint(radius + 20, width - radius), np.random.randint(radius + 80, height - radius)])


        # exit(0)
    result_np = np.array(result)
    result_np = (result_np - result_np.min()) / (result_np.max() - result_np.min())
    membrane_np = (membrane_np - membrane_np.min()) / (membrane_np.max() - membrane_np.min())
    final_np = membrane_np + result_np
    write_mrc(final_np, args.save_mrc_path)
    transformations = np.concatenate((placed_centers, angles), axis=1)
    np.savetxt(args.save_txt_path, transformations, fmt='%d', delimiter='\t')


if __name__ == '__main__':
    main()
