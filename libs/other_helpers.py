from math import pi
import json
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def unit_vector(a, dim=None):
    '''
    Returns the unit vector with the direction of a
    :param dim:
    :param a:
    :return:
    '''
    return a / a.norm(dim=dim).unsqueeze(-1)


def unit_vector_np(a, axis=None):
    return a / np.linalg.norm(a, axis=axis)[..., np.newaxis]


def random_on_unit_sphere(size, device='cpu'):
    # We use the method in https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    # to produce vectors on the surface of a unit sphere

    x = torch.randn(size)
    l = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(1)
    x = (x / l).to(device)

    return x


def degrees_to_radians(d):
    if type(d) == str:
        d = eval(d)
    if type(d) == list or type(d) == tuple:
        return [x * pi / 180. for x in d]
    return d * pi / 180.


def read_json(src):
    try:
        with open(src, "r") as read_file:
            data = json.load(read_file)
    except FileNotFoundError:
        data = None
    return data


def get_transforms(data):
    if len(data['frames']) == 0:
        return None
    lookat = np.array([0., 0., 0.])
    vup = np.array([0., 1., 0.])
    p = []

    for f in data['frames']:
        p.append(get_transform_matrix(f['cam_pos'], lookat, vup))
    return np.array(p)


def images_to_arr(folder):
    filelist = os.listdir(folder)
    # x = np.array([np.array(Image.open(folder + fname)) for fname in filelist])
    x = []
    for fname in filelist:
        img = Image.open(folder + fname)
        # img = img.resize((100, 100), Image.ANTIALIAS)
        x.append(np.array(img) / 255.)
    x = np.array(x)

    return x


def plot_dir(r):
    if type(r) != torch.Tensor:
        r = r.directions
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = r.cpu().numpy()
    ax.scatter(r[:, 0], r[:, 1], r[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def get_transform_matrix(eye, center, up, final_r=4):
    # Renormalizing the camera position first
    eye = np.array(eye)
    r = np.linalg.norm(eye, axis=-1)
    eye = eye / r * final_r

    zaxis = unit_vector_np(eye - center)
    xaxis = unit_vector_np(np.cross(up, zaxis))
    yaxis = np.cross(zaxis, xaxis)

    transform_matrix = np.eye(4)
    transform_matrix[:-1, 0] = xaxis
    transform_matrix[:-1, 1] = yaxis
    transform_matrix[:-1, 2] = zaxis
    transform_matrix[:-1, 3] = eye
    return transform_matrix


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == '__main__':
    eye = np.array((-4., -4.0, -0.5))
    center = np.array((0, 0, 0))
    up = np.array((0, 0, 1))
    print(get_transform_matrix(eye, center, up))