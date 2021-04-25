import os
import torch
import json
import numpy as np
import cv2
import libs.other_helpers as u
from loaders.load_blender import pose_spherical


def load_custom_data(basedir, half_res=False, testskip=1):
    data = u.read_json(basedir + '/poses.json')
    poses = u.get_transforms(data)
    images = u.images_to_arr(basedir + '/imgs/')
    images, poses = u.unison_shuffled_copies(images, poses)

    H, W = images[0].shape[:2]
    fov = u.degrees_to_radians(45)
    focal = .5 * W / np.tan(.5 * fov)

    counts = [0, int(images.shape[0] * 0.8), int(images.shape[0] * 0.9), images.shape[0]]
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    # i_split = [[0], [0], [0]]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((images.shape[0], H, W, 4))
        for i, img in enumerate(images):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        images = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return images, poses, render_poses, [H, W, focal], i_split
