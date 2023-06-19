import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import cv2
from typing import List, Union
from torch.utils.data import Dataset


def create_rays(im: np.ndarray,
                P: np.ndarray,
                I: Union[np.ndarray, float],
                debug=False):
    """
    for a single image, it creates a tensor of rays
    :param im: H, W, 3
    :param P: 4, 4
    :param I: 4, 4 else if I is float, I is focal length
    :param debug: flag
    :return: (W * H, 3), (W * H, 3)
    """

    H, W, C = im.shape
    if isinstance(I, np.ndarray) and I.shape == (4,4):
        fx, fy, cx, cy = I[0, 0], I[1, 1], I[0, 2], I[1, 2]
    else:
        fx = fy = I
        cx = cy = 0.

    # create a pixel grid
    u = np.arange(0, W)
    v = np.arange(0, H)
    u, v = np.meshgrid(u, v)
    if debug:
        print(f"Pose: {P.shape}  Intrinsics: {I.shape}")
        print(u.shape, v.shape)
        print(H, W, C, fx, fy, cx, cy)
    rays_o = np.zeros((W * H, 3))
    rays_d = np.stack([(u - W / 2 - 0) / fx,
                       -(v - H / 2 - 0) / fy,
                       - np.ones_like(u)]
                      , axis=-1)

    # Move the camera to world frame
    # P is camera2world
    rays_d = (P[:3, :3] @ rays_d[..., None]).squeeze(-1)
    rays_o += P[:3, 3]

    # normalize
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_d = rays_d.reshape(-1, 3)
    if debug:
        print(f"o: {rays_o.shape} d: {rays_d.shape}")
    return rays_o, rays_d