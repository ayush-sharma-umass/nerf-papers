import os, os.path as osp
import numpy as np
import torch
import cv2
from typing import List, Union
from torch.utils.data import Dataset
from tqdm import tqdm


def convert_poses_to_4x4_format(inp_poses, N):
    if inp_poses.shape[-1] == N:
        inp_poses = inp_poses.transpose(0, -1)
    c2w4x4 = []
    int4x4 = []
    metas = []
    for i in range(N):
        p = inp_poses[i, :-2].reshape([3, 5])
        whf = p[:, -1]
        p = p[:, :-1]
        imat = np.zeros((4, 4))
        imat[0, 0] = imat[1, 1] = whf[-1]
        imat[-1, -1] = 1
        p = np.concatenate([p, np.array([[0, 0, 0, 1]])], axis=0)
        meta = whf[:-1]

        c2w4x4.append(p)
        int4x4.append(imat)
        metas.append(meta)

    return np.array(c2w4x4), np.array(int4x4), np.array(metas)


def read_files(root: str,
               objs: Union[str, List[str]],
               res=8,
               poses_file="poses_bounds.npy"):
    def read_one(root_obj, res, poses_file):

        dimgs = osp.join(root_obj, f"images_{res}")
        fimgs = []
        for r, d, fs in os.walk(dimgs):
            fimgs = [osp.join(r, f) for f in fs if f[-4:] == ".png"]
        n = len(fimgs)
        imgs = []
        for i in range(n):
            im = cv2.imread(fimgs[i], cv2.IMREAD_UNCHANGED)
            if im.shape[-1] == 4:  # BGRA
                im = im[..., :3] * im[..., -1:]  # multiply the alpha channel with BGR
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im/255.)

        poses_arr = np.load(osp.join(root_obj, poses_file))
        poses, imats, metas = convert_poses_to_4x4_format(poses_arr, n)
        return np.array(imgs), poses, imats, (metas/ float(res)).astype(int)

    data_obj = {}
    if isinstance(objs, str):
        objs = [objs]
    for obj in objs:
        imgs, poses, imats, metas = read_one(osp.join(root, obj), res, poses_file)
        data_obj[obj] = imgs, poses, imats, metas

    return data_obj


def create_rays(im: np.ndarray,
                P: np.ndarray,
                I: np.ndarray,
                meta: np.ndarray,
                debug=False):
    """
    for a single image, it creates a tensor of rays
    :param im: H, W, 3
    :param P: 4, 4
    :param I: 4, 4
    :param meta: 2,
    :param debug: flag
    :return: (W * H, 3), (W * H, 3)
    """
    _, _, C = im.shape
    H, W = int(meta[0]), int(meta[1])
    fx, fy, cx, cy = I[0, 0], I[1, 1], I[0, 2], I[1, 2]
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


class LLFFDataset(Dataset):

    def __init__(self, imgs, poses, imats, metas, mode='train', device="cpu"):
        super().__init__()
        assert len(imgs) == len(poses) == len(imats) == len(metas)
        N = len(imgs)
        self.device = device
        self.o = []  # origin
        self.d = []  # direction
        self.ims = []  # target pixel value
        for i in tqdm(range(N)):
            o_, d_ = create_rays(imgs[i], poses[i], imats[i], metas[i])
            self.o.append(o_)
            self.d.append(d_)
            self.ims.append(imgs[i])

        self.o = np.array(self.o)
        self.d = np.array(self.d)
        self.ims = np.array(self.ims)
        _, H, W, C = self.ims.shape
        if mode == 'warm-up':
            """
            In warm-up, for synthetic datasets, we want to pass the
            """
            lbw, ubw = int(W / 4), int(3 * W / 4)
            lbh, ubh = int(H / 4), int(3 * H / 4)
            print(f"Center region: Width {lbw}: {ubw},  Height: {lbh}: {ubh}")
            self.o = self.o.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)
            self.d = self.d.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)
            self.ims = self.ims.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)

        print(f"Final shapes: O:{self.o.shape}, D:{self.d.shape}, Imgs:{self.ims.shape}")

        self.o = torch.from_numpy(self.o)
        self.d = torch.from_numpy(self.d)
        self.ims = torch.from_numpy(self.ims)

        self.data = torch.cat((self.o.reshape(-1, 3),
                               self.d.reshape(-1, 3),
                               self.ims.reshape(-1, 3)), dim=-1).float()
        print(f"Final concatenated shape: {self.data.shape}")

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample.to(device=self.device)

