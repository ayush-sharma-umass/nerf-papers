import os, os.path as osp
import numpy as np
import torch
import cv2
from typing import List
from torch.utils.data import Dataset

from src.utils.dataset.common import create_rays


## For this dataset, tn = 8, tf = 12

read_txt = lambda x: np.array(open(x, 'r').readlines(), dtype=float).reshape(4,4)  # we read a txt file--> split it into lines --> parse it as numpy array --> reshape it to 4,4


def make_file_lists(root, mode='train'):
    """
    Makes filepath lists
    :param root: THe root folder of fox dataset
    :param mode: 'train or test'
    :return:
    """
    fimgs = os.listdir(osp.join(root, "imgs"))
    fimgs = [f for f in fimgs if f"{mode}" in f]  # only training images
    fimgs = [osp.join(osp.join(root, "imgs"), f) for f in fimgs if f.endswith('.png')]  # check ig a png image
    fimgs = sorted([osp.abspath(f) for f in fimgs])  # sort in order of id

    fposes = os.listdir(osp.join(root, f'{mode}/pose'))
    fposes = [osp.join(osp.join(root, f'{mode}/pose'), f) for f in fposes if f.endswith('.txt')]
    fposes = sorted([osp.abspath(f) for f in fposes])

    fintrinsics = os.listdir(osp.join(root, f'{mode}/intrinsics'))
    fintrinsics = [osp.join(osp.join(root, f'{mode}/intrinsics'), f) for f in fintrinsics if f.endswith('.txt')]
    fintrinsics = sorted([osp.abspath(f) for f in fintrinsics])
    return fimgs, fposes, fintrinsics


def parse_data(fimgs: List[str],
               fposes: List[str],
               fintrinsics: List[str]):
    """
    Creates arrays of images, poses and intrinsics
    :param fimgs:
    :param fposes:
    :param fintrinsics:
    :return:
    """
    id1 = [im.split("/")[-1].split(".png")[0] for im in fimgs]
    # read an image in BGRA --> remove A channel by multiplying to RGB channels --> normalize the image to -1, 1
    imgs = [cv2.imread(im, cv2.IMREAD_UNCHANGED) for im in fimgs]  ## RGBA
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA) for im in imgs]

    imgs = np.array(imgs) / 255.  # normalizing the image
    if imgs.shape[3] == 4:  ## RGBA --> RGB
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:]) # last addition is done to convert black pixels to white
    N = len(imgs)

    id2 = [f.split("/")[-1].split(".txt")[0] for f in fposes]
    poses = [read_txt(f) for f in fposes]
    poses = np.array(poses)
    assert len(poses) == N, "Images and pose numbers dont match"

    id3 = [f.split("/")[-1].split(".txt")[0] for f in fintrinsics]
    intrinsics = [read_txt(f) for f in fintrinsics]
    intrinsics = np.array(intrinsics)
    assert len(intrinsics) == N, "Images and intrinsics numbers dont match"

    print(imgs.shape, poses.shape, intrinsics.shape)
    for i1, i2, i3 in zip(id1, id2, id3):
        assert i1 == i2 == i3, f"Images,  poses and intrinsics are not properly aligned: \
        Image: {i1}, Pose: {i2}, Intrinsics: {i3}"
    return imgs, poses, intrinsics


def create_fox_dataset(root):
    fimgs, fposes, fintrinsics = make_file_lists(root, 'train')
    imgs, poses, intrinsics = parse_data(fimgs, fposes, fintrinsics)

    # small batch experiment
    N = len(imgs)
    o, d, imgt = [], [], []
    for i in range(N):
        o_, d_ = create_rays(imgs[i], poses[i], intrinsics[i])
        o.append(o_)
        d.append(d_)
        imgt.append(imgs[i].reshape(-1, 3))
    o = np.array(o)
    d = np.array(d)
    dataset = np.concatenate([o.reshape(-1, 3), d.reshape(-1, 3), imgt.reshape(-1, 3)], axis=1).astype(np.float32)
    print(f"Loaded dataset shape: {dataset.shape}, {dataset.dtype}")
    return dataset, imgs, poses, intrinsics


class FoxDataset(Dataset):
    def __init__(self, imgs, poses, intrinsics, mode='train', device='cpu'):
        super().__init__()
        self.imgs = imgs
        self.poses = poses
        self.intrinsics = intrinsics
        self.device = device

        N = len(imgs)
        self.o = []  # origin
        self.d = []  # direction
        self.ims = []  # target pixel value
        for i in range(N):
            o_, d_ = create_rays(imgs[i], poses[i], intrinsics[i])
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
            lbw, ubw = int(W/4), int(3*W/4)
            lbh, ubh = int(H/4), int(3*H/4)
            print(f"Center region: Width {lbw}: {ubw},  Height: {lbh}: {ubh}")
            self.o = self.o.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)
            self.d = self.d.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)
            self.ims = self.ims.reshape(N, H, W, -1)[:, lbw:ubh, lbw:ubw, :].reshape(-1, 3)

        print(f"Final shapes: O:{self.o.shape}, D:{self.d.shape}, Imgs:{self.ims.shape}")

        self.o = torch.from_numpy(self.o).to(device=device)
        self.d = torch.from_numpy(self.d).to(device=device)
        self.ims = torch.from_numpy(self.ims).to(device=device)

        self.data = torch.cat((self.o.reshape(-1, 3),
                               self.d.reshape(-1, 3),
                               self.ims.reshape(-1, 3)), dim=-1).to(device=self.device).float()
        print(f"Final concatenated shape: {self.data.shape}")

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

