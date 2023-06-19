import os.path as osp
import numpy as np

from src.utils.dataset.common import create_rays

## For this dataset, tn = 2, tf = 6

def create_lego100x100_data(root):
    focal = np.load(osp.join(root, "focal.npy"))
    poses = np.load(osp.join(root, "poses.npy"))
    imgs = np.load(osp.join(root, "images.npy"))

    print(imgs.shape, poses.shape)
    o, d, imgt = [], [], []
    N = len(imgs)
    for i in range(N):
        o_, d_ = create_rays(imgs[i], poses[i], focal)
        o.append(o_)
        d.append(d_)
        imgt.append(imgs[i].reshape(-1, 3))

    o = np.array(o)
    d = np.array(d)
    imgt = np.array(imgt)
    dataset = np.concatenate([o.reshape(-1, 3), d.reshape(-1, 3), imgt.reshape(-1, 3)], axis=1).astype(np.float32)
    print(f"Loaded dataset shape: {dataset.shape}, {dataset.dtype}")
    return dataset, imgs, poses, focal