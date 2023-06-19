import os, os.path as osp
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.dataset.lego_dataset import create_lego100x100_data
from src.utils.dataset.common import create_rays
from src.utils.bookkeeping import generate_new_session
from src.utils.render import render, batchify_image_and_render
from src.lf_model.fourier_nerf import FourierNerf
from src.utils.metrics import get_mse_and_psnr


## Config
obj = "lego100x100"
model_name = "fourier_nerf"
device = 'cuda'
torch.backends.cuda.max_split_size_mb = 256

## Model Hyper-params
NF_feats=256
B_scale=6.05
hidden_dim = 128

## Rendering params
tn = 2.
tf = 6.
num_bins = 100
directional_input = False
white_bgr = False

## Training
batch_size = 1024
num_epochs= 15
warmup_epochs = 1
lr = 1e-3


def test(session,
         model,
         test_ids,
         imgs, poses, intrinsics,
         tn,
         tf,
         num_bins,
         directional_input,
         white_bgr,
         epoch):
    with torch.no_grad():
        for tid in test_ids:
            if len(intrinsics.shape) == 3:
                I = intrinsics[i]
            else:
                I = intrinsics
            o_, d_ = create_rays(imgs[tid], poses[tid], I)
            im  = imgs[tid]
            H, W, _ = im.shape
            Ax = batchify_image_and_render(model=model,
                                   rays_o=o_,
                                   rays_d=d_,
                                   tn=tn,
                                   tf=tf,
                                   num_bins=num_bins,
                                   reduction_factor=16,
                                   device=device,
                                   directional_input=directional_input,
                                   white_bgr=white_bgr)
            Ax = Ax.reshape(H, W, 3)
            psnr, mse = get_mse_and_psnr(im, Ax)
            print(f"Test IDs: {tid}       | PSNR: {psnr: .3f} | MSE :{mse: .3f}")
            Ax = (Ax.reshape(H, W, 3) * 255).astype(np.uint8)
            imdir = osp.join(session.sess_dir, f"epoch_{epoch}")
            if not osp.exists(imdir):
                os.makedirs(imdir)
            impath = osp.join(imdir, f"epoch-{epoch}--image-{tid}.png")
            cv2.imwrite(impath, cv2.cvtColor(Ax, cv2.COLOR_RGB2BGR))


def train(sess,
          model,
          dataset,
          imgs, poses, intrinsics,
          test_ids,
          tn,
          tf,
          num_bins,
          directional_input=True,
          white_bgr=True):
    dataloader_train = DataLoader(torch.from_numpy(dataset).to(device=device), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[5, 10, 15], gamma=0.5)

    loss_history = []
    for e in range(num_epochs):
        avg_loss = 0.
        for batch in tqdm(dataloader_train):
            BS = batch.shape[0]
            batch = batch.reshape(-1, 9)
            o = batch[:, 0:3]
            d = batch[:, 3:6]
            c = batch[:, 6:]
            Ax = render(model, o, d, tn, tf, num_bins,
                        directional_input=directional_input, device=device,
                        white_bgr=white_bgr)
            loss = ((Ax - c) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_history.append(loss.item())
            avg_loss += loss.item()
        print(f"Epoch: {e}: avg_loss: {avg_loss / len(dataloader_train)}")
        scheduler.step()
        modelpath = osp.join(sess.sess_dir, f'model_{obj}-epoch{e + 1}.pth')
        torch.save(model.cpu(), modelpath)
        model.to(device)
        test(sess, model, test_ids, imgs, poses,
             intrinsics, tn, tf, num_bins, directional_input, white_bgr, e)


def main():
    root = f"data/{obj}"
    dataset, imgs, poses, intrinsics = create_lego100x100_data(root)
    N = len(imgs)
    test_ids = np.random.randint(low=0, high=N, size=(5,))
    session = generate_new_session(f"/home/ayush/projects/{model_name}/model_checkpoints/", obj)
    model = FourierNerf(NF_feats=NF_feats, B_scale=B_scale, hidden_dim=hidden_dim, device=device)
    train(session, model, dataset, imgs, poses, intrinsics, test_ids, tn, tf, num_bins, directional_input, white_bgr)



if __name__ == '__main__':
    main()
