import numpy as np


def get_mse_and_psnr(gt, pr):
    mse = ((gt - pr)**2).mean()
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr, mse