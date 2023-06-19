import torch
import torch.nn as nn
import numpy as np


def find_t0_for_image(o, d, verbose=True):
    """
    o is point origins and d is point shape per image.
    This func gives statistic on how far the origin from this point
    """
    eps = 1e-6
    tx = -o[:,0]/(d[:,0] + eps)
    ty = -o[:,1]/(d[:,1] + eps)
    tz = -o[:,2]/(d[:,2] + eps)

    T = np.concatenate([tx, ty, tz])
    if verbose:
        print(f"Tx:: m: {tx.min(): .3f}  | M: {tx.max(): .3f}  | mu: {tx.mean(): .3f}")
        print(f"Ty:: m: {ty.min(): .3f}  | M: {ty.max(): .3f}  | mu: {ty.mean(): .3f}")
        print(f"Tz:: m: {tz.min(): .3f}  | M: {tz.max(): .3f}  | mu: {tz.mean(): .3f}")
        print(f"T:: m: {T.min(): .3f}  | M: {T.max(): .3f}  | mu: {T.mean(): .3f}")
    return T


def accumulated_transmittance(alphas):
    """
    beta: Npts, NBins
    """
    T = torch.cumprod(alphas, axis=1)
    return torch.cat((torch.ones((T.shape[0], 1), device=alphas.device),
                      T[:, :-1]), dim=-1)


def render(model: nn.Module,
           rays_o: np.ndarray,
           rays_d: np.ndarray,
           tn: float,
           tf: float,
           num_bins=100,
           device='cpu',
           directional_input=True,
           white_bgr=True):
    """
    Assigns color values to each ray
    :param model: A model to represent the 3D shape
    :param rays_o: Origin of rays coming from camera
    :param rays_d: Direction of rays coming from camera
    :param tn: initial query time
    :param tf: final query time. We will integreate bw tn, tf
    :param num_bins: the num of bins to divide this query window in
    :param device: {cuda, cpu}
    :param white_bckgr: if the image has a white background
    :return:
    """
    # rays_o: Npt, 3
    # rays_d: Npt, 3
    num_points = rays_o.shape[0]
    if isinstance(rays_o, np.ndarray):
        rays_o = torch.from_numpy(rays_o).float().to(device=device)
    if isinstance(rays_d, np.ndarray):
        rays_d = torch.from_numpy(rays_d).float().to(device=device)
    t = torch.linspace(tn, tf, num_bins).to(device=device)  # NBins
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))  # NBins,
    r = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)
    if directional_input:
        colors, density = model.intersect(r.reshape(-1, 3),
                                      rays_d.expand(num_bins, num_points, 3).transpose(0,1).reshape(-1, 3))
    else:
        colors, density = model.intersect(r.reshape(-1, 3))

    density = density.reshape(num_points, num_bins)  # density: Npts, Nbins
    colors = colors.reshape(num_points, num_bins, 3)  # colors: Npts , Nbins, 3
    alpha = 1. - torch.exp(- density * delta.unsqueeze(0))  # Npts, Nbins
    weights = accumulated_transmittance(1 - alpha) * alpha  # [nb_rays, nb_bins]

    if white_bgr:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
        weight_sum = weights.sum(-1)  # [nb_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [nb_rays, 3]
    return c


def render_with_perturbation(model: nn.Module,
           rays_o: np.ndarray,
           rays_d: np.ndarray,
           tn: float,
           tf: float,
           num_bins=100,
           device='cpu',
           directional_input=True,
           white_bgr=True):
    """
    Assigns color values to each ray
    :param model: A model to represent the 3D shape
    :param rays_o: Origin of rays coming from camera
    :param rays_d: Direction of rays coming from camera
    :param tn: initial query time
    :param tf: final query time. We will integreate bw tn, tf
    :param num_bins: the num of bins to divide this query window in
    :param device: {cuda, cpu}
    :param white_bckgr: if the image has a white background
    :return:
    """

    num_points = rays_o.shape[0]
    if isinstance(rays_o, np.ndarray):
        rays_o = torch.from_numpy(rays_o).float().to(device=device)
    if isinstance(rays_d, np.ndarray):
        rays_d = torch.from_numpy(rays_d).float().to(device=device)
    t = torch.linspace(tn, tf, num_bins).to(device=device).expand(rays_o.shape[0], num_bins)  # Npts, nbins
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(rays_o.shape[0], 1)), -1)

    r = rays_o.unsqueeze(1) + t.unsqueeze(2) * rays_d.unsqueeze(1)  # [batch_size, nb_bins, 3]
    if directional_input:
        colors, density = model.intersect(r.reshape(-1, 3),
                                          rays_d.expand(num_bins, num_points, 3).transpose(0, 1).reshape(-1, 3))
    else:
        colors, density = model.intersect(r.reshape(-1, 3))

    density = density.reshape(num_points, num_bins)  # density: Npts, Nbins
    colors = colors.reshape(num_points, num_bins, 3)  # colors: Npts , Nbins, 3
    alpha = 1. - torch.exp(- density * delta)  # Npts, Nbins
    weights = accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)

    if white_bgr:
        c = (weights * colors).sum(1)  # [nb_rays, 3]
        weight_sum = weights.sum(-1).sum(-1)  # [nb_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        c = (weights * colors).sum(1)  # [nb_rays, 3]
    return c


def batchify_image_and_render(model: nn.Module,
                              rays_o: np.ndarray,
                              rays_d: np.ndarray,
                              tn: float,
                              tf: float,
                              num_bins: int=100,
                              reduction_factor:int=128,
                              directional_input=True,
                              device='cpu',
                              white_bgr=True) -> np.ndarray:
    """

    :param model: A model to represent the 3D shape
    :param rays_o: Origin of rays coming from camera
    :param rays_d: Direction of rays coming from camera
    :param tn: initial query time
    :param tf: final query time. We will integreate bw tn, tf
    :param reduction_factor: Divides the total points by this factor
    :param num_bins: the num of bins to divide this query window in
    :param device: {cuda, cpu}
    :param white_bckgr: if the image has a white background
    :return:
    """
    with torch.no_grad():
        npts = rays_o.shape[0]
        assert npts % reduction_factor == 0
        batch_o = rays_o.reshape(-1, npts//reduction_factor, 3)
        batch_d = rays_d.reshape(-1, npts//reduction_factor, 3)
        if isinstance(rays_o, np.ndarray):
            Ax = np.zeros_like(batch_o)
        else:
            Ax = torch.zeros_like(batch_o)
        for i in range(batch_o.shape[0]):
            # don't call render with perturbation while test
            Ax[i] = render(model,
                           batch_o[i],
                           batch_d[i],
                           tn=tn,
                           tf=tf,
                           num_bins=num_bins,
                           device=device,
                           directional_input=directional_input,
                           white_bgr=white_bgr).detach().cpu()
        if isinstance(Ax, np.ndarray):
            return Ax.reshape(-1, 3)
        return Ax.reshape(-1, 3).cpu().numpy()