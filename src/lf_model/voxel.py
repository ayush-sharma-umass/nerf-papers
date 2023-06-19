import os, os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import cv2
from typing import List
from torch.utils.data import Dataset


class Voxel(nn.Module):

    def __init__(self, scale=1, num_cells=100, device='cpu'):
        """
        creates a voxel model
        :param scale: the total size of the grid
        :param num_cells: number of voxel cells in one dimension. A voxel grid is N x N x N
        :param device:
        """
        super(Voxel, self).__init__()
        # The grid is centered at 0,0,0 with entire grid length going from [-scale/2, scale/2]
        self.grid = nn.Parameter(torch.rand(size=(num_cells, num_cells, num_cells, 4),
                                            device=device, requires_grad=True))
        self.scale = scale
        self.num_cells = num_cells
        self.device = device

    def forward(self, xyz, d=None):
        """
        Takes set of points and their direction vectors and returns color
        :param xyz: points
        :param d:  direction
        :return:
        """
        xyz = xyz.to(self.device)
        N = xyz.shape[0]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Condition to check if a point inside the grid
        # x.abs() < scale/2 means x can lie between -scale/2, scale/2
        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        cond = cond.to(device=self.device)

        # Each voxel cell has side: self.scale/ self.num_cells
        # But since x can be negative, and to convert it to index, we need to push it to
        # num_cells/2 to index it from 0
        indx = ((x[cond] / (self.scale / self.num_cells)) + self.num_cells / 2).type(torch.long)
        indy = ((y[cond] / (self.scale / self.num_cells)) + self.num_cells / 2).type(torch.long)
        indz = ((z[cond] / (self.scale / self.num_cells)) + self.num_cells / 2).type(torch.long)

        color_and_density = torch.zeros((N, 4), device=self.device)
        color_and_density[cond, :3] = self.grid[indx, indy, indz, :3]
        color_and_density[cond, -1] = self.grid[indx, indy, indz, -1]

        ## For debugging
        # color_and_density[cond, 0:3] = torch.tensor([1., 0., 0.], device=device)
        # color_and_density[cond, -1] = 10.

        # sigmoid on color to keep it to (0,1)
        # and relu to make all negative values of density to 0
        return torch.sigmoid(color_and_density[:, :3]), torch.relu(color_and_density[:, -1:])

    def intersect(self, xyz, d=None):
        return self.forward(xyz, d)