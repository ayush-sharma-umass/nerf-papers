import torch
import torch.nn as nn


class MultiResHashEncoding(nn.Module):

    def __init__(self, L=16, T=2 ** 14, F=2, Nmin=16, Nmax=512, device='cpu'):
        super(MultiResHashEncoding, self).__init__()
        self.growth_factor = np.exp((np.log(Nmax) - np.log(Nmin)) / (L - 1))
        self.levels = torch.arange(0, L, device=device)
        self.levels = Nmin * self.growth_factor ** self.levels
        self.hash_tables = nn.Parameter(torch.zeros(L, T, F), requires_grad=True).to(device)  # L, T, F

        self.L = L
        self.T = T
        self.F = F
        self.Nmin = Nmin
        self.Nmax = Nmax
        self.device = device

        # Paper uses three prime numbers for 3 dimesnional points
        self.primes = torch.tensor([1, 2654435761, 805459861]).to(int).to(device)

        # The hash table is initialized using uniform in -10e-4, 10e-4
        for param in self.hash_tables:
            if isinstance(param, nn.Parameter):
                nn.init.uniform_(param, -10e-4, 10e-4)

    def forward(self, xyz_p):
        """
        xyz_p: ray points
        xyz_d: ray_directions
        """
        # tic = time.time()
        xyz_en = self.get_multires_hash_encoding(xyz_p)
        # toc = time.time()
        # print(f"MHRE time: {(toc - tic)* 1000} ms")
        return xyz_en
        pass

    def get_multires_hash_encoding(self, pts):
        BS = pts.shape[0]
        pos_embed = torch.zeros((L, BS, 2), device=self.device)
        level_encodings = []
        for level in range(self.L):
            # scale the points by the growth
            pL = self.levels[level] * pts  # Bs, 3

            # Find the floor and ceil for the points
            fpL = torch.floor(pL)  # Bs, 3
            cpL = torch.ceil(pL)  # Bs, 3

            # Find 8 neighbours of the point in 3D grid
            nbrs = self.get_neighbours(fpL, cpL)  # Nbrs: Bs, 8, 3
            # print(f"neigbors: {nbrs.shape}")

            # find distance from each of these neighbours
            dist = (nbrs - pts.unsqueeze(1)).norm(dim=-1)  # Dist: Bs, 8
            # print(f"dist: {dist.shape}")

            # Get the hashed value of the neighbour
            hashed_nbrs = self.hash_function(nbrs.reshape(-1, 3), level)  # Bs * 8, 2
            hashed_nbrs = hashed_nbrs.reshape(BS, -1, F)  # Bs, 8, 2
            # print(f"hashed_neighbour: {hashed_nbrs.shape}")

            # Interoloate the coordinates. Using vanilla interpolation
            interpolated_coords = (hashed_nbrs * dist.unsqueeze(-1)).sum(1)
            # print(interpolated_coords.shape)

            pos_embed[level] = interpolated_coords

        pos_embed = pos_embed.transpose(0, 1).reshape(BS, -1)
        return pos_embed

    def get_neighbours(self, fpL, cpL):
        neighbours = torch.zeros((fpL.shape[0], 8, 3), device=self.device)
        idxs = torch.tensor([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 0],
                             [0, 1, 1],
                             [1, 0, 0],
                             [1, 0, 1],
                             [1, 1, 0],
                             [1, 1, 1],
                             ], device=self.device)
        # Expand the dimensions of fpL and cpL to match the shape of idxs
        expanded_fpL = fpL[:, None, :]
        expanded_cpL = cpL[:, None, :]

        # Calculate the neighboring points using fpL, cpL, and idxs
        neighbor_coords = expanded_fpL + idxs * (expanded_cpL - expanded_fpL)
        return neighbor_coords.to(int)

    def hash_function(self, key, level):  # N, 3
        # Hash function implementation
        hashed_key = key * self.primes
        hashed_key = torch.bitwise_xor(hashed_key[:, 0], torch.bitwise_xor(hashed_key[:, 1], hashed_key[:, 2])) % self.T
        result = self.hash_tables[level][hashed_key]
        return result

# L = 16
# T = 2 ** 14
# F = 2
# Nmin = 16
# Nmax = 512
# pe = MultiResHashEncoding(L, T, F, Nmin, Nmax)
# pts = torch.randn((1024 * 100,3))
# pe(pts).shape




