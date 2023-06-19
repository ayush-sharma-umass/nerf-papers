import torch
import torch.nn as nn

class Nerf(nn.Module):

    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256, device='cpu'):
        super(Nerf, self).__init__()
        # Pos is x,y,z --> each is encoded 10 times with a sin & cosine value each. so 10* 2* 3
        # Then we also concatenate the actual values of x, y, z
        # So input becomes Lpos * 2* 3 + 3
        self.block1 = nn.Sequential(nn.Linear(Lpos * 2 * 3 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    ).to(device)
        self.block2 = nn.Sequential(nn.Linear(hidden_dim + Lpos * 2 * 3 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), nn.ReLU(),
                                    # This last 1 value is added for sigma
                                    ).to(device)
        self.rgb_head = nn.Sequential(nn.Linear(hidden_dim + Ldir * 2 * 3 + 3, hidden_dim // 2), nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),
                                      ).to(device)
        self.Lpos = Lpos
        self.Ldir = Ldir
        self.hidden_dim = hidden_dim
        self.device = device

    def positional_encoding(self, xyz, L):
        indxs = torch.arange(L).to(device=self.device)
        indxs = 2 ** indxs
        # indxs = indxs * torch.pi # They mention in the paper, but in implementation, they remove this pi
        base = xyz.unsqueeze(1) * indxs.unsqueeze(-1)
        embed1 = torch.sin(base).reshape(-1, L * 3)
        embed2 = torch.cos(base).reshape(-1, L * 3)
        pe = torch.cat([embed1, embed2, xyz], dim=1)
        del embed1
        del embed2
        return pe

    # def positional_encoding(self, x, L):
    #     out = [x]
    #     for j in range(L):
    #         out.append(torch.sin(2 ** j * x))
    #         out.append(torch.cos(2 ** j * x))
    #     return torch.cat(out, dim=1)

    def forward(self, xyz_p, xyz_d):
        """
        xyz_p: ray points
        xyz_d: ray_directions
        """
        if xyz_p.device != self.device:
            xyz_p = xyz_p.to(self.device)
        if xyz_d.device != self.device:
            xyz_d = xyz_d.to(self.device)
        B = xyz_p.shape[0]  # Batch size

        ## getting positional embeddings
        x_emb = self.positional_encoding(xyz_p, self.Lpos)  # B, Lpos* 6 + 3
        d_emb = self.positional_encoding(xyz_d, self.Ldir)  # B, Ldir * 6 + 3

        # run the Neural net
        h = self.block1(x_emb)  # B, hidden
        h = self.block2(torch.cat([h, x_emb], dim=1))  # B, hidden + 1
        density = h[:, -1]
        h = h[:, :-1]
        colors = self.rgb_head(torch.cat([h, d_emb], dim=1))  # B, 3
        return colors, torch.relu(density)

    def intersect(self, xyz_p, xyz_d):
        return self.forward(xyz_p, xyz_d)

# nerf = Nerf(Lpos=10, Ldir=4, hidden_dim=256, device=device)
