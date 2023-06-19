import torch
import torch.nn as nn


class FourierNerf(nn.Module):

    def __init__(self, NF_feats=256, B_scale=6.05, hidden_dim=128, device='cpu'):
        super(FourierNerf, self).__init__()
        self.block = nn.Sequential(nn.Linear(NF_feats * 2, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 4),
                                   ).to(device)

        self.NF_feats = NF_feats
        self.hidden_dim = hidden_dim
        self.device = device
        self.B = torch.randn((3, NF_feats), device=device) * 2 * torch.pi * B_scale

        for _, param in self.block.named_parameters():
            if isinstance(param, nn.Linear):
                nn.init.xavier_uniform_(param.weight)
                if param.bias is not None:
                    nn.init.zeros_(param.bias)

    def positional_encoding(self, xyz):
        Tsin = torch.sin(xyz @ self.B)
        Tcos = torch.cos(xyz @ self.B)
        return torch.cat([Tcos, Tsin], axis=-1)

    def forward(self, xyz_p):
        """
        xyz_p: ray points
        xyz_d: ray_directions
        """
        if xyz_p.device != self.device:
            xyz_p = xyz_p.to(self.device)
        B = xyz_p.shape[0]  # Batch size

        ## getting positional embeddings
        F_embed = self.positional_encoding(xyz_p)  # BS, NF_feat
        # set_trace()

        # run the Neural net
        out = self.block(F_embed)  # BS, 4
        colors = torch.sigmoid(out[:, :3])
        density = torch.relu(out[:, -1:])
        return colors, density

    def intersect(self, xyz_p):
        return self.forward(xyz_p)


# NF_feats = 256
# B_scale = 6.05
# hidden_dim = 128
# device='cuda'
# Fnerf = FourierNerf(NF_feats=NF_feats, B_scale=B_scale, hidden_dim=hidden_dim, device=device)