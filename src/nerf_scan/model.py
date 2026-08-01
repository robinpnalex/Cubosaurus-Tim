import torch
import torch.nn as nn


class TinyNeRFModel(nn.Module):
    def __init__(self, pos_enc_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.rgb_head = nn.Linear(hidden, 3)
        self.sigma_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        rgb = torch.sigmoid(self.rgb_head(h))
        sigma = torch.relu(self.sigma_head(h))
        return rgb, sigma.squeeze(-1)
