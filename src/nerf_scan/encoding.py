import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, input_dims: int = 3, num_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        freq_bands = torch.arange(1, num_freqs + 1, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)
        self.out_dim = input_dims * (int(include_input) + 2 * num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x] if self.include_input else []
        for freq in self.freq_bands:
            outs.append(torch.sin(x * freq))
            outs.append(torch.cos(x * freq))
        return torch.cat(outs, dim=-1)
