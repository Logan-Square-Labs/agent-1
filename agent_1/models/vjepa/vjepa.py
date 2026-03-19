import torch
from agent_1.models.utils.modules import MaskGenerator, PatchEmbed, ViT


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x

class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class VJEPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        self.mask_generator = MaskGenerator()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask_generator(x)
        return self.encoder(x * mask), self.predictor(self.encoder(x, mask))