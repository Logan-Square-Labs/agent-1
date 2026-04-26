import torch
from agent_1.models.utils.modules import PatchEmbed, ViT


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x), self.predict(self.encode(x))