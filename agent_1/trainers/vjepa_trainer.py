import torch
import lightning as L
from typing import Dict
from agent_1.models.vjepa.vjepa import VJEPA


class LitVJEPA(L.LightningModule):
    def __init__(self,  model: VJEPA, config):
        super().__init__()
        self.automatic_optimization = False

        self.config = config
        self.model = model

        self.encoder_lr = config.encoder_lr
        self.predictor_lr = config.predictor_lr

    def configure_model(self) -> None:
        if self.config.compile:
            self.encoder = torch.compile(self.encoder)
            self.predictor = torch.compile(self.predictor)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)

    def training_step(self, batch):
        x = batch["video"]
        self.encoder_train_step(x)
        self.predictor_train_step(x)

    def validation_step(self, batch):
        x = batch["video"]

    def configure_optimizers(self):
        enc_optimizer = torch.optim.Muon(
            self.encoder.parameters(),
            lr=self.encoder_lr
        )

        pred_optimizer = torch.optim.Muon(
            self.predictor.parameters(),
            lr=self.predictor_lr
        )

        if self.config.use_lr_schedule:
            enc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                enc_optimizer, T_max=self.config.training_steps
            )
            pred_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                pred_optimizer, T_max=self.config.training_steps
            )
            return (
                {"optimizer": enc_optimizer, "lr_scheduler": enc_scheduler},
                {"optimizer": pred_optimizer, "lr_scheduler": pred_scheduler},
            )

        return (
            {"optimizer": enc_optimizer},
            {"optimizer": pred_optimizer},
        )

    def encoder_train_step(self, pred_out, x) -> None:
        opt_enc = self.optimizers()[0]

        # TODO: compute encoder loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train/encoder_loss", loss)

        self.toggle_optimizer(opt_enc)
        self.manual_backward(loss)
        opt_enc.step()
        opt_enc.zero_grad()
        self.untoggle_optimizer(opt_enc)

    def predictor_train_step(self, pred_out, x) -> None:
        opt_pred = self.optimizers()[1]

        # TODO: compute predictor loss
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train/predictor_loss", loss)

        self.toggle_optimizer(opt_pred)
        self.manual_backward(loss)
        opt_pred.step()
        opt_pred.zero_grad()
        self.untoggle_optimizer(opt_pred)
