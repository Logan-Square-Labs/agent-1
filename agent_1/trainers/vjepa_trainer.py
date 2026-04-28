import torch
import lightning as L

from agent_1.models.vjepa.vjepa import VJEPA, vjepa_loss


class LitVJEPA(L.LightningModule):
    def __init__(self, model: VJEPA, config):
        super().__init__()
        self.config = config
        self.model = model

    def configure_model(self) -> None:
        if self.config.compile:
            self.model.encoder = torch.compile(self.model.encoder)
            self.model.predictor = torch.compile(self.model.predictor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_batch_end(self, *_) -> None:
        m = self._momentum()
        self.model.update_target(m)
        self.log("train/ema", m)

    def validation_step(self, batch, batch_idx) -> None:
        self.log("val/loss", self._loss(batch), prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            {"params": self.model.encoder.parameters(), "lr": self.config.encoder_lr},
            {"params": self.model.predictor.parameters(), "lr": self.config.predictor_lr},
        ])
        if self.config.use_lr_schedule:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.config.training_steps
            )
            return {"optimizer": opt, "lr_scheduler": sched}
        return opt

    def _loss(self, batch) -> torch.Tensor:
        zs, hs = self.model(batch["video"], batch["masks_enc"], batch["masks_pred"])
        return vjepa_loss(zs, hs, self.config.loss_exp)

    def _momentum(self) -> float:
        s = min(self.global_step, self.config.training_steps)
        a, b = self.config.ema_start, self.config.ema_end
        return a + s * (b - a) / max(self.config.training_steps, 1)
