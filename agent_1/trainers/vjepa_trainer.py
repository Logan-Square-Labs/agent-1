import torch
import lightning as L

from agent_1.models.vjepa.vjepa import VJEPA


class LitVJEPA(L.LightningModule):
    def __init__(self, model: VJEPA, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.model = model

    def configure_model(self) -> None:
        if self.config.compile:
            self.model.encoder = torch.compile(self.model.encoder)
            self.model.predictor = torch.compile(self.model.predictor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(x)

    def training_step(self, batch, batch_idx) -> None:
        loss = self._loss(batch)

        enc_opt, pred_opt = self.optimizers()
        enc_opt.zero_grad()
        pred_opt.zero_grad()
        self.manual_backward(loss)
        enc_opt.step()
        pred_opt.step()

        m = self._momentum()
        self.model.update_target(m)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/ema", m)

    def validation_step(self, batch, batch_idx) -> None:
        self.log("val/loss", self._loss(batch), prog_bar=True)

    def configure_optimizers(self):
        enc_opt = torch.optim.Muon(
            self.model.encoder.parameters(),
            lr=self.config.encoder_lr,
        )
        pred_opt = torch.optim.Muon(
            self.model.predictor.parameters(),
            lr=self.config.predictor_lr,
        )

        if self.config.use_lr_schedule:
            enc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                enc_opt, T_max=self.config.training_steps
            )
            pred_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                pred_opt, T_max=self.config.training_steps
            )
            return (
                {"optimizer": enc_opt, "lr_scheduler": enc_sched},
                {"optimizer": pred_opt, "lr_scheduler": pred_sched},
            )

        return ({"optimizer": enc_opt}, {"optimizer": pred_opt})

    def _loss(self, batch) -> torch.Tensor:
        zs, hs = self.model(batch["video"], batch["masks_enc"], batch["masks_pred"])
        p = self.config.loss_exp
        return sum((z - h).abs().pow(p).mean() / p for z, h in zip(zs, hs)) / len(zs)

    def _momentum(self) -> float:
        s = min(self.global_step, self.config.training_steps)
        a, b = self.config.ema_start, self.config.ema_end
        return a + s * (b - a) / max(self.config.training_steps, 1)
