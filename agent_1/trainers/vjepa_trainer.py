import torch
import torch.nn.functional as F
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
        zs, hs = self.model(batch["video"], batch["masks_enc"], batch["masks_pred"])

        block_losses = [vjepa_loss([z], [h], self.config.loss_exp) for z, h in zip(zs, hs)]
        loss = sum(block_losses) / len(block_losses)
        self.log("train/loss", loss, prog_bar=True)
        for i, l in enumerate(block_losses):
            self.log(f"train/loss_block_{i}", l)

        with torch.no_grad():
            z_cat = torch.cat(zs, dim=1)
            h_cat = torch.cat(hs, dim=1)
            self.log("train/pred_std", z_cat.std(dim=-1).mean())
            self.log("train/target_std", h_cat.std(dim=-1).mean())
            self.log("train/pred_target_cosim", F.cosine_similarity(z_cat, h_cat, dim=-1).mean())

        opt = self.optimizers()
        self.log("train/encoder_lr", opt.param_groups[0]["lr"])
        self.log("train/predictor_lr", opt.param_groups[1]["lr"])

        total_patches = torch.tensor(self.model.predictor.grid_size).prod().item()
        self.log("mask/enc_coverage", batch["masks_enc"][0].shape[1] / total_patches)
        self.log("mask/pred_coverage", batch["masks_pred"][0].shape[1] / total_patches)

        if torch.cuda.is_available():
            self.log("perf/gpu_mem_mb", torch.cuda.max_memory_allocated() / 1e6)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        enc_norm = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), float("inf"))
        pred_norm = torch.nn.utils.clip_grad_norm_(self.model.predictor.parameters(), float("inf"))
        self.log("grad/encoder_norm", enc_norm)
        self.log("grad/predictor_norm", pred_norm)

    def on_train_batch_end(self, *_) -> None:
        m = self._momentum()
        self.model.update_target(m)
        self.log("train/ema", m)

        with torch.no_grad():
            enc_params = torch.cat([p.flatten() for p in self.model.encoder.parameters()])
            tgt_params = torch.cat([p.flatten() for p in self.model.target_encoder.parameters()])
            self.log("train/enc_target_l2", (enc_params - tgt_params).norm())

    def validation_step(self, batch, batch_idx) -> None:
        zs, hs = self.model(batch["video"], batch["masks_enc"], batch["masks_pred"])
        self.log("val/loss", vjepa_loss(zs, hs, self.config.loss_exp), prog_bar=True)

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

    def _momentum(self) -> float:
        ema_steps = self.config.training_steps * self.config.ema_ipe_scale
        s = min(self.global_step, ema_steps)
        a, b = self.config.ema_start, self.config.ema_end
        return a + s * (b - a) / max(ema_steps, 1)
