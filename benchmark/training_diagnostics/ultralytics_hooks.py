"""Diagnostics wiring for ultralytics trainers.

BaseTrainer.optimizer_step has no callback inside and gradients are already
zeroed at on_train_batch_end (verified on 8.4.61), so the mixin REPLICATES
the stock method with measurement points. A drift-guard test asserts the
upstream source still matches the replicated logic; ultralytics is pinned
to 8.4.x in requirements.
"""

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect import DetectionTrainer

from training_diagnostics import core, curvature


class DiagnosticMixin:
    # configured on the class before model.train(trainer=...)
    sink = None  # callable(dict)
    probe_every = 100  # 0 disables probe diagnostics
    ema_decay = 0.98

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diag_step = 0
        self._prev_grads = None
        self._prev_probe_grads = None
        self._probe_batch = None
        self._hvp_ok = True
        self._diag_ema = core.Ema(self.ema_decay)

    # --- replicated from BaseTrainer.optimizer_step (ultralytics 8.4.x) ---
    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grads = core.flat_grads(self.model.parameters())
        pre_clip = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=10.0
        )
        prev_w = core.flat_params(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        rec = {
            "step": self._diag_step,
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm_preclip": float(pre_clip),
            "grad_cosine": (
                core.cosine(grads, self._prev_grads)
                if self._prev_grads is not None
                else 0.0
            ),
            "amp_scale": float(self.scaler.get_scale()),
            **core.momentum_stats(self.optimizer),  # before zero_grad!
            **core.update_stats(prev_w, self.model.parameters()),
        }
        self._prev_grads = grads
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        # --- end of replicated logic ---

        if self.probe_every and self._diag_step % self.probe_every == 0:
            rec.update(self._probe_diagnostics())
        rec.update(self._diag_ema.update(rec))
        if self.sink is not None:
            # diag/ prefix groups these into their own wandb section instead of
            # the catch-all "Charts"; train losses go under their own train/
            out = {f"diag/{k}": v for k, v in rec.items()}
            if getattr(self, "loss_items", None) is not None:
                li = self.label_loss_items(self.loss_items, prefix="train")
                li["train/loss"] = round(sum(li.values()), 5)   # total
                out.update(li)
            self.sink(out)
        self._diag_step += 1

    def _probe_loss(self):
        if self._probe_batch is None:
            raw = next(iter(self.train_loader))
            self._probe_batch = {
                k: v.clone() if torch.is_tensor(v) else v for k, v in raw.items()
            }
        batch = self.preprocess_batch(
            {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in self._probe_batch.items()
            }
        )
        loss, _ = self.model(batch)
        return loss.sum() if loss.dim() else loss

    def _probe_diagnostics(self):
        """Expensive metrics on a frozen probe batch - isolates optimisation
        dynamics from data noise. model.eval() freezes BN running stats
        (the loss path is keyed on dict input, not module mode)."""
        was_training = self.model.training
        self.model.eval()
        out = {}
        try:
            params = [p for p in self.model.parameters() if p.requires_grad]
            loss = self._probe_loss()
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            pg = torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(gs, params)
                ]
            ).detach()
            if self._prev_probe_grads is not None:
                out["probe_grad_cosine"] = core.cosine(pg, self._prev_probe_grads)
            self._prev_probe_grads = pg
            out["probe_curvature"] = curvature.directional_curvature(
                self._probe_loss, params, pg
            )
            if self._hvp_ok:
                lam = curvature.dominant_eig(self._probe_loss, params)
                if lam is None:
                    self._hvp_ok = False  # HVP unsupported - stop trying
                else:
                    out["hessian_dominant_eig"] = lam
            out["hessian_eig_supported"] = float(self._hvp_ok)
        except RuntimeError:
            out["probe_error"] = 1.0
        finally:
            self.model.train(was_training)
        return out


class DiagnosticDetectionTrainer(DiagnosticMixin, DetectionTrainer):
    pass


class DiagnosticRTDETRTrainer(DiagnosticMixin, RTDETRTrainer):
    pass


def pick_trainer(weights):
    from pathlib import Path

    name = Path(weights).name.lower()
    return DiagnosticRTDETRTrainer if "rtdetr" in name else DiagnosticDetectionTrainer
