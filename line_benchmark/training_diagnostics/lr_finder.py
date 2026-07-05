"""LR range test (Smith, arXiv:1506.01186) on ultralytics trainers.

LR grows exponentially per optimizer step from lr_min to lr_max while the
smoothed loss is recorded; the run stops once the loss exceeds 4x its
minimum (divergence) or after n_steps. Suggestion: lr at the smoothed-loss
minimum divided by 10.

warmup_epochs MUST be 0 when running this - the ultralytics warmup writes
its own LR every batch and would silently overwrite the sweep.
"""

import json
from pathlib import Path

from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.yolo.detect import DetectionTrainer


def lr_growth_factor(lr_min, lr_max, n_steps):
    return (lr_max / lr_min) ** (1.0 / n_steps)


def suggest_lr(lrs, losses, smooth=0.7):
    """LR at the smoothed-loss minimum / 10 (order-of-magnitude safety)."""
    if len(lrs) < 2:
        return None
    sm, smoothed = None, []
    for v in losses:
        sm = v if sm is None else smooth * sm + (1 - smooth) * v
        smoothed.append(sm)
    i_min = min(range(len(smoothed)), key=smoothed.__getitem__)
    return lrs[i_min] / 10.0


class LRFinderMixin:
    # configured on the class before model.train(trainer=...)
    lr_min = 1e-6
    lr_max = 1e-1
    n_steps = 200
    out_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lrf_step = 0
        self._growth = lr_growth_factor(self.lr_min, self.lr_max, self.n_steps)
        self._lrs, self._losses = [], []
        self._sm = None
        self._sm_min = None

    def optimizer_step(self):
        # absolute LR from the step index, set right before the step:
        # the per-epoch LambdaLR scheduler resets lr to lr0*lf(epoch) at
        # every epoch boundary, so relative multiplication would sawtooth
        cur_lr = self.lr_min * self._growth ** self._lrf_step
        for pg in self.optimizer.param_groups:
            pg["lr"] = cur_lr
        super().optimizer_step()

        loss = float(self.loss.detach().sum())
        self._lrs.append(cur_lr)
        self._losses.append(loss)
        self._sm = loss if self._sm is None else 0.7 * self._sm + 0.3 * loss
        self._sm_min = (self._sm if self._sm_min is None
                        else min(self._sm_min, self._sm))

        diverged = self._lrf_step > 10 and self._sm > 4 * self._sm_min
        if diverged or self._lrf_step >= self.n_steps:
            self._finish()
            self.stop = True            # honored by the ultralytics loop
        self._lrf_step += 1

    def _finish(self):
        if self.out_path is None:
            return
        out = {"lrs": self._lrs, "losses": self._losses,
               "suggested_lr": suggest_lr(self._lrs, self._losses)}
        path = Path(self.out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        lr = out["suggested_lr"]
        print(f"lr-find: suggested_lr={lr:.2e} ({len(self._lrs)} steps) "
              f"-> {path}" if lr else f"lr-find: too few steps -> {path}")


class LRFinderDetectionTrainer(LRFinderMixin, DetectionTrainer):
    pass


class LRFinderRTDETRTrainer(LRFinderMixin, RTDETRTrainer):
    pass


def pick_lr_finder(weights):
    name = Path(weights).name.lower()
    return (LRFinderRTDETRTrainer if "rtdetr" in name
            else LRFinderDetectionTrainer)
