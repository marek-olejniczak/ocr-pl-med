"""Framework-agnostic training-diagnostics primitives (pure torch)."""

import json
from pathlib import Path

import torch


def flat_grads(parameters):
    gs = [p.grad.detach().flatten() for p in parameters if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(0)


def flat_params(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


def cosine(a, b):
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return 0.0
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def momentum_stats(optimizer):
    """First-moment stats (AdamW exp_avg / SGD momentum_buffer) vs current grads.

    Momentum and grads are collected in the SAME loop so both flat vectors
    share parameter order - mixing iteration orders silently breaks cosine.
    Call before optimizer.zero_grad().
    """
    bufs, grads = [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            buf = state.get("exp_avg", state.get("momentum_buffer"))
            if buf is not None and p.grad is not None:
                bufs.append(buf.detach().flatten())
                grads.append(p.grad.detach().flatten())
    if not bufs:
        return {"momentum_norm": 0.0, "momentum_grad_cosine": 0.0}
    m = torch.cat(bufs)
    return {
        "momentum_norm": float(torch.linalg.vector_norm(m)),
        "momentum_grad_cosine": cosine(m, torch.cat(grads)),
    }


def update_stats(prev_flat_params, parameters):
    """Norm of the actual weight step and the update-to-weight ratio
    (healthy training sits around ~1e-3)."""
    cur = flat_params(parameters)
    dw = cur - prev_flat_params
    un = torch.linalg.vector_norm(dw)
    wn = torch.linalg.vector_norm(cur)
    return {
        "update_norm": float(un),
        "update_weight_ratio": float(un / wn) if wn > 0 else 0.0,
    }


class Ema:
    """Exponential moving average over logged scalars (raw values stay too)."""

    def __init__(self, decay=0.98):
        self.decay, self.state = decay, {}

    def update(self, values):
        out = {}
        for k, v in values.items():
            if not isinstance(v, (int, float)):
                continue
            prev = self.state.get(k, float(v))
            self.state[k] = self.decay * prev + (1 - self.decay) * float(v)
            out[f"{k}_ema"] = self.state[k]
        return out


class JsonlSink:
    """Append-only JSONL writer; one record per diagnostic step."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record):
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
