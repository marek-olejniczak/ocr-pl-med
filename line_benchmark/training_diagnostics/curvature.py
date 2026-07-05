"""Curvature diagnostics on a fixed probe batch.

directional_curvature: forward-only second difference - robust everywhere.
lambda_max: power iteration on Hessian-vector products (double backward);
returns None when the loss graph cannot support it (caller disables it).
"""

import torch


def _split_like(vec, params):
    out, i = [], 0
    for p in params:
        out.append(vec[i : i + p.numel()].view_as(p))
        i += p.numel()
    return out


def _perturb(params, chunks, eps):
    with torch.no_grad():
        for p, d in zip(params, chunks):
            p.add_(d, alpha=eps)


def directional_curvature(loss_fn, parameters, direction, eps=1e-2):
    """(L(w+ed) - 2L(w) + L(w-ed)) / e^2 ~ d^T H d along unit d. Forward-only.

    eps trades truncation error (large eps) against float32 cancellation
    (small eps): the second-difference signal scales with eps^2 while
    subtraction noise stays ~1e-7 relative to the loss value. 1e-2 keeps
    the signal ~100x above the noise for typical loss magnitudes.
    """
    params = [p for p in parameters if p.requires_grad]
    n = torch.linalg.vector_norm(direction)
    if n == 0:
        return 0.0
    chunks = _split_like(direction / n, params)
    with torch.no_grad():
        l0 = float(loss_fn())
        _perturb(params, chunks, eps)
        lp = float(loss_fn())
        _perturb(params, chunks, -2 * eps)
        lm = float(loss_fn())
        _perturb(params, chunks, eps)  # restore exactly
    return (lp - 2 * l0 + lm) / (eps**2)


def dominant_eig(loss_fn, parameters, iters=8, seed=0):
    """Dominant (largest-magnitude) Hessian eigenvalue via power iteration.

    Sign matters: negative dominant curvature means saddle-dominated
    geometry (typical early in training); positive and growing means the
    minimum is sharpening. NOT the algebraic max - power iteration finds
    the eigenvalue of largest |value|.
    """
    params = [p for p in parameters if p.requires_grad]
    try:
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g = torch.cat([x.flatten() for x in grads])
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(g.numel(), generator=gen).to(g.device)
        v = v / torch.linalg.vector_norm(v)
        lam = 0.0
        for _ in range(iters):
            hv = torch.autograd.grad(torch.dot(g, v), params, retain_graph=True)
            hv = torch.cat([x.flatten() for x in hv]).detach()
            nv = torch.linalg.vector_norm(hv)
            if nv == 0:
                return 0.0
            lam = float(torch.dot(v, hv))
            v = hv / nv
        return lam
    except RuntimeError:
        return None
