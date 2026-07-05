"""Lightweight process resource stats for model meta.json (no heavy deps).

Captures peak host RAM and peak per-process GPU memory so the benchmark records
cost (footprint) alongside quality and speed. torch is imported lazily so this
also works in containers/paths where torch isn't present.
"""

import resource


def reset_gpu_peak():
    """Reset the CUDA peak-memory counter; call before the work to measure."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def resource_meta():
    """Peak RAM (always) + peak GPU memory and device name (if CUDA was used)."""
    # ru_maxrss is KB on Linux (the container OS)
    out = {"peak_ram_mb": round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.max_memory_allocated() > 0:
            out["peak_gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated() / 1e6, 1)
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out
