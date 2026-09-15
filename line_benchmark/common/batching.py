"""How a batch that fits on the card becomes the batch the optimizer sees.

One definition for every framework: ultralytics applies this rule itself, the
others are made to follow it, and the numbers land in run_meta.json and the
wandb config so a run states its effective batch instead of leaving it to be
recomputed from two other fields.
"""


def accumulate_steps(nbs, batch):
    """Chunks per optimizer step."""
    return max(round(nbs / batch), 1)


def effective_batch(nbs, batch):
    return batch * accumulate_steps(nbs, batch)


def batching_meta(nbs, batch):
    return {"batch": int(batch),
            "nbs": int(nbs),
            "accumulate": accumulate_steps(nbs, batch),
            "effective_batch": effective_batch(nbs, batch)}


def describe(nbs, batch):
    accum = accumulate_steps(nbs, batch)
    return (f"batch {batch} x {accum} chunks = {batch * accum} images per "
            f"optimizer step")
