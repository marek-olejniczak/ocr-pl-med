"""Wall time per epoch, measured the same way for every framework.

The cost table and the case for training kraken on a subsample rather than on
all 14k pages both rest on these numbers, so they come from one place instead
of from each framework's own progress output.

For the ultralytics models an epoch also carries the per-epoch line-metric
validation this benchmark adds on top of the framework's own; detectron2 and
kraken have no such callback, so cross-framework numbers are biased slightly
against ultralytics, not for it.
"""

import time


def epoch_cost(total_s, epochs, pages):
    """Averages from a run we could not hook into, e.g. a ketos subprocess."""
    per_epoch = total_s / epochs if epochs else 0.0
    return {"total_s": float(total_s),
            "epochs": int(epochs),
            "n_pages": int(pages),
            "s_per_epoch": per_epoch,
            "s_per_page_per_epoch": per_epoch / pages if pages else 0.0}


class EpochTimer:
    """Seconds between epoch ends. The first one carries the dataset scan and
    warmup, so it is the odd one out - kept, not smoothed away."""

    def __init__(self, sink=None, key="time/epoch_s"):
        self.sink = sink
        self.key = key
        self.times = []
        self._last = time.perf_counter()

    def tick(self, epoch):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        self.times.append(dt)
        if self.sink:
            self.sink({"epoch": int(epoch), self.key: dt,
                       "time/elapsed_s": float(sum(self.times))})
        return dt

    def __call__(self, trainer):
        """Ultralytics on_fit_epoch_end signature."""
        self.tick(getattr(trainer, "epoch", len(self.times)) + 1)

    def summary(self, pages=None):
        if not self.times:
            return {}
        # the first epoch is warmup-heavy; the steady-state rate is what a
        # projection onto a bigger training set should use
        steady = self.times[1:] or self.times
        per_epoch = sum(steady) / len(steady)
        out = {"total_s": float(sum(self.times)),
               "epochs": len(self.times),
               "s_per_epoch_first": self.times[0],
               "s_per_epoch": per_epoch}
        if pages:
            out["n_pages"] = int(pages)
            out["s_per_page_per_epoch"] = per_epoch / pages
        return out
