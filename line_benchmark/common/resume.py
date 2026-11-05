"""Predictions kept on disk as they are made, so a killed predict resumes.

Every CLI used to collect results in memory and write predictions.json once at
the end, which meant a Ctrl-C threw the whole job away - six hours of it for
kraken at 7.3 s a page. One line per image goes into predictions.jsonl instead,
and a restart skips the images already in there.

The jsonl is scratch: it is removed once predictions.json is written, so a
finished job never resumes into a rerun that was meant to start over.
"""

import json
from pathlib import Path


class PredictionLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if self.path.exists():
            self._read()

    def _read(self):
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break        # half-written tail from the kill; stop here
                self.records[rec["image_id"]] = rec

    def __len__(self):
        return len(self.records)

    def done(self, image_id):
        return image_id in self.records

    def add(self, image_id, preds, ms):
        rec = {"image_id": int(image_id), "preds": list(preds),
               "ms": float(ms)}
        self.records[rec["image_id"]] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def predictions(self):
        out = []
        for rec in self.records.values():
            out.extend(rec["preds"])
        return out

    def speeds(self):
        return [rec["ms"] for rec in self.records.values()]

    def finish(self, out_path):
        preds = self.predictions()
        Path(out_path).write_text(json.dumps(preds))
        self.path.unlink(missing_ok=True)
        return preds
