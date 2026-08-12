"""Overleaf-ready tables for end-to-end runs."""

import math

_COLUMNS = [
    ("CER", lambda s: s["metrics"]["document"].get("cer")),
    ("WER", lambda s: s["metrics"]["document"].get("wer")),
    ("EMA", lambda s: s["metrics"]["document"].get("ema")),
    ("LineCER", lambda s: s["metrics"]["matched_lines"].get("cer")),
    ("DetF1", lambda s: s["metrics"]["detection"].get("f1")),
    ("AvgTimeS", lambda s: (s.get("timing") or {})
        .get("prediction_seconds_per_page")),
]


def _escape(value):
    if value is None:
        return ""
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def _num(value, decimals=4):
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(num):
        return "n/a"
    return f"{num:.{decimals}f}"


def e2e_table(rows, dataset_id):
    """rows: [(detector_id, ocr_id, summary_dict), ...]"""
    headers = ["Detektor", "OCR"] + [name for name, _ in _COLUMNS]
    alignment = "ll" + "r" * len(_COLUMNS)
    caption = _escape(f"Wyniki end-to-end dla datasetu {dataset_id}")
    label = f"tab:e2e_{dataset_id}"

    lines = [r"\begin{table}[ht]",
             r"\centering",
             f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{alignment}}}",
             r"\hline",
             " & ".join(headers) + r" \\",
             r"\hline"]
    for detector_id, ocr_id, summary in rows:
        row = [_escape(detector_id), _escape(ocr_id)]
        row += [_num(getter(summary)) for _, getter in _COLUMNS]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)
